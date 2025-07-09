#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import shutil
import time
import signal
import fcntl # For non-blocking IO (Unix-like)
import select # For checking file descriptor status
import errno # For handling non-blocking IO errors
import nltk
import torch
from typing import List, Dict, Any, Optional

# Import experiment configurations
from expr_config_global import (
    NAMES, 
    MODEL_CONFIGS, 
    MODEL_PATHS,
    MODEL_TEMPLATES,
    MODEL_LORA_TARGETS,
    DEFAULT_TIMEOUT,
    DEFAULT_CHECK_INTERVAL,
    DEFAULT_MAX_RETRIES
)

def clean_output_dir(output_dir: str) -> None:
    """Clean output directory, keeping only specific files."""
    if os.path.exists(output_dir):
        files_to_keep = ["train_results.json", "training_loss.png", "trainer_log.jsonl"]
        for root, dirs, files in os.walk(output_dir):
            if root == output_dir: # Only process top-level directory
                for f in files:
                    if f not in files_to_keep:
                        try:
                            os.remove(os.path.join(root, f))
                        except Exception as e:
                            print(f"Error deleting file {f}: {e}")
                
                # Delete all subdirectories
                for d in dirs:
                    try:
                        shutil.rmtree(os.path.join(root, d))
                    except Exception as e:
                        print(f"Error deleting directory {d}: {e}")
                break

class TimeoutError(Exception):
    """Timeout exception"""
    pass

def run_with_timeout(cmd, env, timeout=3600, check_interval=60):
    """Command execution function with timeout"""
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffering
        universal_newlines=True, # Ensure text mode and line buffering are effective
        preexec_fn=os.setsid # Create a new process group for easier termination
    )

    start_time = time.time()
    last_output_time = start_time
    output_lines = []

    # Get stdout file descriptor and set to non-blocking
    stdout_fd = process.stdout.fileno()
    flags = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
    fcntl.fcntl(stdout_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    try:
        while True:
            # First check if the process has ended
            return_code = process.poll()
            if return_code is not None:
                # After process ends, try to read remaining output
                while True:
                    try:
                        # Non-blocking read of remaining lines
                        line = process.stdout.readline()
                        if line:
                            output_lines.append(line)
                            yield line
                        else:
                            break # No more output
                    except IOError as e:
                         # EAGAIN/EWOULDBLOCK means no data to read
                        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                            print(f"\nIO error when reading remaining stdout: {e}")
                            raise # Other IO errors
                        break # No data to read
                break # Exit main loop

            current_time = time.time()

            # Check if total running time exceeds timeout
            if timeout and current_time - start_time > timeout:
                print(f"\nProgram ran for more than {timeout} seconds, forcing termination")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait() # Wait for process to actually terminate
                raise TimeoutError(f"Program timeout: ran for more than {timeout} seconds")

            # Check if there has been no output for a while (stuck)
            # Only start checking for stuck process after a certain time to avoid false positives at startup
            if current_time - start_time > check_interval and current_time - last_output_time > check_interval:
                print(f"\nDetected program might be stuck, no output for {check_interval} seconds")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait() # Wait for process to actually terminate
                raise TimeoutError(f"Program stuck: no output for {check_interval} seconds")

            # Use select to check if stdout has data to read, set a short timeout to avoid busy-waiting
            # select timeout set to 1.0 second
            ready_to_read, _, _ = select.select([stdout_fd], [], [], 1.0)

            if ready_to_read:
                while True: # Read as much available data as possible
                    try:
                        line = process.stdout.readline()
                        if line:
                            last_output_time = time.time() # Update time when output was received
                            output_lines.append(line)
                            yield line
                        else:
                            # If readline returns empty string and process is not ended,
                            # this usually means no data for now, but pipe is not closed
                            break
                    except IOError as e:
                         # EAGAIN/EWOULDBLOCK means no data can be read immediately
                        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                            print(f"\nIO error when reading stdout: {e}")
                            raise # Re-raise other IO errors
                        # No more data can be read immediately, break internal read loop
                        break
            # else: select timeout, no new data on stdout in 1 second, continue loop to check timeout and stuck

        # Return result after normal completion (return_code obtained at loop start or first check)
        if return_code is None: # If loop exited due to break but process not ended (theoretically should not happen)
             return_code = process.wait() # Ensure final return code is obtained
        return return_code, output_lines

    except KeyboardInterrupt:
        print("\nUser interrupted, terminating process...")
        try:
            # Ensure SIGTERM is used to terminate the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait()
        except ProcessLookupError:
             print("Process seems to have already ended.") # If process ended before kill
        except Exception as kill_e:
            print(f"Error terminating process: {kill_e}")
        raise # Re-raise KeyboardInterrupt
    except TimeoutError: # Catch our own TimeoutError
        # Ensure process has been terminated (though killpg should have done this)
        try:
            if process.poll() is None:
                 process.wait(timeout=5) # Wait a few seconds for it to end
        except subprocess.TimeoutExpired:
             print("\nWaiting for process termination timed out, manual intervention may be needed.")
        except Exception as wait_e:
             print(f"\nError waiting for process termination: {wait_e}")
        raise # Re-raise TimeoutError to upper caller
    except Exception as e:
        print(f"\nUnexpected exception in run_with_timeout: {e}")
        try:
            # Try to ensure process is terminated
            if process.poll() is None:
                print("\nAttempting to terminate residual process...")
                os.killpg(os.getpgid(process.pid), signal.SIGKILL) # Force terminate
                process.wait()
        except Exception as kill_e:
            print(f"Error terminating process during exception handling: {kill_e}")
        raise # Re-raise original exception

def add_id_to_path(path: str, test_id: Optional[str]) -> str:
    """Add test ID to path"""
    if not test_id:
        return path
    
    # Split path
    dirname, basename = os.path.split(path)
    
    # Add ID
    if basename:
        new_basename = f"{basename}_{test_id}"
        return os.path.join(dirname, new_basename)
    else:
        # If it's a directory path ending with /
        return f"{path}_{test_id}"

def run_experiment(exp_type: str, names: List[str], model_name: str, specific_name: Optional[str] = None, 
                  gpu: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT, max_retries: int = DEFAULT_MAX_RETRIES, 
                  check_interval: int = DEFAULT_CHECK_INTERVAL, test_id: Optional[str] = None) -> None:
    """Run specified type of experiment
    
    Args:
        exp_type: Experiment type
        names: List of names
        model_name: Model name (e.g. "llama3_8b_instruct" or "qwen2_7b_instruct")
        specific_name: Specific name
        gpu: Specify GPU, overrides config value if provided
        timeout: Timeout in seconds
        max_retries: Max retries
        check_interval: Check output interval in seconds
        test_id: Test ID, used to distinguish different experiment groups
    """
    # Get configuration dictionary for the corresponding model
    if model_name not in MODEL_CONFIGS:
        print(f"Error: Unknown model name '{model_name}'")
        return

    model_config = MODEL_CONFIGS[model_name]
    
    if exp_type not in model_config:
        print(f"Error: Model '{model_name}' has no configuration for experiment type '{exp_type}'")
        return
    
    config = model_config[exp_type]
    model_path = MODEL_PATHS[model_name]
    template_name = MODEL_TEMPLATES[model_name]
    lora_target = MODEL_LORA_TARGETS.get(model_name, "q_proj,v_proj")
    
    # If a specific name is provided, only run that one
    if specific_name:
        if specific_name in names:
            names = [specific_name]
        else:
            print(f"Error: Specified name '{specific_name}' not found")
            return
    
    # If test ID is provided, modify output directory and log directory in config
    if test_id:
        # For regular directories, directly add ID suffix
        if "output_dir" in config:
            config["output_dir"] = add_id_to_path(config["output_dir"], test_id)
        if "output_result_dir" in config:
            config["output_result_dir"] = add_id_to_path(config["output_result_dir"], test_id)
        
        # Add ID for log directory
        config["log_dir"] = add_id_to_path(config["log_dir"], test_id)
    
    # Create log directory
    os.makedirs(config["log_dir"], exist_ok=True)
    
    for name in names:
        print(f"Processing: {name}")
        
        # Prepare output directory
        output_dir = config["output_dir"]
        result_dir = config["output_result_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Build command
        cmd = ["python", "src/train_bash.py",
               f"--stage={config['stage']}",
               f"--model_name_or_path={model_path}",
               f"--dataset={name}_{config['dataset_suffix']}",
               "--dataset_dir=./data",
               "--finetuning_type=lora",
               f"--lora_target={lora_target}",
               f"--output_dir={output_dir}",
               "--overwrite_cache",
               "--overwrite_output_dir",
               "--cutoff_len=512",
               "--preprocessing_num_workers=16",
               f"--per_device_train_batch_size={config['batch_size']}",
               f"--per_device_eval_batch_size={config['batch_size']}",
               f"--gradient_accumulation_steps={config['grad_accum_steps']}",
               "--lr_scheduler_type=cosine",
               "--logging_steps=10",
               "--warmup_steps=20",
               "--save_steps=30000",
               "--eval_steps=30000",
               "--evaluation_strategy=steps",
               "--load_best_model_at_end",
               f"--template={template_name}",
               f"--learning_rate={config['learning_rate']}",
               f"--num_train_epochs={config['epochs']}",
               "--val_size=0.0000001",
               "--plot_loss",
               f"--output_result_dir={result_dir}",
               "--fp16",
               f"--eval_dataset_dir={config['eval_dataset_dir']}",
               f"--target={name}",
               "--seed=42"]
        
        # Add or remove specific parameters
        if not config.get("skip_train", False):
            cmd.append("--do_train")

        for arg in config["extra_args"]:
            if arg:  # Skip empty parameters
                cmd.append(arg)
        
        # Set environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = "./"
        env["WANDB_DISABLED"] = "true"
        
        # If GPU is specified in command line, prioritize it
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        elif config["gpu"] is not None:
            env["CUDA_VISIBLE_DEVICES"] = config["gpu"]
        
        # Determine log file
        log_filename = f"{name}"
        if test_id:
            log_filename += f"_{test_id}"
        log_filename += ".log"
        log_file = os.path.join(config["log_dir"], log_filename)
        
        # Retry logic
        success = False
        retry_count = 0
        
        while not success and retry_count <= max_retries:
            if retry_count > 0:
                print(f"Retrying {name} for the {retry_count} time...")
            
            with open(log_file, 'w' if retry_count == 0 else 'a') as f:
                if retry_count > 0:
                    retry_msg = f"\n\n{'='*50}\nRetry {retry_count}\n{'='*50}\n\n"
                    f.write(retry_msg)
                    print(retry_msg)
                
                try:
                    return_code = None
                    for line in run_with_timeout(cmd, env, timeout, check_interval):
                        print(line, end='')
                        f.write(line)
                    
                    # Handle completion
                    success = True
                    print(f"Successfully completed {name}")
                    # Clean output directory
                    clean_output_dir(output_dir)
                    
                except TimeoutError as e:
                    error_msg = f"\nProgram execution interrupted: {str(e)}\n"
                    print(error_msg)
                    f.write(error_msg)
                    retry_count += 1
                    
                except Exception as e:
                    error_msg = f"\nError running command: {str(e)}\n"
                    print(error_msg) 
                    f.write(error_msg)
                    retry_count += 1
        
        if not success:
            print(f"Warning: {name} failed after {max_retries} attempts")

def main():
    # Create GPU memory pool
    parser = argparse.ArgumentParser(description="Run LLaMA-Factory LoRA Experiment")
    parser.add_argument("--type", type=str, required=True, choices=[key for model_config in MODEL_CONFIGS.values() for key in model_config.keys()], 
                        help="Experiment type")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()), default="llama3",
                        help="Model name")
    parser.add_argument("--name", type=str, help="Only run experiment for specified name")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of names to process")
    parser.add_argument("--end_idx", type=int, default=50, help="End index of names to process")
    parser.add_argument("--gpu", type=str, help="GPU ID(s) to use, e.g. '0' or '0,1'")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout for a single task (seconds)")
    parser.add_argument("--check_interval", type=int, default=DEFAULT_CHECK_INTERVAL, help="Interval (seconds) to check if program is stuck")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES, help="Maximum number of retries if task fails or gets stuck")
    parser.add_argument("--test_id", type=str, help="Test ID, appended to output directory name to distinguish different experiment groups")
    parser.add_argument("--memory_pool", type=float, default=0, help="GPU memory pool size (GiB), 0 means not used")
    
    args = parser.parse_args()
    
    # If memory pool size is specified, create memory pool
    if args.memory_pool > 0:
        try:
            # Calculate bytes to allocate (1 GiB = 1024^3 bytes)
            mem_size = int(args.memory_pool * (1024**3) / 4) # Convert to float32 element count
            print(f"Creating GPU memory pool of size {args.memory_pool} GiB...")
            # Check for available GPU
            if not torch.cuda.is_available():
                print("Warning: No CUDA device found, cannot create GPU memory pool")
                memory_pool = None
            else:
                # Determine which GPU to use
                if args.gpu:
                    device_id = int(args.gpu.split(',')[0]) # Use the first specified GPU
                    device = f"cuda:{device_id}"
                else:
                    device = "cuda:0" # Default to first GPU
                # Create memory pool
                memory_pool = torch.zeros(mem_size, dtype=torch.float32, device=device)
                print(f"Memory pool created successfully on {device}")
        except ImportError:
            print("Warning: torch library not found, cannot create GPU memory pool")
            memory_pool = None
        except Exception as e:
            print(f"Error creating memory pool: {e}")
            memory_pool = None
    
    # Handle specific name or range
    if args.name:
        run_experiment(
            args.type, NAMES, args.model, args.name, 
            args.gpu, args.timeout, args.max_retries, args.check_interval, args.test_id
        )
    else:
        names_to_process = NAMES[args.start_idx:args.end_idx]
        run_experiment(
            args.type, names_to_process, args.model, None,
            args.gpu, args.timeout, args.max_retries, args.check_interval, args.test_id
        )

if __name__ == "__main__":
    # nltk.download('punkt')
    main()

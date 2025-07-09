import math
import os.path
import json
import re
from typing import TYPE_CHECKING, List, Optional
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM
from ...data import get_dataset, split_dataset
from ...extras.ploting import plot_loss
from ...model import load_tokenizer
from ..utils import create_modelcard_and_push

from ...eval import *
from ...eval.eval_ort import *

# rocr related imports
from .rocr.rocr_hparams import rocrHyperParams
from .rocr.rocr_main import apply_rocr_to_model, get_cov
from .util import nethook
from .util.globals import *
import json
import os
import torch

FORGET_LEVEL1 = 'forget_level1.json'
FORGET_LEVEL2 = 'forget_level2.json'
FORGET_LEVEL3 = 'forget_level3.json'
NEIGHBOR_LEVEL1 = 'neighbor_level1.json'
NEIGHBOR_LEVEL2 = 'neighbor_level2.json'
FORGET_MCP = 'forget_mcp.json'
NEIGHBOR_MCP = 'neighbor_mcp.json'

RETAIN_MMLU = 'retain_mmlu.json'
TRUTHFUL = 'truthful.json'
TRIVIAQA = 'triviaqa.json'
FLUENCY = 'fluency.json'


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_rocr(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
    )
    
    # Load rocr hyperparameters
    model_path = model_args.model_name_or_path.lower()
    if "llama" in model_path:
        model_name = "llama3-8b-instruct"
    elif "gpt-j" in model_path:
        model_name = "EleutherAI_gpt-j-6B"
    elif "gpt2" in model_path:
        model_name = "gpt2-xl"
    elif "phi" in model_path:
        model_name = "phi-1.5"
    elif "qwen" in model_path:
        model_name = "qwen2.5-7b-instruct"
    elif "vicuna" in model_path:
        model_name = "vicuna-7b-v1.5"
    elif "mistral" in model_path:
        model_name = "mistral-7b-instruct"
    else:
        model_name = "llama3-8b-instruct" 
    
    print(f"Using hyperparameters for model {model_name}")
    
    os.makedirs(data_args.output_result_dir, exist_ok=True)
    hparams_output_path = os.path.join(data_args.output_result_dir, f"{model_name}_hparams.json")
    hparams_default_path = os.path.join(HPARAMS_DIR, f"{model_name}.json")

    hparams_dict = None

    # 1. Try to load from output_result_dir
    if os.path.exists(hparams_output_path):
        print(f"Attempting to load hparams from output_result_dir: {hparams_output_path}")
        try:
            with open(hparams_output_path, 'r') as f:
                hparams_dict = json.load(f)
            print(f"Successfully loaded hparams from output_result_dir.")
        except Exception as e:
            print(f"Failed to load hparams file '{hparams_output_path}' from output_result_dir: {e}. Attempting to load from HPARAMS_DIR.")
            hparams_dict = None # Ensure hparams_dict is None on load failure
    else:
        print(f"No hparams file found in output_result_dir: {hparams_output_path}. Attempting to load from HPARAMS_DIR.")
    
    # 2. If not found or failed to load from output_result_dir, load from HPARAMS_DIR
    if hparams_dict is None:
        if os.path.exists(hparams_default_path):
            print(f"Attempting to load hparams from HPARAMS_DIR: {hparams_default_path}")
            try:
                with open(hparams_default_path, 'r') as f:
                    hparams_dict = json.load(f)
                print(f"Successfully loaded hparams from HPARAMS_DIR.")
                
                # Since loaded from HPARAMS_DIR, save/update a copy to output_result_dir
                print(f"Hparams loaded from HPARAMS_DIR, now saving/updating to output_result_dir: {hparams_output_path}")
                try:
                    with open(hparams_output_path, 'w') as f:
                        json.dump(hparams_dict, f, indent=2)
                    print(f"Successfully saved/updated hparams to: {hparams_output_path}")
                except Exception as e_save:
                    print(f"Warning: Failed to save/update hparams to {hparams_output_path}: {e_save}")
            except Exception as e_load_default:
                # If HPARAMS_DIR also fails to load
                raise FileNotFoundError(
                    f"Failed to load hparams from HPARAMS_DIR ({hparams_default_path}), previous attempt from output_result_dir ({hparams_output_path}) also failed. Error: {e_load_default}"
                )
        else:
            # If hparams file not found in both locations
            raise FileNotFoundError(
                f"hparams file not found in {hparams_output_path} (output_result_dir) and "
                f"{hparams_default_path} (HPARAMS_DIR)."
            )

    # At this point, hparams_dict should have been loaded, otherwise the above logic would have raised an exception
    if hparams_dict is None:
        # This case should not happen, indicating a logical error or inability to load the file
        raise ValueError("Failed to load hparams_dict. Please check file paths, content, and program logic.")

    hparams = rocrHyperParams(**hparams_dict)
    
    # Create request format
    requests = []
    # Adjust according to dataset format
    subject = data_args.target
    formatted_subject = format_target_name(subject)
    prompt = f"{{}}."
    request = {
        "case_id": subject,
        "subject": formatted_subject,
        "prompt": prompt,
        "target_new": {"str": f"Unfortunately"}
    }
    requests.append(request)
    
    # Run rocr algorithm
    if training_args.do_train:
        # Initialize cache
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out
        
        # Verify layers are continuous
        assert len(hparams.layers) > 0, "layers cannot be empty"
        for i in range(1, len(hparams.layers)):
            assert hparams.layers[i] == hparams.layers[i-1] + 1, f"Layers must be continuous, but gap found: {hparams.layers[i-1]} -> {hparams.layers[i]}"
        
        # Attempt to load precomputed P matrix layer by layer
        model_data_dir = os.path.join(DATA_DIR, model_name)
        os.makedirs(model_data_dir, exist_ok=True)
        
        all_layers_exist = True
        # Check if P matrix files exist for all layers
        for layer in hparams.layers:
            p_filename = f"null_space_project_layer_{layer}.pt"
            p_path = os.path.join(model_data_dir, p_filename)
            if not os.path.exists(p_path):
                all_layers_exist = False
                break
        
        if all_layers_exist:
            print(f"Loading precomputed P matrix layer by layer")
            for i, layer in enumerate(hparams.layers):
                p_filename = f"null_space_project_layer_{layer}.pt"
                p_path = os.path.join(model_data_dir, p_filename)
                print(f"Loading P matrix for layer {layer}: {p_path}")
                P[i,:,:] = torch.load(p_path, map_location="cpu")
        else:
            print(f"Some or all P matrix files do not exist, recomputing and saving layer by layer")
            for i, layer in enumerate(hparams.layers):
                layer_p = get_project(model,tokenizer,layer,hparams)
                P[i,:,:] = layer_p
                # Save P matrix for each layer separately (save independent tensors, not views)
                p_filename = f"null_space_project_layer_{layer}.pt"
                p_path = os.path.join(model_data_dir, p_filename)
                torch.save(layer_p, p_path)
                print(f"Saving P matrix for layer {layer}: {p_path}")
        
        # Apply rocr algorithm
        print(f"Applying rocr algorithm")
        edited_model, _ = apply_rocr_to_model(
            model=model,
            tok=tokenizer,
            requests=requests,
            hparams=hparams,
            cache_c=cache_c,
            P=P
        )
        
    # Evaluation section
    eval_dataset_dir = data_args.eval_dataset_dir
    target = data_args.target
    eval_dataset_dir = os.path.join(eval_dataset_dir, target)

    with open(os.path.join(eval_dataset_dir, FORGET_LEVEL1), 'r') as f:
        forget_level1 = json.load(f)
    with open(os.path.join(eval_dataset_dir, FORGET_LEVEL2), 'r') as f:
        forget_level2 = json.load(f)
    with open(os.path.join(eval_dataset_dir, FORGET_LEVEL3), 'r') as f:
        forget_level3 = json.load(f)
    with open(os.path.join(eval_dataset_dir, NEIGHBOR_LEVEL1), 'r') as f:
        neighbor_level1 = json.load(f)
    with open(os.path.join(eval_dataset_dir, NEIGHBOR_LEVEL2), 'r') as f:
        neighbor_level2 = json.load(f)
    with open(os.path.join(eval_dataset_dir, FORGET_MCP), 'r') as f:
        forget_mcp = json.load(f)
    with open(os.path.join(eval_dataset_dir, NEIGHBOR_MCP), 'r') as f:
        neighbor_mcp = json.load(f)
    with open(os.path.join(eval_dataset_dir, RETAIN_MMLU), 'r') as f:
        retain_mmlu = json.load(f)
    with open(os.path.join(eval_dataset_dir, TRUTHFUL), 'r') as f:
        truthfulqa = json.load(f)
    with open(os.path.join(eval_dataset_dir, TRIVIAQA), 'r') as f:
        triviaqa = json.load(f)
    with open(os.path.join(eval_dataset_dir, FLUENCY), 'r') as f:
        fluency = json.load(f)

    output_result_dir = os.path.join(data_args.output_result_dir, target)
    os.makedirs(os.path.join(output_result_dir), exist_ok=True)

    model.eval()
    with torch.no_grad():
        e_tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, padding_side='left')
        e_tokenizer.pad_token = e_tokenizer.eos_token
        print("Evaluating forgetting...")
        eval_forget(edited_model, e_tokenizer, forget_level1, forget_level2, forget_level3, batch_size=16,
                    output_result_dir=os.path.join(output_result_dir, 'forget.json'), use_prompt=data_args.use_prompt)
        print("Evaluating neighbor...")
        eval_neighbor(edited_model, e_tokenizer, neighbor_level1, neighbor_level2, batch_size=16, output_result_dir=os.path.join(
            output_result_dir, 'neighbor.json'), use_prompt=data_args.use_prompt)
        
        print("Evaluating subtoken forget prob...")
        eval_subtoken_forget_prob(edited_model, e_tokenizer, forget_level1, forget_level2, forget_level3, batch_size=1,
                                  output_result_dir=os.path.join(output_result_dir, 'subtoken_forget_prob.json'), use_prompt=data_args.use_prompt)
        print("Evaluating forget prob...")
        eval_forget_prob(edited_model, e_tokenizer, forget_level1, forget_level2, forget_level3, forget_mcp, batch_size=1,
                         output_result_dir=os.path.join(output_result_dir, 'forget_prob.json'), use_prompt=data_args.use_prompt)
        print("Evaluating neighbor prob...")
        eval_neighbor_prob(edited_model, e_tokenizer, neighbor_level1, neighbor_level2, neighbor_mcp, batch_size=1,
                         output_result_dir=os.path.join(output_result_dir, 'neighbor_prob.json'), use_prompt=data_args.use_prompt)
        print("Evaluating subtoken neighbor prob...")
        eval_subtoken_neighbor_prob(edited_model, e_tokenizer, neighbor_level1, neighbor_level2, batch_size=1,
                                  output_result_dir=os.path.join(output_result_dir, 'subtoken_neighbor_prob.json'), use_prompt=data_args.use_prompt)
        print("Evaluating mmlu...")
        eval_mmlu(edited_model, e_tokenizer, retain_mmlu, batch_size=1, output_result_dir=os.path.join(
            output_result_dir, 'mmlu.json'), use_prompt=data_args.use_prompt)
        print("Evaluating truthful...")
        eval_truthfulqa(edited_model, e_tokenizer, truthfulqa, batch_size=4, output_result_dir=os.path.join(
            output_result_dir, 'truthful.json'), use_prompt=data_args.use_prompt)
        print("Evaluating triviaqa...")
        eval_triviaqa(edited_model, e_tokenizer, triviaqa, batch_size=16, output_result_dir=os.path.join(
            output_result_dir, 'triviaqa.json'), use_prompt=data_args.use_prompt)
        print("Evaluating fluency...")
        eval_fluency(edited_model, e_tokenizer, fluency, batch_size=8, output_result_dir=os.path.join(
            output_result_dir, 'fluency.json'), use_prompt=data_args.use_prompt)


def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T

def format_target_name(target: str) -> str:
    name = re.sub(r'^\d+_', '', target)
    name = name.replace('_', ' ')
    return name
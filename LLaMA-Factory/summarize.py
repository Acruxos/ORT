#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import csv
import pandas as pd
from collections import defaultdict
from typing import Dict, Any, List, Set
from summarize_config import IGNORE_KEYS, SKIP_TASKS, SKIP_METRICS, NAME_MAPPING

# Mapping of model names to their respective result paths
MODEL_PATH_MAPPING = {
    "llama3": "./results/lora/llama3_8b_instruct/",
    "mistral": "./results/lora/mistral_7b_instruct/",
    # Add more models here if needed
}

def load_json_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_metrics(base_dir: str, ignore_keys: Set[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Collects evaluation metrics from all result files.
    
    Args:
        base_dir: The base path of the results folder.
        ignore_keys: A set of keys to ignore.
    
    Returns:
        A dictionary of aggregated evaluation metrics.
    """
    if ignore_keys is None:
        ignore_keys = set(IGNORE_KEYS)
    
    # Dictionary to store all results for each task
    task_metrics = defaultdict(list)
    
    # Iterate through all named folders within the base directory
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)

        
        # Ensure it's a directory
        if not os.path.isdir(dir_path):
            continue
        
        print(f"Processing directory: {dir_name}")
        
        # Iterate through all JSON files in the directory
        for json_file in os.listdir(dir_path):
            if not json_file.endswith('.json'):
                continue
            
            # Get the task name (remove .json suffix)
            task_name = os.path.splitext(json_file)[0]
            file_path = os.path.join(dir_path, json_file)
            
            try:
                # Load JSON data
                data = load_json_file(file_path)
                
                # Filter out keys to ignore
                filtered_data = {k: v for k, v in data.items() if k not in ignore_keys}
                
                # Add results to the corresponding task
                task_metrics[task_name].append(filtered_data)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    # Calculate average metrics for each task
    summary = {}
    for task_name, metrics_list in task_metrics.items():
        # Initialize aggregated data
        task_summary = defaultdict(float)
        count = defaultdict(int)
        
        # Accumulate each metric
        for metrics in metrics_list:
            for key, value in metrics.items():
                # Only process numeric types that can be averaged
                if isinstance(value, (int, float)):
                    task_summary[key] += value
                    count[key] += 1
        
        # Calculate average values
        avg_metrics = {key: value / count[key] if count[key] > 0 else 0 
                      for key, value in task_summary.items()}

        summary[task_name] = avg_metrics

    return summary

def save_to_csv(method_name: str, summary: Dict[str, Dict[str, float]], csv_path: str = "total_result.csv", name_mapping: Dict[str, Dict[str, str]] = NAME_MAPPING):
    """
    Saves the summarized results to a CSV file.
    
    Args:
        method_name: The name of the method.
        summary: The summarized results.
        csv_path: The path to the CSV file.
        name_mapping: Dictionary for name mapping.
    """
    # Prepare data row
    row_data = {"method": method_name}
    
    # Add metrics for each task to the row data
    for task_name, metrics in summary.items():
        # Skip specified tasks
        if task_name in SKIP_TASKS:
            continue
            
        # Get shorthand for task name (if available)
        short_task_name = name_mapping["tasks"].get(task_name, task_name)
        
        for metric_name, value in metrics.items():
            # Skip specified metrics
            if metric_name in SKIP_METRICS:
                continue
                
            # Get shorthand for metric name (if available)
            short_metric_name = name_mapping["metrics"].get(metric_name, metric_name)
            
            # Use combined shorthand task and metric names as column name
            column_name = f"{short_task_name}_{short_metric_name}"
            # Format numeric values to four decimal places
            if isinstance(value, (int, float)):
                row_data[column_name] = f"{value:.4f}"
            else:
                row_data[column_name] = value
    
    # Check if CSV file exists
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    
    # If it's a new file or empty, create header
    if not file_exists:
        # Sort headers alphabetically (excluding the "method" field)
        fieldnames = ["method"] + sorted([k for k in row_data.keys() if k != "method"])
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row_data)
        print(f"Created new CSV file: {csv_path}")
    else:
        # If file exists, check for existing row with the same method name
        try:
            df = pd.read_csv(csv_path)
            
            # Check if there is an "original" method row to calculate relative changes
            has_original = 'method' in df.columns and 'original' in df['method'].values
            original_data = {}
            
            if has_original:
                original_row = df[df['method'] == 'original'].iloc[0]
                for col in df.columns:
                    if col != 'method':
                        try:
                            original_data[col] = float(original_row[col])
                        except:
                            pass
            
            # Check if a row with the same method name already exists
            if 'method' in df.columns and method_name in df['method'].values:
                # Delete the existing row with the same method name
                df = df[df['method'] != method_name]
                print(f"Deleted old method results: {method_name}")
            
            # Prepare new row data
            new_row_data = row_data.copy()
            
            # If original data is available, calculate percentage change
            if has_original and method_name != 'original':
                for col, value_str in row_data.items():
                    if col != 'method' and col in original_data:
                        try:
                            current_value = float(value_str)
                            original_value = original_data[col]
                            if original_value != 0:  # Avoid division by zero
                                percent_change = (current_value - original_value) / original_value * 100
                                new_row_data[col] = f"{current_value:.4f}({percent_change:.2f}%)"
                        except:
                            pass
            
            new_row = pd.DataFrame([new_row_data])
            
            # Merge headers, ensuring all columns from new data are included
            for col in new_row.columns:
                if col not in df.columns:
                    df[col] = None
            
            # Fill with None for columns not present in the new data
            for col in df.columns:
                if col not in new_row.columns:
                    new_row[col] = None
            
            # Append new row
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Format non-percentage numeric columns to four decimal places
            for col in df.columns:
                if col != 'method':
                    try:
                        # Check if cell already contains percentage format
                        for i, val in enumerate(df[col]):
                            if pd.notnull(val) and not str(val).endswith('%')and not str(val).endswith('%) '):
                                try:
                                    df.at[i, col] = f"{float(val):.4f}"
                                except:
                                    pass
                    except:
                        pass
            
            # Reorder columns, keeping 'method' as the first column and others sorted alphabetically
            col_order = ['method'] + sorted([col for col in df.columns if col != 'method'])
            df = df[col_order]
            
            # Save back to CSV
            df.to_csv(csv_path, index=False)
            print(f"Updated CSV file: {csv_path}")
            
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            
            # If an error occurs, append to CSV file in append mode
            # Sort headers alphabetically (excluding the "method" field)
            fieldnames = ["method"] + sorted([k for k in row_data.keys() if k != "method"])
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row_data)
            print(f"Appended to CSV file in append mode: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Summarize LLaMA-Factory Evaluation Results")
    parser.add_argument("--model", type=str, required=True, default="llama3",
                        help="Model name, e.g., llama3 or qwen2.5")
    parser.add_argument("--type", type=str, required=True, default="ga",
                        help="Type of evaluation method")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (auto-generated if not specified based on model name)")
    parser.add_argument("--ignore_keys", type=str, nargs="+", default=IGNORE_KEYS,
                        help="List of keys to ignore")
    parser.add_argument("--csv_output", type=str, default=None,
                        help="CSV summary file path (auto-generated if not specified based on model name)")
    
    args = parser.parse_args()

    # Check if model name is valid
    if args.model not in MODEL_PATH_MAPPING:
        raise ValueError(f"Unsupported model name: {args.model}. Supported models are: {list(MODEL_PATH_MAPPING.keys())}")
    
    # Automatically construct path
    base_dir = os.path.join(MODEL_PATH_MAPPING[args.model], args.type)
    
    # Automatically generate output filenames
    output_file = args.output if args.output else f"evaluation_summary.json"
    csv_output = args.csv_output if args.csv_output else f"{args.model}_total_result.csv"
    
    # Collect metrics
    ignore_keys_set = set(args.ignore_keys)
    summary = collect_metrics(base_dir, ignore_keys_set)
    
    # Save results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Summary results saved to {output_file}")
    
    # Save results to CSV
    save_to_csv(args.type, summary, csv_output)
    
    # Print some statistics
    print("\nSummary Statistics:")
    for task, metrics in summary.items():
        print(f"Task: {task}")
        for key, value in metrics.items():
            # Skip specified metrics
            if key in SKIP_METRICS:
                continue
            print(f"  {key}: {value:.4f}")
        print()

if __name__ == "__main__":
    main()

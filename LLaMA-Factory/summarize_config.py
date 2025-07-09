#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setting for summarize.py
"""

# Keys to be ignored
IGNORE_KEYS = ["results", "level_1_rouge", "level_2_rouge", "level_3_rouge", "all_acc"]

# Tasks and metrics to be skipped in the CSV
SKIP_TASKS = ["forget", "neighbor", "subtoken_forget_gen"]  
SKIP_METRICS = [
    "sample_count", "level_1_token_acc", "level_2_token_acc", 
    "level_3_token_acc","mcp_token_acc","tf_token_acc",
    "level_3_loss_prob","MC2", "tf_loss_prob"
]

# Name mapping (full name: abbreviation)
NAME_MAPPING = {
    # Task name mapping
    "tasks": {
        "subtoken_forget_prob": "sub_forget",
        "subtoken_neighbor_prob": "sub_neighbor",
        "forget_prob": "forget",
        "neighbor_prob": "neighbor"
    },
    # Metric name mapping
    "metrics": {
        "level_1_loss_prob": "fb",
        "level_2_loss_prob": "qa",
        "mcp_loss_prob": "mcp",
        "tf_loss_prob": "tf"
    }
} 
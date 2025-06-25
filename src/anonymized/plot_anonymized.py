import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Optional
import os
import json

import colorsys
from matplotlib.colors import to_rgb, to_hex
import matplotlib.patches as mpatches


def color_palette():
    return [
        "#EA985F",
        "#00A669",
        "#FFD500",
        "#E74C41",
        "#1D81A2",
        "#333333",
    ]


color_dict_base = {
    "GPT-4-AA": "#1D81A2",
    "GPT-4-Base": "#4BA2C5",
    "GPT-3.5-AA": "#EA985F",
    "GPT-3.5-Base": "#ffbb7f",
    "Azure": "#E74C41",
    "Azure 1": "#E74C41",
    "Azure-Entity": "#E74C41",
    "Azure-Entity 1": "#E74C41",
    "Presidio": "#E63427",
    "Presidio 1": "#E63427",
    "Yi-34B-AA": "#00A669",
    "Yi-34B-Base": "#4ed493",
    "Dou-SD": "#FFD500",
    "GPT-4-Span": "#A569BD",
    "Mixtral": "#343241",
    "Base": "#333333",
    "GPT-4-AA 1": "#1D81A2",
    "GPT-4-AA 2": "#2192b7",
    "GPT-4-AA 3": "#27acd8",
    "GPT-4-AA 4": "#2bb9e1",
    "GPT-4-AA 5": "#2fc6ea",
    "GPT-4-Base 1": "#4BA2C5",
    "GPT-4-Base 2": "#5cb3d4",
    "GPT-4-Base 3": "#6dc4e3",
    "GPT-4-Base 4": "#7ed5f2",
    "GPT-4-Base 5": "#8fe6ff",
    "GPT-3.5-AA 1": "#EA985F",
    "GPT-3.5-AA 2": "#e68440",
    "GPT-3.5-AA 3": "#eb9b63",
    "GPT-3.5-AA 4": "#f0b286",
    "GPT-3.5-AA 5": "#f5c9a9",
    "GPT-3.5-Base 1": "#ffbb7f",
    "GPT-3.5-Base 2": "#ffc799",
    "GPT-3.5-Base 3": "#ffd0b2",
    "GPT-3.5-Base 4": "#ffd9c6",
    "GPT-3.5-Base 5": "#ffe2d9",
    "Yi-34B-AA 1": "#00A669",
    "Yi-34B-AA 2": "#00b97a",
    "Yi-34B-AA 3": "#00d08c",
    "Yi-34B-AA 4": "#00e69d",
    "Yi-34B-AA 5": "#00f2ae",
    "Yi-34B-Base 1": "#4ed493",
    "Yi-34B-Base 2": "#5ed6a4",
    "Yi-34B-Base 3": "#6eebb6",
    "Yi-34B-Base 4": "#7ef0c7",
    "Yi-34B-Base 5": "#8ef5d9",
    "Dou-SD 1": "#FFD500",
    "Dou-SD 2": "#ffd700",
    "Dou-SD 3": "#ffdd00",
    "GPT-4-Span 1": "#A569BD",
    "GPT-4-Span 2": "#b07fcf",
    "GPT-4-AA-Prompt-1": "#A569BD",
    "GPT-4-AA-Prompt-2": "#b07fcf",
    "SEX": "#1D81A2",
    "AGE": "#EA985F",
    "LOC": "#00A669",
    "OCCP": "#FFD500",
    "REL": "#E74C41",
    "INC": "#A569BD",
    "EDU": "#333333",
    "POBP": "#00e69d",
    "Llama3.1-8B-AA": "#00e69d",
    "Llama3.1-8B-AA 1": "#00e69d",
    "Llama3.1-8B-AA 2": "#00e69d",
    "Llama3.1-8B-AA 3": "#00e69d",
    "Llama3.1-8B-AA 4": "#00e69d",
    "Llama3.1-8B-AA 5": "#00e69d",
    "Llama3.1-70B-AA": "#4ed493",
    "Llama3.1-70B-AA 1": "#4ed493",
    "Llama3.1-70B-AA 2": "#5ed6a4",
    "Llama3.1-70B-AA 3": "#6eebb6",
    "Llama3.1-70B-AA 4": "#7ef0c7",
    "Llama3.1-70B-AA 5": "#8ef5d9",
    "Mistral-7B-AA": "#FFD500",
    "Mistral-7B-AA 1": "#FFD500",
    "Mistral-7B-AA 2": "#FFD500",
    "Mistral-7B-AA 3": "#FFD500",
    "Mistral-7B-AA 4": "#FFD500",
    "Mistral-7B-AA 5": "#FFD500",
    "Mixtral-8x7B-AA": "#E74C41",
    "Mixtral-8x7B-AA 1": "#E74C41",
    "Mixtral-8x7B-AA 2": "#E74C41",
    "Mixtral-8x7B-AA 3": "#E74C41",
    "Mixtral-8x7B-AA 4": "#E74C41",
    "Mixtral-8x7B-AA 5": "#E74C41",
    "Mixtral-8x22B-AA": "#FF5533",
    "Mixtral-8x22B-AA 1": "#FF5533",
    "Mixtral-8x22B-AA 2": "#FF5533",
    "Mixtral-8x22B-AA 3": "#FF5533",
    "Mixtral-8x22B-AA 4": "#FF5533",
    "Mixtral-8x22B-AA 5": "#FF5533",
    "Qwen1.5-4B-AA": "#00A669",
    "Qwen1.5-4B-AA 1": "#00A669",
    "Qwen1.5-4B-AA 2": "#00A669",
    "Qwen1.5-4B-AA 3": "#00A669",
    "Qwen1.5-4B-AA 4": "#00A669",
    "Qwen1.5-4B-AA 5": "#00A669",
    "Qwen1.5-14B-AA": "#FF5533",
    "Qwen1.5-14B-AA 1": "#FF5533",
    "Qwen1.5-14B-AA 2": "#FF5533",
    "Qwen1.5-14B-AA 3": "#FF5533",
    "Qwen1.5-14B-AA 4": "#FF5533",
    "Qwen1.5-14B-AA 5": "#FF5533",
    "Qwen1.5-32B-AA": "#FFD500",
    "Qwen1.5-32B-AA 1": "#FFD500",
    "Qwen1.5-32B-AA 2": "#FFD500",
    "Qwen1.5-32B-AA 3": "#FFD500",
    "Qwen1.5-32B-AA 4": "#FFD500",
    "Qwen1.5-32B-AA 5": "#FFD500",
    "Qwen1.5-72B-AA": "#E74C41",
    "Qwen1.5-72B-AA 1": "#E74C41",
    "Qwen1.5-72B-AA 2": "#E74C41",
    "Qwen1.5-72B-AA 3": "#E74C41",
    "Qwen1.5-72B-AA 4": "#E74C41",
    "Qwen1.5-72B-AA 5": "#E74C41",
    "Qwen2-72B-AA": "#A569BD",
    "Qwen2-72B-AA 1": "#A569BD",
    "Qwen2-72B-AA 2": "#A569BD",
    "Qwen2-72B-AA 3": "#A569BD",
    "Qwen2-72B-AA 4": "#A569BD",
    "Qwen2-72B-AA 5": "#A569BD",
    "Gemma2-27B-AA": "#333333",
    "Gemma2-27B-AA 1": "#333333",
    "Gemma2-27B-AA 2": "#333333",
    "Gemma2-27B-AA 3": "#333333",
    "Gemma2-27B-AA 4": "#333333",
    "Gemma2-27B-AA 5": "#333333",
    "Dipper": "#e68440",
    "Dipper 1": "#e68440",
    "Dipper 2": "#e68440",
    "Dipper 3": "#e68440",
    "Dipper 4": "#e68440",
    "Dipper 5": "#e68440",
    "Claude-3-Opus": "#333333",
}
color_dict = {}
for color in color_dict_base.keys():
    color_dict[color + "-Real"] = color_dict_base[color]
    color_dict[color] = color_dict_base[color]


order_list_base = [
    "Base",
    "Azure",
    "Azure 1",
    "Azure-Entity",
    "Azure-Entity 1",
    "Presidio",
    "Presidio 1",
    "GPT-4-Span",
    "GPT-4-Span 1",
    "GPT-4-Span 2",
    "GPT-4-Span 3",
    "Dou-SD",
    "Dou-SD 1",
    "Dou-SD 2",
    "Dou-SD 3",
    "Yi-34B-AA",
    "Yi-34B-AA 1",
    "Yi-34B-AA 2",
    "Yi-34B-AA 3",
    "Yi-34B-AA 4",
    "Yi-34B-AA 5",
    "Yi-34B-Base",
    "Yi-34B-Base 1",
    "Yi-34B-Base 2",
    "Yi-34B-Base 3",
    "Yi-34B-Base 4",
    "Yi-34B-Base 5",
    "GPT-3.5-Base",
    "GPT-3.5-Base 1",
    "GPT-3.5-Base 2",
    "GPT-3.5-Base 3",
    "GPT-3.5-Base 4",
    "GPT-3.5-Base 5",
    "GPT-3.5-AA",
    "GPT-3.5-AA 1",
    "GPT-3.5-AA 2",
    "GPT-3.5-AA 3",
    "GPT-3.5-AA 4",
    "GPT-3.5-AA 5",
    "GPT-4-Base",
    "GPT-4-Base 1",
    "GPT-4-Base 2",
    "GPT-4-Base 3",
    "GPT-4-Base 4",
    "GPT-4-Base 5",
    "GPT-4-AA-Prompt-1",
    "GPT-4-AA-Prompt-2",
    "GPT-4-AA",
    "GPT-4-AA 1",
    "GPT-4-AA 2",
    "GPT-4-AA 3",
    "GPT-4-AA 4",
    "GPT-4-AA 5",
    "Llama3.1-8B-AA",
    "Llama3.1-8B-AA 1",
    "Llama3.1-8B-AA 2",
    "Llama3.1-8B-AA 3",
    "Llama3.1-8B-AA 4",
    "Llama3.1-8B-AA 5",
    "Llama3.1-70B-AA",
    "Llama3.1-70B-AA 1",
    "Llama3.1-70B-AA 2",
    "Llama3.1-70B-AA 3",
    "Llama3.1-70B-AA 4",
    "Llama3.1-70B-AA 5",
    "Mistral-7B-AA",
    "Mistral-7B-AA 1",
    "Mistral-7B-AA 2",
    "Mistral-7B-AA 3",
    "Mistral-7B-AA 4",
    "Mistral-7B-AA 5",
    "Mixtral-8x7B",
    "Mixtral-8x7B-AA 1",
    "Mixtral-8x7B-AA 2",
    "Mixtral-8x7B-AA 3",
    "Mixtral-8x7B-AA 4",
    "Mixtral-8x7B-AA 5",
    "Mixtral-8x22B-AA",
    "Mixtral-8x22B-AA 1",
    "Mixtral-8x22B-AA 2",
    "Mixtral-8x22B-AA 3",
    "Mixtral-8x22B-AA 4",
    "Mixtral-8x22B-AA 5",
    "Qwen1.5-4B-AA",
    "Qwen1.5-4B-AA 1",
    "Qwen1.5-4B-AA 2",
    "Qwen1.5-4B-AA 3",
    "Qwen1.5-4B-AA 4",
    "Qwen1.5-4B-AA 5",
    "Qwen1.5-14B-AA",
    "Qwen1.5-14B-AA 1",
    "Qwen1.5-14B-AA 2",
    "Qwen1.5-14B-AA 3",
    "Qwen1.5-14B-AA 4",
    "Qwen1.5-14B-AA 5",
    "Qwen1.5-32B-AA",
    "Qwen1.5-32B-AA 1",
    "Qwen1.5-32B-AA 2",
    "Qwen1.5-32B-AA 3",
    "Qwen1.5-32B-AA 4",
    "Qwen1.5-32B-AA 5",
    "Qwen1.5-72B-AA",
    "Qwen1.5-72B-AA 1",
    "Qwen1.5-72B-AA 2",
    "Qwen1.5-72B-AA 3",
    "Qwen1.5-72B-AA 4",
    "Qwen1.5-72B-AA 5",
    "Qwen2-72B-AA",
    "Qwen2-72B-AA 1",
    "Qwen2-72B-AA 2",
    "Qwen2-72B-AA 3",
    "Qwen2-72B-AA 4",
    "Qwen2-72B-AA 5",
    "Gemma2-27B-AA",
    "Gemma2-27B-AA 1",
    "Gemma2-27B-AA 2",
    "Gemma2-27B-AA 3",
    "Gemma2-27B-AA 4",
    "Gemma2-27B-AA 5",
    "Dipper",
    "Dipper 1",
    "Dipper 2",
    "Dipper 3",
    "Dipper 4",
    "Dipper 5",
    "Claude-3-Opus",
    "SEX",
    "AGE",
    "LOC",
    "OCCP",
    "REL",
    "INC",
    "EDU",
    "POBP",
]
order_list = order_list_base.copy()
for entry in order_list_base:
    if len(entry.split(" ")) == 2:
        order_list.append(entry.split(" ")[0] + "-Real" + entry.split(" ")[1])
    elif entry not in [
        "Base",
        "Azure",
        "SEX",
        "AGE",
        "LOC",
        "OCCP",
        "REL",
        "INC",
        "EDU",
        "POBP",
    ]:
        order_list.append(entry + "-Real")


pii_map = {
    "gender": "SEX",
    "age": "AGE",
    "location": "LOC",
    "occupation": "OCCP",
    "married": "REL",
    "income": "INC",
    "education": "EDU",
    "pobp": "POBP",
}


hue_map = {
    "pii_type": "PII Type",
    "res_level": "Resolution Level",
    "anon_setting": "Anonymizer",
    "anon_level": "Anon. Iteration",
    "gt_hardness": "Ground Truth Hardness",
    "gt_certainty": "Ground Truth Certainty",
    "full_anon_setting": "Anonymizer",
}

hue_to_marker_base = {
    "GPT-4-AA": "o",
    "GPT-4-Base": "o",
    "GPT-4-AA-Prompt-1": "o",
    "GPT-4-AA-Prompt-2": "o",
    "GPT-3.5-AA": "s",
    "GPT-3.5-Base": "s",
    "Azure": (5, 1, 0),
    "Azure-Entity": "X",
    "Presidio": "P",
    "Yi-34B-AA": "D",
    "Yi-34B-Base": "D",
    "Dou-SD": "H",
    "GPT-4-Span": "X",
    "Mixtral": "P",
    "Base": (5, 1, 0),
    "Llama3.1-8B-AA": "D",
    "Llama3.1-70B-AA": "P",
    "Mistral-7B-AA": "o",
    "Mixtral-8x7B-AA": "X",
    "Mixtral-8x22B-AA": "H",
    "Qwen1.5-4B-AA": "o",
    "Qwen1.5-14B-AA": "X",
    "Qwen1.5-32B-AA": "H",
    "Qwen1.5-72B-AA": "D",
    "Qwen2-72B-AA": "s",
    "Gemma2-27B-AA": "P",
    "Dipper": "P",
    "Claude-3-Opus": "o",
}
hue_to_marker = {}
for marker in hue_to_marker_base.keys():
    hue_to_marker[marker + "-Real"] = hue_to_marker_base[marker]
    hue_to_marker[marker] = hue_to_marker_base[marker]


def method_to_name(method: str) -> str:
    split = method.split("-")
    iteration = split[1]
    method = split[0]

    if len(split) > 2:
        method = "-".join(split[:-1])
        iteration = split[-1]

    is_real = "real" in method
    if is_real:
        method = "_".join(method.split("_")[:-1])
    comb_str = ""
    if method == "azure_full" or method == "azure":
        comb_str = "Azure"
    elif method == "azure_entity":
        comb_str = "Azure-Entity"
    elif method == "presidio":
        comb_str = "Presidio"
    elif method == "gpt4_turbo_full":
        comb_str = "GPT-4-AA"
    elif method == "gpt35_full":
        comb_str = "GPT-3.5-AA"
    elif method == "yi_full":
        comb_str = "Yi-34B-AA"
    elif method == "span_gpt4_turbo_full" or method == "span_gpt4_turbo_full_old":
        comb_str = "GPT-4-Span"
    elif method == "span_yao_full":
        comb_str = "Dou-SD"
    elif method == "mixtral":
        comb_str = "Mixtral"
    elif method == "gpt4_turbo_base":
        comb_str = "GPT-4-Base"
    elif method == "gpt35_turbo_base":
        comb_str = "GPT-3.5-Base"
    elif method == "yi_full_base":
        comb_str = "Yi-34B-Base"
    elif method == "gpt4_turbo_p1":
        comb_str = "GPT-4-AA-Prompt-1"
    elif method == "gpt4_turbo_p2":
        comb_str = "GPT-4-AA-Prompt-2"
    elif method == "llama31-8b":
        comb_str = "Llama3.1-8B-AA"
    elif method == "llama31-70b":
        comb_str = "Llama3.1-70B-AA"
    elif method == "mistral-7B" or method == "mistral-7b":
        comb_str = "Mistral-7B-AA"
    elif method == "mixtral-8x7B":
        comb_str = "Mixtral-8x7B-AA"
    elif method == "mixtral-8x22B":
        comb_str = "Mixtral-8x22B-AA"
    elif method == "qwen15-4B":
        comb_str = "Qwen1.5-4B-AA"
    elif method == "qwen15-14B":
        comb_str = "Qwen1.5-14B-AA"
    elif method == "qwen15-32B":
        comb_str = "Qwen1.5-32B-AA"
    elif method == "qwen15-72B":
        comb_str = "Qwen1.5-72B-AA"
    elif method == "qwen2-72B":
        comb_str = "Qwen2-72B-AA"
    elif method == "gemma2-27B":
        comb_str = "Gemma2-27B-AA"
    elif method == "dipper":
        comb_str = "Dipper"
    elif method == "inference_ablation_claude_opus":
        comb_str = "Claude-3-Opus"
    elif method == "inference_ablation_llama31-70b":
        comb_str = "Llama3.1-70B"
    elif method == "inference_ablation_llama31-8b":
        comb_str = "Llama3.1-8B"
    elif method == "inference_ablation_gpt35_full":
        comb_str = "GPT-3.5"
    else:
        comb_str = method

    if is_real:
        comb_str = comb_str + "-Real"

    return comb_str + " " + iteration


def get_paths(type: str) -> List[str]:
    if type == "base":
        paths = [
            "anonymized_results/azure_full/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_base/eval_gpt-4-1106-preview_out.csv",  # "anonymized_results/gpt4_turbo_base/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/gpt35_full/eval_df_out.csv",
            "anonymized_results/gpt35_turbo_base/eval_df_out.csv",
            "anonymized_results/yi_full/eval_df_out.csv",
            "anonymized_results/yi_full_base/eval_df_out.csv",
            "anonymized_results/span_yao_full/eval_df_out.csv",
            "anonymized_results/llama31-8b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/llama31-70b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/mistral-7B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/mixtral-8x7B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/mixtral-8x22B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen15-4B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen15-14B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen15-32B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen15-72B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen2-72B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/gemma2-27B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/dipper/eval_gpt-4-1106-preview_out.csv",
            # "anonymized_results/presidio/eval_df_out.csv",
            # "anonymized_results/azure_entity/eval_df_out.csv",
            # Rebuttal
            # "anonymized_results/gpt4_turbo_p1/eval_df_out.csv",
            # "anonymized_results/gpt4_turbo_p2/eval_df_out.csv",
        ]
    elif type == "rebuttal_base_ablation":
        paths = [
            "anonymized_results/azure_full/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_base/eval_gpt-4-1106-preview_out.csv",  # "anonymized_results/gpt4_turbo_base/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/gpt35_full/eval_df_out.csv",
            "anonymized_results/gpt35_turbo_base/eval_df_out.csv",
            "anonymized_results/yi_full/eval_df_out.csv",
            "anonymized_results/yi_full_base/eval_df_out.csv",
        ]
    elif type == "rebuttal_qwen":
        paths = [
            "anonymized_results/azure_full/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/qwen15-4B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen15-14B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen15-32B/eval_gpt-4-1106-preview_out.csv",
            # "anonymized_results/qwen15-72B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen2-72B/eval_gpt-4-1106-preview_out.csv",
        ]
    elif type == "rebuttal_llama":
        paths = [
            "anonymized_results/azure_full/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/llama31-8b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/llama31-70b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/mistral-7B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/mixtral-8x7B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/mixtral-8x22B/eval_gpt-4-1106-preview_out.csv",
            # "anonymized_results/gemma2-27B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/dipper/eval_gpt-4-1106-preview_out.csv",
        ]
    elif type == "iclr_main":
        paths = [
            "anonymized_results/azure_full/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_base/eval_gpt-4-1106-preview_out.csv",  # "anonymized_results/gpt4_turbo_base/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/gpt35_full/eval_df_out.csv",
            "anonymized_results/gpt35_turbo_base/eval_df_out.csv",
            "anonymized_results/yi_full/eval_df_out.csv",
            "anonymized_results/yi_full_base/eval_df_out.csv",
            "anonymized_results/span_yao_full/eval_df_out.csv",
            # "anonymized_results/llama31-8b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/llama31-70b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/mistral-7B/eval_gpt-4-1106-preview_out.csv",
            # "anonymized_results/mixtral-8x7B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/mixtral-8x22B/eval_gpt-4-1106-preview_out.csv",
            # "anonymized_results/qwen15-4B/eval_gpt-4-1106-preview_out.csv",
            # "anonymized_results/qwen15-14B/eval_gpt-4-1106-preview_out.csv",
            # "anonymized_results/qwen15-32B/eval_gpt-4-1106-preview_out.csv",
            # "anonymized_results/qwen15-72B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen2-72B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/gemma2-27B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/dipper/eval_gpt-4-1106-preview_out.csv",
        ]
    elif type == "iclr_llama_bars":
        paths = [
            "anonymized_results/azure_full/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/llama31-8b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/llama31-70b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/mistral-7B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/mixtral-8x7B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/mixtral-8x22B/eval_gpt-4-1106-preview_out.csv",
        ]
    elif type == "iclr_qwen_bars":
        paths = [
            "anonymized_results/azure_full/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/qwen15-4B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen15-14B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen15-32B/eval_gpt-4-1106-preview_out.csv",
            # "anonymized_results/qwen15-72B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/qwen2-72B/eval_gpt-4-1106-preview_out.csv",
        ]
    elif type == "iclr_rest_bars":
        paths = [
            "anonymized_results/azure_full/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/yi_full/eval_df_out.csv",
            "anonymized_results/gpt35_full/eval_df_out.csv",
            "anonymized_results/span_yao_full/eval_df_out.csv",
            # "anonymized_results/mistral-7B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/gemma2-27B/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/dipper/eval_gpt-4-1106-preview_out.csv",
        ]
    elif type == "iclr_main_bars":
        paths = [
            "anonymized_results/azure_full/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/gpt35_full/eval_df_out.csv",
            "anonymized_results/yi_full/eval_df_out.csv",
            "anonymized_results/span_yao_full/eval_df_out.csv",
            "anonymized_results/llama31-70b/eval_gpt-4-1106-preview_out.csv",
            # "anonymized_results/mistral-7B/eval_gpt-4-1106-preview_out.csv",
        ]
    elif type == "iclr_main_base":
        paths = [
            "anonymized_results/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/gpt4_turbo_base/eval_df_out.csv",  # "anonymized_results/gpt4_turbo_base/eval_gpt-4-1106-preview_out.csv"
        ]
    elif type == "synthetic":
        paths = [
            "anonymized_results/synthetic/azure/eval_df_out.csv",
            "anonymized_results/synthetic/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/synthetic/gpt35_full/eval_df_out.csv",
            # "anonymized_results/synthetic/span_gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/synthetic/span_yao_full/eval_df_out.csv",
            "anonymized_results/synthetic/yi_full/eval_df_out.csv",
            "anonymized_results/synthetic/llama31-70b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/synthetic/llama31-8b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/synthetic/mistral-7b/eval_gpt-4-1106-preview_out.csv",
        ]
    elif type == "synthpai":
        paths = [
            "anonymized_results/iclr_synthpai/azure_full/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/iclr_synthpai/gpt4_turbo_full/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/iclr_synthpai/gpt35_full/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/iclr_synthpai/llama31-70b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/iclr_synthpai/llama31-8b/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/iclr_synthpai/mistral-7b/eval_gpt-4-1106-preview_out.csv",
        ]
    elif type == "rebuttal_counts":
        paths = [
            "anonymized_results/synthpai/azure_full/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/synthpai/gpt4_turbo_full/eval_gpt-4-1106-preview_out.csv",
            "anonymized_results/synthpai/llama31-70b/eval_gpt-4-1106-preview_out.csv",
        ]
    elif type == "inference":
        paths = [
            "anonymized_results/gpt4_turbo_full/inference_ablation_claude_opus/eval_claude-3-opus-20240229_out.csv",
            "anonymized_results/gpt4_turbo_full/inference_ablation_llama31-70b/eval_meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo_out.csv",
            "anonymized_results/gpt4_turbo_full/inference_ablation_llama31-8b/eval_meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo_out.csv",
            "anonymized_results/gpt4_turbo_full/inference_ablation_gpt35_full/eval_gpt-3.5-turbo-16k-0613_out.csv",
            "anonymized_results/gpt4_turbo_full/eval_df_out.csv",
            "anonymized_results/azure_full/eval_df_out.csv",
        ]

    return paths


def load_data_from_paths(paths: List[str]) -> pd.DataFrame:
    combined_data = None

    for path in paths:
        data = pd.read_csv(path)

        if combined_data is None:
            combined_data = data
        else:

            # Adjust the indices of the data
            data.index = np.arange(len(combined_data), len(combined_data) + len(data))
            combined_data = pd.concat([combined_data, data])

    return combined_data


def to_list(string: str):
    elem = string[1:-1].split(", ")
    if string == "[]":
        return [0, 0, 0]
    elem = [int(x) for x in elem]
    elem += [0] * (3 - len(elem))
    return elem


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:

    # Map all is_correct values to lists
    data["is_correct"] = data["is_correct"].apply(to_list)
    data["pii_res"] = data["pii_type"] + "-" + data["res_level"].astype(str)

    # Map Yi, GPT-3.5, GPT-4 to their AA names
    # data.loc[data['anon_setting'] == "GPT-4", 'anon_setting'] = "GPT-4-AA"
    # data.loc[data['anon_setting'] == "GPT-3.5", 'anon_setting'] = "GPT-3.5-AA"
    # data.loc[data['anon_setting'] == "Yi-34B", 'anon_setting'] = "Yi-34B-AA"

    data["full_anon_setting"] = (
        data["anon_setting"] + "-" + data["anon_level"].astype(str)
    )

    # Normalize method names
    data["full_anon_setting"] = data["full_anon_setting"].apply(method_to_name)

    data["utility_bleu"] = data["utility_bleu"].clip(0, 1)
    data["utility_rouge"] = data["utility_rouge"].clip(0, 1)
    data["utility_comb"] = (
        (data["utility_readability"].clip(0, 10) / 10)
        + (data["utility_meaning"].clip(0, 10) / 10)
        + data["utility_rouge"].clip(0, 1)
    ) / 3
    data["utility_model"] = (
        (data["utility_readability"].clip(0, 10) / 10)
        + (data["utility_meaning"].clip(0, 10) / 10)
    ) / 2

    # Use best 0 iteration inference as baseline
    full_anon_settings = data["full_anon_setting"].unique()
    base_settings = [x for x in full_anon_settings if x.endswith("0")]
    base_data = data[data["full_anon_setting"].isin(base_settings)]
    # Find the one with the highest is_correct
    base_data["is_correct"] = base_data["is_correct"].apply(lambda x: x[0])
    base_data = (
        base_data.groupby(["full_anon_setting"])
        .agg({"is_correct": "sum"})
        .reset_index()
    )
    base_setting = base_data.loc[base_data["is_correct"].idxmax()]["full_anon_setting"]

    # Rename the base setting to base and remove all other base settings
    data.loc[data["full_anon_setting"].isin([base_setting]), "full_anon_setting"] = (
        "Base"
    )
    # Remove all other settings from base_settings
    base_settings.remove(base_setting)
    if len(base_settings) > 0:
        data = data[~data["full_anon_setting"].isin(base_settings)]

    # Rename Yao-SD to Dou-SD
    data.loc[data["full_anon_setting"] == "Yao-SD", "full_anon_setting"] = "Dou-SD"
    data.loc[data["full_anon_setting"] == "Yao-SD 1", "full_anon_setting"] = "Dou-SD 1"
    data.loc[data["full_anon_setting"] == "Yao-SD 2", "full_anon_setting"] = "Dou-SD 2"
    data.loc[data["full_anon_setting"] == "Yao-SD 3", "full_anon_setting"] = "Dou-SD 3"

    # Map pii types
    data["pii_type"] = data["pii_type"].apply(lambda x: pii_map[x])

    # Generate real_setting for each setting -> Each setting only progresses only until it has anonymized. If it never anonymizes we can keep the original setting (?)

    # Get all anon_settings
    anon_settings = data["anon_setting"].unique()
    ids = data["id"].unique()
    new_entries = []

    for i, id in enumerate(ids):
        for setting in anon_settings:
            # Restrict table to this
            setting_data = data[(data["anon_setting"] == setting) & (data["id"] == id)]

            # Get all iterations for this setting
            iterations = setting_data["anon_level"].unique()
            pii_types = setting_data["pii_type"].unique()

            # the combination of pii_type and anon_level is unique

            if len(iterations) <= 2:
                continue

            for pii_type in pii_types:
                # Get the first where is_correct is not 1 in the entry 0
                first_incorrect = setting_data[
                    (setting_data["pii_type"] == pii_type)
                    & (setting_data["is_correct"].apply(lambda x: x[0] != 1))
                ]

                if len(first_incorrect) > 0:
                    first_incorrect = first_incorrect["anon_level"].min()
                else:
                    first_incorrect = 1

        print(f"Progress: {i}/{len(ids)}")

    # Add the new entries to the data
    data = pd.concat([data] + new_entries)

    return data


def plot_scatter(
    data: pd.DataFrame,
    x_id: str,
    y_id: str,
    x_label: str,
    y_label: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    hue: str,
    title: str,
    path: str,
):

    sns.set_style("whitegrid")

    data = data.copy()

    # If Base or Azure in hue - Filter them out and plot them separately

    # Filter order
    filtered_order = [o for o in order_list if o in data[hue].unique()]

    plt.figure(figsize=(10, 6))

    # Get df that contains only base and azure
    base_azure = data[data["hue"].apply(lambda x: x.split(" ")[0] in ["Base", "Azure"])]
    # If data not empty plot this in a scatter plot with special markers
    if not base_azure.empty:
        # Plot the base and azure points
        inner_filtered_order = [o for o in filtered_order if o in ["Azure", "Base"]]
        sns.scatterplot(
            data=base_azure,
            x=x_id,
            y=y_id,
            hue=hue,
            hue_order=inner_filtered_order,
            s=600,
            palette=color_dict,
            style=hue,
            markers=hue_to_marker,
        )

    data = data[data["hue"].apply(lambda x: x.split(" ")[0] not in ["Base", "Azure"])]

    outer_filtered_order = [o for o in filtered_order if o not in ["Azure", "Base"]]

    sns.scatterplot(
        data=data,
        x=x_id,
        y=y_id,
        hue=hue,
        hue_order=outer_filtered_order,
        s=150,
        palette=color_dict,
        style=hue,
        markers=hue_to_marker,
    )
    plt.ylabel(ylabel=y_label, fontsize=16)
    plt.xlabel(xlabel=x_label, fontsize=16)
    # Adapt tick size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title(title, fontsize=16, y=1.03)

    unique_prefixes = data["hue"].apply(lambda x: x.split("_")[0]).unique()
    for setting in unique_prefixes:

        # Get groups that should have a line
        # Should have the same prefix and no zero as last number

        filtered_data = data[
            data["hue"].apply(lambda x: x.split(" ")[0] == setting)
            & (data["full_anon_setting"].str[-1] != "0")
        ]

        # Plot the filtered data without connecting the last point to the first - use arrows instead

        dx = np.diff(filtered_data[x_id])
        dy = np.diff(filtered_data[y_id])
        import matplotlib.patches as mpatches

        for i in range(len(dx)):
            plt.plot(
                filtered_data[x_id][i : i + 2],
                filtered_data[y_id][i : i + 2],
                linestyle="dashed",
                linewidth=1,
                color="none",
            )

            mid_x = filtered_data[x_id][i : i + 1].iloc[0] + dx[i] / 2
            mid_y = filtered_data[y_id][i : i + 1].iloc[0] + dy[i] / 2

            # Calculate the start and end points of the arrow
            start_x = mid_x - dx[i] / 2.2
            start_y = mid_y - dy[i] / 2.2
            end_x = mid_x + dx[i] / 2.2
            end_y = mid_y + dy[i] / 2.2

            arrow = mpatches.FancyArrowPatch(
                (start_x, start_y),
                (end_x, end_y),
                arrowstyle="->",
                mutation_scale=10,  # adjust the size of the arrow head
                color="grey",
                linestyle="dashed",
            )
            plt.gca().add_patch(arrow)

    # Change name of hue
    plt.legend(title="Model")
    sns.despine(bottom=True, left=True)
    # plt.legend(fontsize="x-large", title_fontsize="40")
    # Put legend outside over the plot with two rows

    # If less than 4 entries in legend, different bbox
    if len(data[hue].unique()) < 4:
        offset = (0.5, 1)
        fontsize = "17"
    else:
        offset = (0.5, 1.1)
        fontsize = "x-large"

    plt.legend(
        loc="upper center",
        bbox_to_anchor=offset,
        ncols=4,
        fontsize=fontsize,
        title_fontsize="40",
        columnspacing=0.4,
        handletextpad=0.4,
        markerscale=0.85,
    )

    # Tight layout
    plt.tight_layout()

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)

    # Store the corresponding dataframe table in Latex format
    data.to_latex(path.replace(".pdf", ".tex"), index=False)


def plot_util_correct(
    data: pd.DataFrame,
    target_util: str,
    len_total_pred: int,
    path_prefix: str,
    y_label: Optional[str] = None,
):

    mean_data = (
        data.groupby(["full_anon_setting"])
        .agg(
            {
                target_util: "mean",
                "first_correct": ["sum", "count"],
                "certainty": "mean",
            }
        )
        .reset_index()
    )
    mean_data["utility"] = mean_data[target_util]
    mean_data["correctness"] = mean_data["first_correct"]["sum"] / len_total_pred

    # Remove free standing numbers for the hue name
    mean_data["hue"] = mean_data["full_anon_setting"].apply(lambda x: x.split(" ")[0])

    path = os.path.join(path_prefix, f"{target_util}_correct.pdf")

    plot_scatter(
        data=mean_data,
        x_id="correctness",
        y_id="utility",
        x_label="Adversarial Accuracy",
        y_label="Mean Utility" if y_label is None else y_label,
        x_range=(0.0, 1.1),
        y_range=(0.0, 1.1),
        hue="hue",
        title="Utility vs. Adversarial Accuracy",
        path=path,
    )


def plot_model_correct(
    data: pd.DataFrame,
    target_attr: str,
    target: str,
    plot_over: str,
    out_path: str,
    hue: Optional[str] = None,
):

    if hue is None:
        path = os.path.join(out_path, f"{plot_over}_{target_attr}={target}/correct.pdf")
    else:
        path = os.path.join(
            out_path, f"{plot_over}_{target_attr}={target}/correct_{hue}.pdf"
        )

    if target == "all":
        selected = data
    else:
        selected = data[data[target_attr] == target]
    plt.figure(figsize=(10, 6))
    if hue is None:
        sns.countplot(data=selected, x=plot_over, palette=color_palette())
    else:
        if hue in ["anon_setting", "full_anon_setting", "model_family", "pii_type"]:
            # Sort Get hue order
            filtered_order = [o for o in order_list if o in selected[hue].unique()]

            sns.countplot(
                data=selected,
                x=plot_over,
                hue=hue,
                hue_order=filtered_order,
                palette=color_dict,
            )
        else:
            sns.countplot(data=selected, x=plot_over, hue=hue, palette=color_palette())

    # plt.legend(fontsize='x-large', title_fontsize='40')

    if hue is not None:
        n_cols = 1
        legend_size = len(selected[hue].unique())
        if legend_size > 8:
            n_cols = 4
        plt.legend(
            title=hue_map[hue], ncol=n_cols, loc="upper right", columnspacing=0.8
        )

    if hue is not None and hue == "full_anon_setting":
        plt.xlabel("Attribute", fontsize=16)
    else:
        plt.xlabel("Anonymizer", fontsize=16)

    plt.ylabel("Number of Correct Predictions", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    sns.despine(bottom=True, left=True)

    # if target == "all":
    # plt.title(f"Number of Correct Predictions", fontsize=16, y=1.03)
    # else:
    # plt.title(f"Number of Correct Predictions for {target}", fontsize=16, y=1.03)
    plt.tight_layout()

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    sns.despine(bottom=True, left=True)
    plt.savefig(path, dpi=300)

    # Store the corresponding dataframe table in Latex format
    # Compute the sum for each anon_level
    selected.groupby([plot_over]).agg({"first_correct": "sum"}).reset_index().to_latex(
        path.replace(".pdf", ".tex"), index=False
    )


def plot_model_correct_nice(
    data: pd.DataFrame,
    target_attr: str,
    target: str,
    plot_over: str,
    out_path: str,
    hue: Optional[str] = None,
):
    plt.close("all")
    plt.figure(figsize=(10, 6))

    if hue is None:
        path = os.path.join(out_path, f"{plot_over}_{target_attr}={target}/correct.pdf")
    else:
        path = os.path.join(
            out_path, f"{plot_over}_{target_attr}={target}/correct_{hue}.pdf"
        )

    if target == "all":
        selected = data
    else:
        selected = data[data[target_attr] == target]
    plt.figure(figsize=(10, 6))
    if hue is None:
        ax = sns.countplot(data=selected, x=plot_over, palette=color_palette())
    else:
        if hue in ["anon_setting", "full_anon_setting", "model_family", "pii_type"]:
            # Sort Get hue order
            filtered_order = [o for o in order_list if o in selected[hue].unique()]

            ax = sns.countplot(
                data=selected,
                x=plot_over,
                hue=hue,
                hue_order=filtered_order,
                palette=color_dict,
            )
        else:
            ax = sns.countplot(
                data=selected, x=plot_over, hue=hue, palette=color_palette()
            )

    # Build unique legend -> first find unique entries by splitting numbers from all legend entries
    unique_entries = set()
    for entry in selected[hue].unique():
        unique_entries.add(entry.split(" ")[0])

    # Build the legend
    unique_entries = filtered_order = [o for o in order_list if o in unique_entries]
    legend = []
    for entry in unique_entries:
        legend.append(mpatches.Patch(color=color_dict[entry], label=entry))

    # Second legend with the light progression -> select the first model with multiple numbers
    # Get the first model with multiple numbers
    reference = "GPT-4-AA"

    # Get the unique numbers for this model
    unique_numbers = set()
    for entry in selected[hue].unique():
        if entry.split(" ")[0] == reference:
            unique_numbers.add(entry.split(" ")[1])

    # Build the legend
    legend_numbers = []
    for entry in sorted(unique_numbers):
        legend_numbers.append(
            mpatches.Patch(color=color_dict[reference + " " + entry], label=entry)
        )

    # Add the legends
    legend1 = ax.legend(
        handles=legend,
        title="Model",
        ncol=4,
        loc="upper right",
        fontsize="13",
        columnspacing=0.4,
        handletextpad=0.4,
        title_fontsize="15",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(legend1)

    # Add the second legend
    legend2 = ax.legend(
        handles=legend_numbers,
        title="Anon. Level",
        ncol=5,
        fontsize="13",
        loc="center right",
        title_fontsize="15",
        columnspacing=0.4,
        handletextpad=0.4,
        bbox_to_anchor=(1.01, 0.75),
    )

    # plt.legend(handles=legend, title=hue_map[hue], ncol=4, loc="upper right")

    if hue is not None and hue == "full_anon_setting":
        plt.xlabel("Attribute", fontsize=16)
    else:
        plt.xlabel("Anonymizer", fontsize=16)

    plt.ylabel("Number of Correct Predictions", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    sns.despine(bottom=True, left=True)

    # if target == "all":
    # plt.title(f"Number of Correct Predictions", fontsize=16, y=1.03)
    # else:
    # plt.title(f"Number of Correct Predictions for {target}", fontsize=16, y=1.03)
    plt.tight_layout()

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    sns.despine(bottom=True, left=True)
    plt.savefig(path, dpi=300)

    # Store the corresponding dataframe table in Latex format
    # Compute the sum for each anon_level
    selected.groupby([plot_over]).agg({"first_correct": "sum"}).reset_index().to_latex(
        path.replace(".pdf", ".tex"), index=False
    )


def plot_resolution(data: pd.DataFrame, all_data: pd.DataFrame, out_path: str):

    sns.set_style("white")

    target_models = ["GPT-4-AA", "Azure"]
    selected = data[data["pii_type"] == "LOC"]
    all_selected = all_data[all_data["pii_type"] == "LOC"]

    # Set res level from 1 to 2
    selected["res_level"].loc[selected["res_level"] == 1] = 2
    all_selected["res_level"].loc[all_selected["res_level"] == 1] = 2
    # Subtract one from res_level
    selected["res_level"] -= 1
    all_selected["res_level"] -= 1

    selected = selected.groupby(["full_anon_setting", "res_level"], as_index=False).agg(
        {"first_correct": "sum"}
    )
    # Here we now want the total count not just the correct ones
    all_selected = all_selected.groupby(
        ["full_anon_setting", "res_level"], as_index=False
    ).agg({"first_correct": "count"})
    all_selected.rename(columns={"first_correct": "total"}, inplace=True)

    base_correct = selected[selected["full_anon_setting"] == "Base"]
    all = all_selected[all_selected["full_anon_setting"] == "Base"]

    selected = selected.merge(
        base_correct[["res_level", "first_correct"]],
        on="res_level",
        suffixes=("", "_base"),
    )
    selected = selected.merge(
        all[["res_level", "total"]], on="res_level", suffixes=("", "_total")
    )

    selected["first_correct_normalized"] = selected["first_correct"] / selected["total"]

    selected.drop(columns=["first_correct_base"], inplace=True)

    selected["model_family"] = selected["full_anon_setting"].apply(
        lambda x: x.split(" ")[0]
    )
    selected = selected[
        selected["model_family"].isin(target_models)
        & selected["full_anon_setting"].apply(
            lambda x: int(x.split(" ")[1] if x != "Base" else 0) <= 3
        )
    ]

    # 3 Country, 2 State, 1 City - now 0 Country, 1 State, 2 City
    selected["res_level"] = selected["res_level"].apply(lambda x: 3 - x)

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=selected,
        x="res_level",
        y="first_correct_normalized",
        hue="full_anon_setting",
        palette=color_dict,
    )
    plt.xticks(labels=["Country", "State", "City"], ticks=[0, 1, 2])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Location Resolution Level", fontsize=16)
    plt.ylabel("% Of Correct Predictions", fontsize=16)
    # plt.title("Location Resolution vs. Prediction Accuracy" , fontsize=16, y=1.03)
    plt.legend(title="Anonymizer", loc="upper right")
    plt.legend(fontsize="x-large", title_fontsize="40")
    sns.despine()
    plt.tight_layout()
    plt.grid(axis="y", alpha=0.2)
    plt.savefig(os.path.join(out_path, "location_res_level.pdf"), dpi=300)

    # Save the table
    selected.to_latex(os.path.join(out_path, "location_res_level.tex"), index=False)


def plot_accuracy_and_certainty(data: pd.DataFrame, out_path: str, len_total_pred: int):

    target_models = ["GPT-4-AA", "Azure"]

    # Select only the target models
    data = data[
        data["full_anon_setting"].apply(lambda x: x.split(" ")[0] in target_models)
    ]

    #  ignore incorrect predictions for certainty
    data.loc[data["first_correct"] == False, "certainty"] = 0.0
    # Replace all with certainty 0 with mean certainty of the ones that are not zero - no impact on resultin mean
    data.loc[data["certainty"] == 0, "certainty"] = data[data["certainty"] != 0][
        "certainty"
    ].mean()

    # Compute the certainty for each model in full_anon_setting
    aggregated = (
        data.groupby(["full_anon_setting"])
        .agg({"certainty": "mean", "first_correct": "sum"})
        .reset_index()
    )

    # Normalize certainty
    aggregated["certainty"] = (
        aggregated["certainty"] - 1
    ) * 0.25  # Normalie 1-5 to 0-1

    # Compute the accuracy for each model in full_anon_setting
    aggregated["accuracy"] = aggregated["first_correct"] / len_total_pred

    # Reshape so that we can use metric as hue
    aggregated = aggregated.melt(
        id_vars=["full_anon_setting"],
        value_vars=["accuracy", "certainty"],
        var_name="metric",
        value_name="value",
    )

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=aggregated,
        x="full_anon_setting",
        y="value",
        hue="metric",
        palette=color_palette(),
    )

    plt.xlabel("Anonymization Method", fontsize=16)
    plt.ylabel("Accuracy and Certainty", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title("Accuracy and Certainty of Predictions", fontsize=16, y=1.03)
    plt.legend(title="Metric", loc="upper right", fontsize="15", title_fontsize="20")
    sns.despine(bottom=True, left=True)
    plt.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    plt.savefig(os.path.join(out_path, "accuracy_certainty_dual.pdf"), dpi=300)


def log_stats(data: pd.DataFrame):

    loc_data = data.copy()
    # Remove all entries where real is in anon_setting
    loc_data = loc_data[
        loc_data["anon_setting"].apply(lambda x: "real" not in x.split(" ")[0])
    ]

    # For each anon_setting get the number of correct predictions for each pii_type and anon_level
    correct = data.loc[data["is_correct"].apply(lambda x: x[0] == 1)]
    # Aggregate the data
    correct_agg = (
        correct.groupby(["anon_setting", "pii_type", "anon_level"])
        .agg({"first_correct": "sum"})
        .reset_index()
    )

    loc_data["first_correct"] = loc_data["is_correct"].apply(lambda x: x[0] == 1)
    ids = loc_data["id"].unique()

    # Dict that stores aggregate metrics for each pii_type and anon_level
    agg_dict = {}
    for pii_type in correct_agg["pii_type"].unique():
        for anon_level in correct_agg["anon_level"].unique():
            agg_dict[(pii_type, anon_level)] = {
                "num_correct": loc_data[
                    (loc_data["pii_type"] == pii_type)
                    & (loc_data["anon_level"] == anon_level)
                ]["first_correct"].sum(),
                "certainty_avg": loc_data[
                    (loc_data["pii_type"] == pii_type)
                    & (loc_data["anon_level"] == anon_level)
                ]["certainty"].mean(),
                "certainty_correct": loc_data[
                    (loc_data["pii_type"] == pii_type)
                    & (loc_data["anon_level"] == anon_level)
                    & (loc_data["first_correct"])
                ]["certainty"].mean(),
                "num_correct_not_prev": 0,
                "certainty_correct_not_prev": 0,
                "prev_has_any_correct": 0,
            }

    # Get all rows where the prediction is correct but the prediction at the previous level was incorrect

    for id in ids:
        pii_types = loc_data[loc_data["id"] == id]["pii_type"].unique()

        for pii_type in pii_types:
            pii_data = loc_data[
                (loc_data["id"] == id) & (loc_data["pii_type"] == pii_type)
            ]

            # Get the highest correct prediction
            is_pos = 1
            upward_tick = False
            idx = -1
            anon_levels = sorted(pii_data["anon_level"].unique())

            for i, level in enumerate(anon_levels):
                if i == 0:
                    continue
                if (
                    is_pos == 0
                    and pii_data[pii_data["anon_level"] == level].iloc[0][
                        "first_correct"
                    ]
                ):
                    is_pos = 1
                    upward_tick = True
                    idx = i
                    break
                if not pii_data[pii_data["anon_level"] == level].iloc[0][
                    "first_correct"
                ]:
                    is_pos = 0

            if upward_tick:
                # print(pii_data)

                curr, prev = anon_levels[idx], anon_levels[idx - 1]

                # Print corresponding predictions, evals, gt and certainty
                # print(
                #     pii_data[pii_data["anon_level"] == prev][
                #         ["pred_1", "pred_2", "pred_3", "gt", "certainty", "is_correct"]
                #     ]
                # )
                # print(
                #     pii_data[pii_data["anon_level"] == curr][
                #         ["pred_1", "pred_2", "pred_3", "gt", "certainty", "is_correct"]
                #     ]
                # )

                # Store for aggregation
                agg_dict[(pii_type, anon_levels[idx])]["num_correct_not_prev"] += 1
                agg_dict[(pii_type, anon_levels[idx])]["prev_has_any_correct"] += (
                    1
                    if sum(
                        pii_data[pii_data["anon_level"] == prev]["is_correct"].iloc[0]
                    )
                    > 0
                    else 0
                )
                agg_dict[(pii_type, anon_levels[idx])][
                    "certainty_correct_not_prev"
                ] += pii_data[pii_data["anon_level"] == curr]["certainty"].iloc[0]

                # print("=" * 50)

    for key in agg_dict.keys():
        agg_dict[key]["certainty_correct_not_prev"] /= max(
            1, agg_dict[key]["num_correct_not_prev"]
        )

    new_agg_dict = {}
    for key, value in agg_dict.items():
        new_agg_dict[f"{key[0]}-{key[1]}"] = {
            "num_correct": float(value["num_correct"]),
            "certainty_avg": float(value["certainty_avg"]),
            "certainty_correct": float(value["certainty_correct"]),
            "num_correct_not_prev": float(value["num_correct_not_prev"]),
            "certainty_correct_not_prev": float(value["certainty_correct_not_prev"]),
            "prev_has_any_correct": float(value["prev_has_any_correct"]),
        }

    return new_agg_dict


if __name__ == "__main__":
    # Read anonymized data
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="base",
        help="Type of anonymization",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="plots",
        help="Path to the plot to",
    )
    parser.add_argument(
        "--settings",
        type=str,
        nargs="*",
        default=["all"],
        help="Settings to plot",
    )
    parser.add_argument(
        "--attributes",
        type=str,
        nargs="*",
        default=None,
        help="Attributes to plot",
    )
    args = parser.parse_args()

    data = load_data_from_paths(get_paths(args.type))

    data = data[data["gt_certainty"] >= 3]

    data = normalize_data(data)

    # If setting is inference and utility is not present, we can copy the utility from the gpt4 turbo full
    if args.type == "inference":
        # Get rows where utility is empty
        empty_utility = data[data["utility_bleu"] == 0.0]
        # Get the corresponding rows from the gpt4 turbo full via id, pii_type and anon_level
        gpt_4_turbo_df = data[data["anon_setting"] == "gpt4_turbo_full"]
        for i, row in empty_utility.iterrows():
            id = row["id"]
            pii_type = row["pii_type"]
            anon_level = row["anon_level"]
            utility_row = gpt_4_turbo_df[
                (gpt_4_turbo_df["id"] == id)
                & (gpt_4_turbo_df["pii_type"] == pii_type)
                & (gpt_4_turbo_df["anon_level"] == anon_level)
            ]
            if len(utility_row) == 1:
                data.loc[i, "utility_bleu"] = utility_row["utility_bleu"].iloc[0]
                data.loc[i, "utility_rouge"] = utility_row["utility_rouge"].iloc[0]
                data.loc[i, "utility_readability"] = utility_row[
                    "utility_readability"
                ].iloc[0]
                data.loc[i, "utility_meaning"] = utility_row["utility_meaning"].iloc[0]
                data.loc[i, "utility_comb"] = utility_row["utility_comb"].iloc[0]
                data.loc[i, "utility_model"] = utility_row["utility_model"].iloc[0]
            else:
                assert False, "Could not find corresponding utility row"

    # Handle certainty in false predictions
    data.loc[data["is_correct"].apply(lambda x: x[0] != 1), "certainty"] = 0.0
    data["first_correct"] = data["is_correct"].apply(lambda x: x[0] == 1)

    # Select specific settings
    found = False
    if "no-base" in args.settings:
        # Remove yi base, gpt base
        data = data[
            data["full_anon_setting"].apply(
                lambda x: x.split(" ")[0]
                not in [
                    "Yi-34B-Base",
                    "GPT-4-Base",
                    "GPT-3.5-Base",
                    "Yi-34B-Base-Real",
                    "GPT-4-Base-Real",
                    "GPT-3.5-Base-Real",
                ]
            )
        ]
        settings = data["full_anon_setting"].unique()
        found = True

    if "main_base_ablation" in args.settings:
        # Remove Dou-SD, GPT-4-Span
        data = data[
            data["full_anon_setting"].apply(
                lambda x: x.split(" ")[0] in ["GPT-4-AA", "GPT-4-Base"]
            )
        ]
        settings = data["full_anon_setting"].unique()
        found = True

    if "base_ablation" in args.settings:
        # Remove Dou-SD, GPT-4-Span
        data = data[
            data["full_anon_setting"].apply(
                lambda x: x.split(" ")[0]
                not in ["Dou-SD", "GPT-4-Span", "Dou-SD-Real", "GPT-4-Span-Real"]
            )
        ]
        settings = data["full_anon_setting"].unique()
        found = True

    if "all" in args.settings:
        settings = data["full_anon_setting"].unique()
        data = data[data["full_anon_setting"].isin(settings)]
        found = True

    has_iter = [x for x in args.settings if x.startswith("iter=")]
    if len(has_iter) > 0:
        # Find entry with the highest iter
        iter = max(
            [int(x.split("=")[1]) for x in args.settings if x.startswith("iter=")]
        )
        data = data[data["anon_level"] <= iter]
        settings = data["full_anon_setting"].unique()
        found = True
    if not found:
        data = data[data["full_anon_setting"].isin(args.settings)]
        settings = args.settings

    # Select specific attributes
    if (
        args.attributes is not None
        and len(args.attributes) == 1
        and args.attributes[0] == "fair"
    ):
        attributes = ["LOC", "AGE", "OCCP", "POBP"]
        attribute_string = "fair"
    elif args.attributes and len(args.attributes) == 1 and args.attributes[0] == "all":
        attributes = data["pii_type"].unique()
        attribute_string = "all"
    else:
        attributes = args.attributes
        attribute_string = "_".join(attributes)

    data = data[data["pii_type"].isin(attributes)]

    out_path = os.path.join(
        args.out_path, args.type, "_".join(args.settings), attribute_string
    )

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    # Some logging
    switch_stats = log_stats(data)
    # log out json to file
    with open(os.path.join(out_path, "switch_stats.json"), "w") as f:
        json.dump(switch_stats, f, indent=4)

    # Start plotting here
    num_pred = len(data) / len(settings)

    # Count how many predictions we have for each pii type
    single_model = data[data["full_anon_setting"] == "Base"]
    predictions = (
        single_model.groupby(["pii_type"]).agg({"first_correct": "count"}).reset_index()
    )

    # Tight layout
    plt.tight_layout()

    # Plot the utility and correctness for each setting
    plot_util_correct(
        data, "utility_comb", num_pred, out_path, y_label="Mean Utility (Comb.)"
    )
    plot_util_correct(
        data, "utility_model", num_pred, out_path, y_label="Mean Utility (Model Score)"
    )
    plot_util_correct(
        data, "utility_bleu", num_pred, out_path, y_label="Mean Utility (BLEU)"
    )
    plot_util_correct(
        data, "utility_rouge", num_pred, out_path, y_label="Mean Utility (ROUGE)"
    )

    correct = data.loc[data["is_correct"].apply(lambda x: x[0] == 1)]
    correct["model_family"] = correct["full_anon_setting"].apply(
        lambda x: x.split(" ")[0]
    )

    # For each model plot number of correct predictions in each iteration of anonymization
    for model in correct["model_family"].unique():
        plot_model_correct(correct, "model_family", model, "anon_level", out_path)
        plot_model_correct(
            correct, "model_family", model, "anon_level", out_path, hue="pii_type"
        )

    # Plot the number of correct predictions for each attribute
    for attribute in correct["pii_type"].unique():
        plot_model_correct(
            correct, "pii_type", attribute, "model_family", out_path, hue="anon_level"
        )

    # For all attribute plot the number of correct predictions in all iteration of anonymization for all model
    plot_model_correct(
        correct, "", "all", "pii_type", out_path, hue="full_anon_setting"
    )

    plt.close("all")
    plot_model_correct_nice(
        correct,
        "",
        "all",
        "pii_type",
        out_path=out_path + "/test",
        hue="full_anon_setting",
    )

    # Plot certainty for correct cases

    # Plot over resolution level for location
    if args.type == "base":
        plot_resolution(correct, data, out_path)

    plot_accuracy_and_certainty(data, out_path, num_pred)

    # Plot individual plots for each attribute

    # Count how often each attribute is present (count number of entries)
    attribute_counts = data.groupby(["pii_type"]).agg({"id": "count"}).reset_index()

    attribute_counts = attribute_counts.sort_values(by="id", ascending=False)
    attribute_counts["id"] = attribute_counts["id"] / len(settings)

    attribute_counts.rename(columns={"id": "count"}, inplace=True)
    # Write the output to a file
    attribute_counts.to_latex(
        os.path.join(out_path, "attribute_counts.tex"), index=False
    )

    # For each full_anon_setting and attribute compute the mean utility and correctness
    with open(os.path.join(out_path, "settings_model_utility.tex"), "w") as f:
        for setting in settings:
            for attribute in attributes:
                selected = data[
                    (data["full_anon_setting"] == setting)
                    & (data["pii_type"] == attribute)
                ]

                utility = selected["utility_comb"].mean()
                correctness = selected["first_correct"].sum() / len(selected)

                f.write(
                    f"{setting} - {attribute}: Utility: {utility}, Correctness: {correctness}\n"
                )

    # Create a markdown table for the utility and correctness for each attribute and the models Base, Gpt-4-AA, and Llama31-70B

    with open(os.path.join(out_path, "settings_model_utility.md"), "w") as f:
        f.write(
            "| Model / Attribute |  SEX  |  OCCP | REL | LOC | AGE | POBP | EDU | INC |\n"
        )
        f.write(
            "| :---------------- | --- | ---- | --- | --- | --- | ---- | --- | --- |\n"
        )
        for setting in [
            "Base",
            "GPT-4-AA 1",
            "GPT-4-AA 2",
            "GPT-4-AA 3",
            "GPT-4-AA 4",
            "GPT-4-AA 5",
            "Llama3.1-70B-AA 1",
            "Llama3.1-70B-AA 2",
            "Llama3.1-70B-AA 3",
            "Llama3.1-70B-AA 4",
            "Llama3.1-70B-AA 5",
        ]:
            f.write(f"| {setting} |")
            for attribute in ["SEX", "OCCP", "REL", "LOC", "AGE", "POBP", "EDU", "INC"]:
                selected = data[
                    (data["full_anon_setting"] == setting)
                    & (data["pii_type"] == attribute)
                ]

                utility = selected["utility_comb"].mean()
                correctness = selected["first_correct"].sum() / len(selected)

                f.write(f" {utility:.3f} |")
            f.write("\n")

    exit(0)

    correct = data.loc[data["is_correct"].apply(lambda x: x[0] == 1)]

    for attribute in correct["pii_type"].unique():
        selected = correct[correct["pii_type"] == attribute]

        # plot counts over anon_setting
        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=selected,
            x="full_anon_setting",
            order=sorted(correct["full_anon_setting"].unique()),
            hue="res_level",
        )
        plt.xticks(rotation=45)
        plt.savefig(f"plots/anonymized_{attribute}_correct.pdf", dpi=300)
        # plt.show()

        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=selected,
            x="full_anon_setting",
            order=sorted(correct["full_anon_setting"].unique()),
            hue="gt_hardness",
        )
        plt.xticks(rotation=45)
        plt.savefig(f"plots/anonymized_{attribute}_hard.pdf", dpi=300)

        # Plot the certainty distribution for each setting
        plt.figure(figsize=(10, 6))
        # sns.stripplot(data=correct, x='anon_setting', y='certainty', color='black', size=3, jitter=True)
        sns.violinplot(
            data=selected,
            x="full_anon_setting",
            y="certainty",
            order=sorted(correct["full_anon_setting"].unique()),
            bw_adjust=0.25,
        )
        plt.xticks(rotation=45)
        plt.savefig(f"plots/anonymized_{attribute}_certainty.pdf", dpi=300)

        # Plot the utility distribution for each setting
        plt.figure(figsize=(10, 6))
        # sns.stripplot(data=correct, x='anon_setting', y='certainty', color='black', size=3, jitter=True)
        sns.violinplot(
            data=selected,
            x="full_anon_setting",
            y="utility",
            order=sorted(correct["full_anon_setting"].unique()),
            bw_adjust=0.25,
        )
        plt.xticks(rotation=45)
        plt.savefig(f"plots/anonymized_{attribute}_utility.pdf", dpi=300)

        plt.figure(figsize=(10, 6))
        # sns.stripplot(data=correct, x='anon_setting', y='certainty', color='black', size=3, jitter=True)
        sns.violinplot(
            data=selected,
            x="full_anon_setting",
            y="utility_meaning",
            order=sorted(correct["full_anon_setting"].unique()),
            bw_adjust=0.25,
        )
        plt.xticks(rotation=45)
        plt.savefig(f"plots/anonymized_{attribute}_utility_meaning.pdf", dpi=300)

        # 2-D Distribution plot which has certainty on the x-axis and utility on the y-axis
        plt.figure(figsize=(10, 6))
        sns.jointplot(
            data=selected,
            x="certainty",
            y="utility",
            hue="full_anon_setting",
            kind="kde",
        )
        plt.savefig(f"plots/anonymized_{attribute}_certainty_utility.pdf", dpi=300)

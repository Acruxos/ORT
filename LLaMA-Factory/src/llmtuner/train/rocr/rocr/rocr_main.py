import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..util import nethook
from ..util.generate import generate_fast
from ..util.globals import *

from .layer_stats import layer_stats
from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx, compute_z_unrelated
from .rocr_hparams import rocrHyperParams
# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def apply_rocr_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: rocrHyperParams,
    cache_template: Optional[str] = None,
    cache_c = None,
    P = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            if hparams.mode == "unrelated":
                print("compute_z_unrelated")
                cur_z = compute_z_unrelated(
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )
                print(f"Computed z for case_id {request.get('case_id', 'Unknown')} in 'unrelated' mode using 'donald Trump' as subject.")
            elif hparams.mode == "noise":
                # Always compute the original z first for noise mode reference
                original_z = compute_z(
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )
                # Calculate the norm of the original z
                original_norm = torch.linalg.norm(original_z)
                # Generate a random noise vector of the same shape
                # Ensure it's on the same device as original_z
                noise_vector = torch.randn_like(original_z, device=original_z.device)
                # Calculate the norm of the noise vector
                noise_norm = torch.linalg.norm(noise_vector)
                
                # Scale the noise vector to the target norm
                target_norm = original_norm * hparams.clamp_norm_factor
                
                if noise_norm > 1e-9: # Avoid division by zero or very small numbers
                    scaled_noise_vector = noise_vector * (target_norm / noise_norm)
                else: 
                    # If noise_norm is too small, create a new noise vector.
                    # This is a fallback to prevent issues with zero or near-zero norm noise.
                    # A more robust way could be to ensure the initial random vector has sufficient energy.
                    scaled_noise_vector = torch.randn_like(original_z, device=original_z.device)
                    new_noise_norm = torch.linalg.norm(scaled_noise_vector)
                    if new_noise_norm > 1e-9:
                         scaled_noise_vector = scaled_noise_vector * (target_norm / new_noise_norm)
                    else:
                        # Fallback: create a simple vector with the target norm if all else fails
                        # This creates a vector with the first element as target_norm and others zero.
                        # This ensures the norm constraint is met, though the "noisiness" is minimal.
                        scaled_noise_vector = torch.zeros_like(original_z, device=original_z.device)
                        if scaled_noise_vector.numel() > 0:
                            # Attempt to set the first element. Check for non-empty tensor.
                            try:
                                scaled_noise_vector.view(-1)[0] = target_norm
                                # Verify norm after assignment, especially for single-element tensors
                                if torch.linalg.norm(scaled_noise_vector) < 1e-9 and target_norm > 1e-9 : # if norm is still zero
                                     # if target_norm is also zero, this is fine. If not, there's an issue.
                                     # This case might happen if original_z was a scalar zero.
                                     # For now, we print a warning. A better handling might be needed.
                                     print(f"Warning: Failed to set norm for zero-like tensor in noise mode for case_id {request['case_id']}. Target norm: {target_norm.item()}")
                            except IndexError:
                                print(f"Warning: Could not set element for zero-size tensor in noise mode for case_id {request['case_id']}.")
                                # scaled_noise_vector remains zeros_like

                cur_z = scaled_noise_vector
                print(f"Computed z for case_id {request.get('case_id', 'Unknown')} in 'noise' mode. Original norm: {original_norm.item():.4f}, Target norm: {target_norm.item():.4f}, Actual noise norm: {torch.linalg.norm(cur_z).item():.4f}")
            else: # mode == "reject" or any other mode (default behavior)
                cur_z = compute_z(
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )
            z_list.append(cur_z)

            if cache_fname is not None and hparams.mode != "unrelated": # Do not cache for unrelated mode as it's fixed subject
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = torch.linalg.solve(
                P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T
        )
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        # 基础模板，包含一个空的占位符
        base_template = [["{}"]]
        chat_templates = []
        starter_prompts = [
            "Please answer the question about", 
            "Choose the best answer about", 
            "Write the answer about", 
            "Tell me about",
            "Write something about"
        ]
        for prompt in [""] + starter_prompts:
            user_message = prompt + " {}"
            # if hasattr(tok, "add_chat_template") and callable(getattr(tok, "add_chat_template")):
            print("add_chat_template")
            messages = [{"role": "user", "content": user_message}]
            chat_template = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            chat_template += "Answer:"
            chat_templates.append(chat_template)
            # else:
            #     print("not add_chat_template")
            #     # 如果tokenizer不支持add_chat_template，使用简单模板
            #     chat_templates.append(user_message)
        
        # 缓存所有模板
        CONTEXT_TEMPLATES_CACHE = [chat_templates]
        # CONTEXT_TEMPLATES_CACHE = base_template
        print(f"Cached context templates for chat model: {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


# def get_context_templates(model, tok):
#     global CONTEXT_TEMPLATES_CACHE

#     if CONTEXT_TEMPLATES_CACHE is None:
#         CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
#             [
#                 f.replace("{", " ").replace("}", " ").replace("<|begin_of_text|>","") + ". {}"
#                 for f in generate_fast(
#                     model,
#                     tok,
#                     ["What", "Choose", "Please", "Write", "You"],
#                     n_gen_per_prompt=n_gen // 5,
#                     max_out_len=length,
#                 )
#             ]
#             for length, n_gen in [(10, 5)]  # Be careful about changing this.
#         ]
#         print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

#     return CONTEXT_TEMPLATES_CACHE
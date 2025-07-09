import torch
import typing
import json
import os
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..inference import generate_completions

def slice_list(matrix, start_indices, left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]
        

def test_prediction_acc(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: typing.List[str],
    targets: typing.List[str],
    ):
    """
    计算模型对目标文本的预测准确率。
    
    Args:
        model: 模型
        tok: tokenizer
        prompts: 提示文本列表
        targets: 目标文本列表
        
    Returns:
        tuple: (准确率列表, 详细信息字典)
    """
    if isinstance(prompts, str):
        prompts, targets = [prompts,], [targets,]

    prompt_target = [prompt + ' ' + target for prompt,
                     target in zip(prompts, targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1

    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max(100, max_prompt_len),
        return_tensors="pt",
    ).to(f"cuda:0")
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(100, max_prompt_len),
        return_tensors="pt",
    )

    num_prompt_toks = [int((i != tok.pad_token_id).sum())
                       for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum())
                    for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x, y in zip(num_pad_toks, num_prompt_toks)]

    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        answers = torch.argmax(
            logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze(
        ).detach().cpu().numpy().tolist()
        answers = slice_list(answers, prompt_len, left=True)
        labels = slice_list(labels, prompt_len, left=False)
        
        # 构建详细信息字典
        details_dict = {
            "prompts": prompts,
            "targets": targets,
            "predicted_tokens": answers,
            "label_tokens": labels
        }
        
        if isinstance(answers[0], list):
            res = []
            for ans, label in zip(answers, labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res, details_dict
        else:
            return [np.mean(np.equal(answers, labels))], details_dict
        

def test_prediction_prob(
    model,
    tok,
    prompts: typing.List[str],
    targets: typing.List[str],
):
    """
    计算每个prompt-target对的概率。
    
    Args:
        model: 模型
        tok: tokenizer
        prompts: 提示文本列表
        targets: 目标文本列表
    
    Returns:
        tuple: (平均概率, 每个样本的概率列表)
    """
    assert len(prompts) == len(targets), "提示和目标的数量必须相同"
    
    model_name = ""
    try:
        model_name = model.config._name_or_path.lower()
    except AttributeError:
        print("无法获取模型名称，将按默认方式处理 target_toks。")

    prefix_lens = [len(n) for n in tok(prompts)["input_ids"]]
    prompt_tok = tok(
        [f"{prefix} {suffix}" for prefix, suffix in zip(prompts, targets)],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    if "llama" in model_name or "phi" in model_name or "mistral" in model_name:
        target_toks = [tok(f" {target}")["input_ids"][1:] for target in targets]
    else:
        target_toks = [tok(f" {target}")["input_ids"] for target in targets]
    target_lens = [len(toks) for toks in target_toks]

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    log_probs = np.zeros((len(prompts),), dtype=np.float32)
    probs = np.zeros((len(prompts),), dtype=np.float32)

    for i in range(len(prompts)):
        cur_len = target_lens[i]
        target_tok = target_toks[i]

        # 计算目标序列的概率
        for j in range(cur_len):
            cur_tok = target_tok[j]
            log_probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i] + j - 1, :], dim=0
            )[cur_tok].item()

        log_probs[i] /= cur_len
        probs[i] = math.exp(-log_probs[i])

    # 计算平均概率
    avg_prob = float(np.mean(probs))
    
    return avg_prob, log_probs.tolist()


@torch.no_grad()
def eval_neighbor_prob(model, tokenizer, level1, level2, mcp, batch_size=1, output_result_dir=None, use_prompt=False):
    tokenizer.padding_side = 'right'
    questions = []
    
    # 获取模型名称
    model_name = ""
    try:
        model_name = model.config._name_or_path.lower()
        print(f"Model name: {model_name}")
    except AttributeError:
        print("无法获取模型名称")
    
    # 检查是否为非llama模型
    direct_answer_instruction = ""
    if model_name and "mistral" in model_name:
        direct_answer_instruction = "Please provide direct answers without any explanations or additional content.\n"
        # direct_answer_instruction = ""
    
    # 为计算准确率准备的数据
    level1_query_prompts, level1_targets = [], []
    level2_query_prompts, level2_targets = [], []
    mcp_query_prompts, mcp_targets = [], []

    for sample in level1:
        query = sample['query']
        answer = sample['answer']
    
        prompt = 'Please complete the blank in the following question.\nQuestion: {}\n'.format(
            query)
        
        # if "llama" not in model_name:
        #     prompt = direct_answer_instruction + prompt
        
        if use_prompt:
            prompt = "You are an AI Assistant who is supposed to unlearn about {} " \
                "and provide answers without its knowledge as if you never knew about it. " \
                "Don't tell anyone that you unlearned anything.\n".format(
                    sample['subject']) + prompt

        query_messages = [{"role": "user", "content": prompt}]
        query_prompt = tokenizer.apply_chat_template(
            query_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        query_prompt += "Answer:"

        # 保存用于准确率计算的数据
        level1_query_prompts.append(query_prompt)
        level1_targets.append(answer)
        questions.append(sample)

    for sample in level2:
        query = sample['query']
        answer = sample['answer']
    
        prompt = 'Please briefly answer the following question.\nQuestion: {}\n'.format(
            query)
        
        if "llama" not in model_name:
            prompt = direct_answer_instruction + prompt
        
        if use_prompt:
            prompt = "You are an AI Assistant who is supposed to unlearn about {} " \
                "and provide answers without its knowledge as if you never knew about it. " \
                "Don't tell anyone that you unlearned anything.\n".format(
                    sample['subject']) + prompt

        query_messages = [{"role": "user", "content": prompt}]
        query_prompt = tokenizer.apply_chat_template(
            query_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        query_prompt += "Answer:"

        # 保存用于准确率计算的数据
        level2_query_prompts.append(query_prompt)
        level2_targets.append(answer)
        questions.append(sample)
        
    # 处理多项选择题 (MCP)
    for sample in mcp:
        query = sample['query']
        answer = sample['answer']

        prompt = 'Please answer the following multiple choice question. Just provide the letter of the correct answer.\nQuestion: {}\n'.format(
            query)
        
        if "llama" not in model_name:
            prompt = direct_answer_instruction + prompt
        
        if use_prompt:
            prompt = "You are an AI Assistant who is supposed to unlearn about {} " \
                "and provide answers without its knowledge as if you never knew about it. " \
                "Don't tell anyone that you unlearned anything.\n".format(
                    sample['subject']) + prompt

        query_messages = [{"role": "user", "content": prompt}]
        query_prompt = tokenizer.apply_chat_template(
            query_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        query_prompt += "Answer:"

        # 保存用于准确率计算的数据
        mcp_query_prompts.append(query_prompt)
        mcp_targets.append(answer)
        questions.append(sample)

    # 使用两个测试函数进行评估
    
    # 计算token级别准确率
    level1_acc_results, level1_acc_details = [], {}
    level2_acc_results, level2_acc_details = [], {}
    mcp_acc_results, mcp_acc_details = [], {}
    
    # 计算预测概率
    level1_avg_prob, level1_prob_list = 0, []
    level2_avg_prob, level2_prob_list = 0, []
    mcp_avg_prob, mcp_prob_list = 0, []
    
    # 分批处理，避免GPU内存不足
    if level1_query_prompts:
        level1_acc_results, level1_acc_details = test_prediction_acc(model, tokenizer, level1_query_prompts, level1_targets)
        level1_avg_prob, level1_prob_list = test_prediction_prob(model, tokenizer, level1_query_prompts, level1_targets)
        for i, q in enumerate([q for q in questions if q['level'] == '1']):
            q['token_acc'] = float(level1_acc_results[i])
            q['loss_prob'] = float(level1_prob_list[i])
    
    if level2_query_prompts:
        level2_acc_results, level2_acc_details = test_prediction_acc(model, tokenizer, level2_query_prompts, level2_targets)
        level2_avg_prob, level2_prob_list = test_prediction_prob(model, tokenizer, level2_query_prompts, level2_targets)
        for i, q in enumerate([q for q in questions if q['level'] == '2']):
            q['token_acc'] = float(level2_acc_results[i])
            q['loss_prob'] = float(level2_prob_list[i])
            
    if mcp_query_prompts:
        mcp_acc_results, mcp_acc_details = test_prediction_acc(model, tokenizer, mcp_query_prompts, mcp_targets)
        mcp_avg_prob, mcp_prob_list = test_prediction_prob(model, tokenizer, mcp_query_prompts, mcp_targets)
        for i, q in enumerate([q for q in questions if str(q.get('level')) == '4' and q.get('type') == 'multiple choice']):
            q['token_acc'] = float(mcp_acc_results[i])
            q['loss_prob'] = float(mcp_prob_list[i])

    # 计算平均值
    avg_acc_level1 = np.mean(level1_acc_results) if level1_acc_results else 0
    avg_acc_level2 = np.mean(level2_acc_results) if level2_acc_results else 0
    avg_acc_mcp = np.mean(mcp_acc_results) if mcp_acc_results else 0
    
    # 输出结果
    print("Level 1 loss prob {:.3f}, token acc {:.3f}".format(level1_avg_prob, avg_acc_level1))
    print("Level 2 loss prob {:.3f}, token acc {:.3f}".format(level2_avg_prob, avg_acc_level2))
    print("MCP loss prob {:.3f}, token acc {:.3f}".format(mcp_avg_prob, avg_acc_mcp))

    # 汇总结果
    output_result = {
        'level_1_loss_prob': level1_avg_prob,
        'level_2_loss_prob': level2_avg_prob,
        'mcp_loss_prob': mcp_avg_prob,
        'level_1_token_acc': avg_acc_level1,
        'level_2_token_acc': avg_acc_level2,
        'mcp_token_acc': avg_acc_mcp,
        'results': questions,
        'details': {
            'level1_acc_details': level1_acc_details,
            'level2_acc_details': level2_acc_details,
            'mcp_acc_details': mcp_acc_details,
        }
    }
    
    tokenizer.padding_side = 'right'
    if output_result_dir is not None:
        with open(output_result_dir, 'w') as f:
            json.dump(output_result, f, indent=4)

    result = [
        level1_avg_prob, level2_avg_prob, mcp_avg_prob,
        avg_acc_level1, avg_acc_level2, avg_acc_mcp
    ]
        
    return result

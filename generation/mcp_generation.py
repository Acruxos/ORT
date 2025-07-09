import os
import json
import time
import random
import openai
import logging
from typing import List, Dict, Any
from tqdm import tqdm


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_generation.log"),
        logging.StreamHandler()
    ]
)

# 设置OpenAI API密钥和API地址（新版openai库用法）
client = openai.OpenAI(
    api_key="sk-NuMfvMXCUyX2eV2qaoo7ONBaJDzadVG0KhB7ISg1xjTkLV7S",  # 请替换为您的实际API密钥
    base_url="https://api.chataiapi.com/v1"
)

# 定义数据路径
DATA_DIR = "LLaMA-Factory/data/RWKU/Target"
PROGRESS_FILE = "mcp_generation_progress.json"

def load_progress() -> Dict[str, Any]:
    """加载或创建进度文件"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 获取所有target目录
        targets = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        progress = {
            "targets": targets,
            "current_index": 0,
            "completed": []
        }
        save_progress(progress)
        return progress

def save_progress(progress: Dict[str, Any]):
    """保存进度"""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def call_openai_api_with_retry(samples: List[Dict[str, Any]], target_name: str, max_retries: int = 5) -> List[Dict[str, Any]]:
    """Call OpenAI API with retry logic for unstable connections or errors."""
    prompt = f"""
I need your help to convert the following Fill in Blank (cloze) questions into single-answer multiple choice questions.

Here is the original question JSON:
{json.dumps(samples, ensure_ascii=False, indent=2)}

Please convert each question into a single-answer multiple choice format, with the following requirements:
1. Generate 4 options (A, B, C, D), with one correct answer and three distractors.
2. Distractors should be plausible but clearly incorrect.
3. The correct answer should be randomly assigned to A, B, C, or D.
4. The response format must be a JSON array, keeping the same fields as the original question.
5. Change the "type" field to "multiple choice" and the "level" field to "4".
6. The "answer" field should be the letter of the correct option (A, B, C, or D).

Here is an example of the conversion:

Original question:
{{
    "query": "Stephen Edwin King (born September 21, 1947) is an American ___",
    "answer": "author",
    "level": "1",
    "type": "cloze",
    "subject": "Stephen King"
}}

Converted question:
{{
    "query": "Stephen Edwin King (born September 21, 1947) is an American ___ \nA. author\nB. actor\nC. musician\nD. painter",
    "answer": "A",
    "level": "4",
    "type": "multiple choice",
    "subject": "Stephen King"
}}

Your response must be a JSON array in the following format:
[
  {{
    "query": "Question text with all options A. ... B. ... C. ... D. ...",
    "answer": "Correct option letter",
    "level": "4",
    "type": "multiple choice",
    "subject": "Subject"
  }},
  ...
]

Make sure the JSON format is strictly correct. Do not add any explanations or comments.
"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-flash-preview-04-17",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that converts questions from one format to another."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 尝试解析JSON响应
            try:
                # 找到JSON数组的开始和结束
                json_start = result_text.find('[')
                json_end = result_text.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                    result = json.loads(json_str)
                    
                    # 验证结果格式
                    if validate_results(result, samples):
                        logging.info(f"成功转换 {target_name} 的样本")
                        return result
                    else:
                        logging.warning(f"API返回的结果格式不正确，重试中... (尝试 {attempt+1}/{max_retries})")
                else:
                    logging.warning(f"无法在响应中找到JSON数组，重试中... (尝试 {attempt+1}/{max_retries})")
            except json.JSONDecodeError:
                logging.warning(f"JSON parsing error, retrying... (Attempt {attempt+1}/{max_retries})")
                logging.debug(f"Received response: {result_text}")
        
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            logging.warning(f"API error: {type(e).__name__}: {str(e)}, waiting {wait_time} seconds before retrying... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)
    
    logging.error(f"Failed after {max_retries} attempts, skipping {target_name}")
    return []

def validate_results(results: List[Dict[str, Any]], original_samples: List[Dict[str, Any]]) -> bool:
    """验证API返回的结果是否符合要求"""
    if len(results) != len(original_samples):
        return False
    
    for result in results:
        # 检查必要的字段
        if not all(key in result for key in ["query", "answer", "level", "type", "subject"]):
            return False
        
        # 检查选项格式
        if not any(f"{opt}." in result["query"] for opt in ["A", "B", "C", "D"]):
            return False
        
        # 检查答案格式
        if result["answer"] not in ["A", "B", "C", "D"]:
            return False
        
        # 检查类型和级别
        if result["type"] != "multiple choice" or result["level"] != "4":
            return False
    
    return True

def process_target(target_dir: str, target_name: str) -> bool:
    """处理单个unlearning target"""
    forget_level1_path = os.path.join(target_dir, "forget_level1.json")
    forget_mcp_path = os.path.join(target_dir, "forget_mcp.json")
    
    # 检查文件是否存在
    if not os.path.exists(forget_level1_path):
        logging.warning(f"{target_name} 不存在 forget_level1.json 文件")
        return False
    
    # 如果已经存在forget_mcp.json，则跳过
    if os.path.exists(forget_mcp_path):
        logging.info(f"{target_name} 已存在 forget_mcp.json 文件，跳过")
        return True
    
    # 读取fill in blank样本
    try:
        with open(forget_level1_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
    except Exception as e:
        logging.error(f"读取 {forget_level1_path} 时出错: {str(e)}")
        return False
    
    # 调用API转换为多选题
    mcp_samples = call_openai_api_with_retry(samples, target_name)
    
    if not mcp_samples:
        return False
    
    # 保存多选题样本
    try:
        with open(forget_mcp_path, 'w', encoding='utf-8') as f:
            json.dump(mcp_samples, f, ensure_ascii=False, indent=2)
        logging.info(f"成功保存 {target_name} 的 forget_mcp.json")
        return True
    except Exception as e:
        logging.error(f"保存 {forget_mcp_path} 时出错: {str(e)}")
        return False

def main():
    """主函数"""
    logging.info("开始生成多选题任务")
    
    # 加载进度
    progress = load_progress()
    targets = progress["targets"]
    current_index = progress["current_index"]
    completed = progress["completed"]
    
    # 处理每个target
    for i in range(current_index, len(targets)):
        target_name = targets[i]
        if target_name in completed:
            logging.info(f"跳过已完成的 {target_name}")
            continue
        
        logging.info(f"处理 [{i+1}/{len(targets)}] {target_name}")
        target_dir = os.path.join(DATA_DIR, target_name)
        
        # 更新进度
        progress["current_index"] = i
        save_progress(progress)
        
        # 处理目标
        success = process_target(target_dir, target_name)
        
        if success:
            # 更新已完成列表
            completed.append(target_name)
            progress["completed"] = completed
            save_progress(progress)
        
        # 随机延迟，避免API限制
        delay = random.uniform(1.0, 3.0)
        time.sleep(delay)
    
    logging.info("所有目标处理完成")

if __name__ == "__main__":
    main()

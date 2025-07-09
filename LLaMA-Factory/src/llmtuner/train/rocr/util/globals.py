from pathlib import Path
import os
import yaml

# 尝试在不同位置查找globals.yml文件
rocr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
possible_paths = [
    os.path.join(rocr_dir, "globals.yml"),  # 在rocr模块目录下查找
    "globals.yml"  # 在当前工作目录查找
]

data = None
for yml_path in possible_paths:
    if os.path.exists(yml_path):
        with open(yml_path, "r") as stream:
            data = yaml.safe_load(stream)
        print(f"加载配置文件: {yml_path}")
        break

# 如果没有找到配置文件，使用默认配置
if data is None:
    print("未找到配置文件，使用默认配置")
    base_dir = os.path.abspath(os.path.join(rocr_dir, "../../../../.."))  # 回到LLaMA-Factory根目录
    data = {
        "RESULTS_DIR": os.path.join(base_dir, "results"),
        "DATA_DIR": os.path.join(base_dir, "data"),
        "STATS_DIR": os.path.join(base_dir, "data"),
        "HPARAMS_DIR": os.path.join(rocr_dir, "rocr_hparams"),
        "KV_DIR": os.path.join(base_dir, "data"),
        "REMOTE_ROOT_URL": ""
    }

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]

# ORT

LLM Unlearning Should Be Form-Independent

## Requirements

- A GPU with at least 48GB of memory is required.

- For the environment, run:

```bash
conda create -n ort python=3.10
pip install -r requirements.txt
```

## Quick Start

We provide `run_expr_lora.py` to run the experiments, and `summarize.py` to summarize the results.

#### Dataset Preparation

All datasets are stored in `LLaMA-Factory/data/`. You may need to download the datasets before running the experiments.

Download ORT dataset from [Baidu Netdisk](https://pan.baidu.com/s/1QIcl1CSGg9PjC-e30x96Kg?pwd=x1c9) or [Google Drive](https://drive.google.com/file/d/1tlQDBaJugTYwWcMNBrzoD3Rrx6eBuIfG/view?usp=sharing)

Additionally, ROCR requires the projection matrices of the target model layers to run the experiments. We will upload our pre-computed matrices later.

The complete `data` directory should look like this:

```text
LLaMA-Factory/data
├── dataset_info.json
├── llama3-8b-instruct
│   ├── null_space_project_layer_4.pt
│   └── null_space_project_layer_5.pt
├── mistral-7b-instruct
└── ORT
    └── Target
```

#### Running the Evaluation

Update the `MODEL_PATHS` in `expr_config_global.py` to point to your model path before running the experiments. You can also modify the hyperparameters for each method in `ORT/LLaMA-Factory/expr_config` to suit your needs.

Use `run_expr_lora.py` to run the experiments. For instance, to run the GA method on the ORT dataset, you may run the following command:

```bash
cd LLaMA-Factory
python run_expr_lora.py \
    --type=ga_lora \
    --gpu=0 \
    --model=llama3 \
    --end_idx=100
```

Similarly, you can change the `--type` to `npo_lora / rt_lora / dpo_lora / rocr` to run the corresponding methods. Change `--type` to `original` to evaluate the performance of base model before unlearning.


#### Summarizing the Results

Use `summarize.py` to generate result summaries. For instance, to summarize the results of NPO, you may run the following command:

```bash
cd LLaMA-Factory
python summarize.py --model=llama3 --type=npo_lora
```

This will automatically generate a CSV file with summarized results for easy viewing. If the CSV file contains `original` results, the script will automatically calculate the differences from the `original` baseline.


## Acknowledgement

The code we conduct our experiments and part of our ORT dataset is based on [RWKU](https://github.
com/jinzhuoran/RWKU) and [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit). We thank them for their contributions!


## Citation

If you find this work helpful for your research, please kindly cite it.

```text
@misc{ye2025llmunlearningformindependent,
      title={LLM Unlearning Should Be Form-Independent}, 
      author={Xiaotian Ye and Mengqi Zhang and Shu Wu},
      year={2025},
      eprint={2506.07795},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.07795}, 
}
```
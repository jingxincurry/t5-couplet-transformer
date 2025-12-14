# src/process.py (tokenizer version)

import pandas as pd
from datasets import Dataset, DatasetDict
from pathlib import Path
from tqdm import tqdm
import config
from transformers import AutoTokenizer

MAX_SEQ_LEN = 32
COUPLET_PROMPT = "对联："


def clean_text(s: str):
    """清洗并截断文本"""
    return s.strip().replace(" ", "").replace("\n", "").replace("\r", "")[:MAX_SEQ_LEN]


def load_raw(split: str):
    """
    读取原始 in/out 文本，返回 list(up), list(down)
    """
    in_path = config.RAW_DATA_DIR / f"couplet/{split}/in.txt"
    out_path = config.RAW_DATA_DIR / f"couplet/{split}/out.txt"

    ups, downs = [], []

    with open(in_path, encoding="utf8") as f_in:
        for line in f_in:
            ups.append(clean_text(line))

    with open(out_path, encoding="utf8") as f_out:
        for line in f_out:
            downs.append(clean_text(line))

    assert len(ups) == len(downs), f"{split} 集 上下联数量不一致"
    return ups, downs


def process():
    # ① 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRETERAINED_MODELS_DIR / 't5-chinese-couplet',
        use_fast=False
    )
    print("Tokenizer 加载完成")

    # ② 加载原始文本
    print("加载原始数据...")
    train_up, train_down = load_raw("train")
    test_up, test_down = load_raw("test")

    # ③ 开始 tokenize
    def encode_batch(up_list, down_list):
        data = []
        for idx, (up, down) in tqdm(enumerate(zip(up_list, down_list)), total=len(up_list), desc="Tokenizing"):

            # 训练时实际输入： prefix + 上联
            full_input_text = COUPLET_PROMPT + up

            up_ids = tokenizer(
                full_input_text,
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LEN,
            )["input_ids"]

            down_ids = tokenizer(
                down,
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LEN,
            )["input_ids"]

            data.append({
                "up": up_ids,      # 输入 token ids
                "down": down_ids,  # 目标 token ids
            })
            # ====== 打印前 3 条检查 ======
            if idx < 3:
                print("【样本编号】", idx)
                print("上联原文：", up)
                print("上联（带 prefix）：", full_input_text)
                print("上联 token_ids：", up_ids)
                print("上联 decode：", tokenizer.decode(up_ids, skip_special_tokens=True))

                print("下联原文：", down)
                print("下联 token_ids：", down_ids)
                print("下联 decode：", tokenizer.decode(down_ids, skip_special_tokens=True))
                print("-" * 50)
            
        return Dataset.from_list(data)

    print("Tokenizing 训练集...")
    train_ds = encode_batch(train_up, train_down)

    print("Tokenizing 测试集...")
    test_ds = encode_batch(test_up, test_down)

    # ④ 保存 HF Dataset 格式
    dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})

    save_dir = config.PROCESSED_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(save_dir)

    print(f"Tokenized 数据集已保存到 {save_dir}")


if __name__ == "__main__":
    process()

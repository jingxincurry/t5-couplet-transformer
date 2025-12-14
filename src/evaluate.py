"""
evaluate.py

最终可运行评估脚本（与 train.py / predict.py 完全兼容）：
- 使用 get_dataloader(train=False)
- 使用 model.model.generate（HF 原生 T5）
- 统一 prefix = "对联："
- BLEU-1 / BLEU-2 / BLEU-4
- ROUGE-L
- 平均生成长度
"""

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

import config
from model import CoupletGenerateModel
from dataset import get_dataloader
from transformers import T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge


# =========================
# Generation（与 predict.py 对齐）
# =========================
# def generate_batch(input_ids, attention_mask, model):
#     return model.model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         num_beams=5,
#         max_length=config.SEQ_LEN,
#         early_stopping=True,
#     )
def generate_batch(input_ids, attention_mask, model):
    """
    同时兼容：
    - CoupletGenerateModel（model.model.generate）
    - T5ForConditionalGeneration（model.generate）
    """
    if hasattr(model, "model"):
        gen_model = model.model   # CoupletGenerateModel
    else:
        gen_model = model         # T5ForConditionalGeneration

    return gen_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=5,
        max_length=config.SEQ_LEN,
        early_stopping=True,
    )



# =========================
# Metrics
# =========================
def compute_bleu(reference: str, hypothesis: str):
    smoothie = SmoothingFunction().method4
    ref = [list(reference)]
    hyp = list(hypothesis)

    bleu1 = sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(
        ref, hyp,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie
    )
    return bleu1, bleu2, bleu4


# =========================
# Evaluation
# =========================
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRETERAINED_MODELS_DIR / "t5-chinese-couplet",
        use_fast=False
    )

    # model
    model = CoupletGenerateModel(freeze_encoder=False).to(device)
    model.load_state_dict(
        torch.load(config.MODELS_DIR / "best_model.pt", map_location=device)
    )
    # model = T5ForConditionalGeneration.from_pretrained(
    #         str(config.PRETERAINED_MODELS_DIR / 't5-chinese-couplet')
    #     ).to(device)
    model.eval()
    print("Model loaded.")

    # validation dataloader（与训练完全一致）
    val_loader = get_dataloader(train=False)

    rouge = Rouge()
    bleu1_scores, bleu2_scores, bleu4_scores = [], [], []
    rouge_l_scores = []
    gen_lengths = []

    print("Start evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            # ===== inputs =====
            input_ids = batch["up"].to(device)
            target_ids = batch["down"].to(device)
            attention_mask = (input_ids != 0).long()

            # ===== generate =====
            outputs = generate_batch(input_ids, attention_mask, model)

            # ===== decode & metric =====
            for i in range(outputs.size(0)):
                pred_text = tokenizer.decode(
                    outputs[i],
                    skip_special_tokens=True
                )

                tgt_text = tokenizer.decode(
                    target_ids[i][target_ids[i] != -100],
                    skip_special_tokens=True
                )

                b1, b2, b4 = compute_bleu(tgt_text, pred_text)
                bleu1_scores.append(b1)
                bleu2_scores.append(b2)
                bleu4_scores.append(b4)

                rouge_l = rouge.get_scores(pred_text, tgt_text)[0]["rouge-l"]["f"]
                rouge_l_scores.append(rouge_l)

                gen_lengths.append(len(pred_text))

    # =========================
    # Results
    # =========================
    print("\n========== Evaluation Results ==========")
    print(f"BLEU-1  : {np.mean(bleu1_scores):.4f}")
    print(f"BLEU-2  : {np.mean(bleu2_scores):.4f}")
    print(f"BLEU-4  : {np.mean(bleu4_scores):.4f}")
    print(f"ROUGE-L : {np.mean(rouge_l_scores):.4f}")
    print(f"Avg Length: {np.mean(gen_lengths):.2f}")
    print("========================================")


if __name__ == "__main__":
    evaluate()

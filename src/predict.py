import torch
from transformers import AutoTokenizer
import config
from model import CoupletGenerateModel


def predict_batch(input_ids, attention_mask, model, tokenizer):
    """
    批量预测下联（使用 HuggingFace 原生 generate）
    """
    model.eval()
    with torch.no_grad():
        outputs = model.model.generate(   # ★★★ 使用底层 T5 generate
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=5,
            max_length=config.SEQ_LEN,
            early_stopping=True,
        )

    results = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        results.append(text)
    return results


def predict(user_input, model, tokenizer, device):
    """
    单句预测
    """
    prefix = "对联："  # 与训练一致
    text = prefix + user_input

    encoded = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=config.SEQ_LEN,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    batch_result = predict_batch(input_ids, attention_mask, model, tokenizer)
    return batch_result[0]


def run_prediction():
    # 1. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 加载 tokenizer & 模型
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRETERAINED_MODELS_DIR / 't5-chinese-couplet'
    )

    model = CoupletGenerateModel(freeze_encoder=False).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / "best_model.pt", map_location=device))
    model.eval()
    print("模型加载完成！")

    # 3. 循环预测
    print("请输入上联：(输入 q 或 quit 退出)")

    while True:
        user_input = input("> ")

        if user_input.lower() in ["q", "quit"]:
            print("退出程序")
            break

        if user_input.strip() == "":
            print("输入不能为空，请重新输入")
            continue

        result = predict(user_input, model, tokenizer, device)
        print("下联：", result)


if __name__ == "__main__":
    run_prediction()

import sys
from pathlib import Path

# 先把 src 目录加入 Python 搜索路径
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import torch
import config
from model import CoupletGenerateModel

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer
from fastapi.responses import FileResponse

# =====================================================
# 全局初始化：只加载一次模型 → 超快响应
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

tokenizer = AutoTokenizer.from_pretrained(
    config.PRETERAINED_MODELS_DIR / "t5-chinese-couplet",
    use_fast=False
)

model = CoupletGenerateModel(freeze_encoder=False).to(device)
model.load_state_dict(torch.load(config.MODELS_DIR / "best_model.pt", map_location=device))
model.eval()

# =====================================================
# 生成函数（后端核心，仅数十毫秒）
# =====================================================
def generate_couplet(text: str) -> str:
    print("DEBUG: generate 被调用了")
    prefix = "对联：" + text

    encoded = tokenizer(
        [prefix],
        padding="max_length",
        truncation=True,
        max_length=config.SEQ_LEN,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    print("DEBUG: 当前使用的 generate 参数：max_length=", config.SEQ_LEN)
    with torch.no_grad():
            output_ids = model.model.generate(   # ★★★ 关键修复点：使用原生 generate
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=5,
                max_length=config.SEQ_LEN,
                early_stopping=True,
            )
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("DEBUG: 生成结果：", result)
    return result

# =====================================================
# FastAPI 主应用
# =====================================================

app = FastAPI()

# 静态页面挂载
app.mount("/static", StaticFiles(directory="static"), name="static")

# 首页路由
@app.get("/")
async def index():
    return FileResponse("static/index.html")

# AJAX 调用接口
@app.get("/api/generate")
async def generate_api(text: str):
    result = generate_couplet(text)
    return JSONResponse({"couplet": result})

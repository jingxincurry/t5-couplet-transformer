# README.md

# Couplet-Transformer-T5

# 1.概述

**Couplet-Transformer-T5** 是一个基于 **T5（Text-to-Text Transfer Transformer）模型** 的中文对联生成系统，覆盖从数据处理、模型训练、评估到在线推理与 Web 部署的完整流程。

本项目实现了：

- 中文对联数据的预处理与加载
- 基于 HuggingFace Transformers 的 T5 微调训练
- 模型评估与批量预测
- 基于 **FastAPI + Uvicorn + Gradio** 的 Web 交互界面
- 支持 **GPU 加速训练与推理**，可直接部署上线

模型核心采用 `T5ForConditionalGeneration`，适用于中文生成任务。

**在线演示 Demo**

👉[https://coupletai.xyz/](https://coupletai.xyz/)（对联生成器）

# 2.项目结构
```bash
COUPLET-TRANSFORMER-T5
│
├── data/                 # 数据集（上联/下联）
├── logs/                 # TensorBoard 日志
├── models/               # 训练后的模型权重
├── pretrained/           # 预训练 T5 模型
│   └── t5-chinese-couplet/
├── src/
│   ├── [app.py](http://app.py/)            # Web 服务（FastAPI/Gradio）
│   ├── [config.py](http://config.py/)         # 全局配置
│   ├── [dataset.py](http://dataset.py/)        # 数据加载 & Tokenizer 封装
│   ├── [evaluate.py](http://evaluate.py/)       # 模型评估
│   ├── [model.py](http://model.py/)          # T5 模型定义
│   ├── [predict.py](http://predict.py/)        # 预测脚本（CLI 批量预测）
│   ├── [process.py](http://process.py/)        # 数据预处理
│   └── [train.py](http://train.py/)          # 主训练脚本
│
├── static/               # Web 前端静态资源
├── requirements.txt
└── [README.md](http://readme.md/)
```

# 3.环境配置

- **Python**：3.10（已验证）
- **CUDA**：12.1（用于 GPU 加速训练与推理，可选）
- **操作系统**：Linux / macOS / Windows（推荐 Linux 服务器环境）
- 已在 Windows/**Ubuntu 20.04 / 22.04 + CUDA 12.1** 环境下稳定运行。

```python
pip install -r requirements.txt
```

## 4. 数据集准备 Dataset

本项目使用公开的中文对联数据集：

- **couplet-dataset**
    
    [https://github.com/wb14123/couplet-dataset](https://github.com/wb14123/couplet-dataset)
    

下载后按照项目约定放置于 `data/`raw 目录下，并通过 `process.py` 进行预处理。

使用公开中文对联数据集[couplet-dataset](https://github.com/wb14123/couplet-dataset)

## 5.预训练模型

项目支持直接微调已公开的中文对联 T5 预训练模型：

[t5-chinese-couplet](https://huggingface.co/shibing624/t5-chinese-couplet)
下载完成后，模型文件将存放于 `pretrained/` 目录中。

通过 CLI 下载：

```bash
huggingface-cli download shibing624/t5-chinese-couplet \
    --local-dir ./pretrained/t5-chinese-couplet \
    --resume-download

```

## 6. 模型训练 Training

运行以下命令：

```bash
python src/train.py

```

训练日志保存在 `./logs/` 中，可使用 TensorBoard 查看。

```python
tensorboard --logdir=./logs
```

## 8. 命令行预测 CLI Prediction

```bash
python src/predict.py

```

## 9. Web 前端服务 Web UI

使用 Uvicorn 启动：

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

```

启动后浏览器访问：

```
http://localhost:8000
```

---

# 10.输出示例

---

以下是模型生成的部分示例结果：

| **上联** | **下联** |
| --- | --- |
| 书香醉我凌云梦 | 墨韵怡人揽月心 |
| 春回大地，对对黄莺鸣暖树 | 福满人间,声声喜鹊闹红梅 |
| 庙貌并吴头楚尾,纵远来粤北,亦难分一脉馨香 | 庙貌并吴头楚尾,纵远来粤北,亦难分一脉馨香 |
| 已彻骨深寒，倩影依稀，有我空庭长饮月 | 已随风渐老,伊人宛在,凭谁玉笛暗飞声 |
| 千秋月色君长看 | 一片冰心我自知 |
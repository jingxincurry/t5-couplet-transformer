# src/model.py
from torch import nn
import torch
from transformers import T5ForConditionalGeneration
import config


class CoupletGenerateModel(nn.Module):
    """
    自定义 T5 对联生成模块
    输入：上联 token_ids
    输出：下联 token_ids（训练时计算 loss）
    """

    def __init__(self, freeze_encoder=False):
        super().__init__()

        # 1. 加载预训练 T5（本地或在线）
        self.model = T5ForConditionalGeneration.from_pretrained(
            str(config.PRETERAINED_MODELS_DIR / 't5-chinese-couplet')
        )

        # 2. 选择性冻结 encoder（可选）
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, target_ids=None):
        """
        input_ids: 上联 token_ids [batch, seq_len]
        target_ids: 下联 token_ids [batch, seq_len]
        return: loss, logits 
        """
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()   # 0 是 pad_token_id
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids
        )

        return outputs.loss, outputs.logits

    # @torch.no_grad()
    # def generate(self, input_ids, attention_mask=None):
    #     if attention_mask is None:
    #         attention_mask = (input_ids != 0).long()

    #     return self.model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,

    #         # 推荐参数 — 不会产生“、”
    #         num_beams=5,
    #         max_length=config.SEQ_LEN,
    #         min_length=5,
    #         no_repeat_ngram_size=3,
    #         length_penalty=1.0,
    #         early_stopping=True,
    #     )
       


if __name__ == '__main__':
    model = CoupletGenerateModel()
    print(model)

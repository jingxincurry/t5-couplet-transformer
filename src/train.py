import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import CoupletGenerateModel
from torch.utils.tensorboard import SummaryWriter
import time

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["up"].to(device)
        target_ids = batch["down"].to(device)
        attention_mask = (input_ids != 0).long()
        optimizer.zero_grad()

        # forward
        loss, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ids=target_ids
        )

        loss.backward()

        # 梯度裁剪，防止 exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["up"].to(device)
            target_ids = batch["down"].to(device)
            attention_mask = (input_ids != 0).long()
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_ids=target_ids
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 载入模型
    model = CoupletGenerateModel(freeze_encoder=False).to(device)
    
    # 2. 加载数据
    train_loader = get_dataloader(train=True)
    val_loader = get_dataloader(train=False)

    # 3. 优化器
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
   
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d-%H-%M-%S"))
    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n===== Epoch {epoch} / {config.EPOCHS} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

        # 记录到 TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        # 5. 保存当前 Epoch 模型
        current_ckpt = config.MODELS_DIR / f"epoch_{epoch}.pt"
        current_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), current_ckpt)
        print(f"模型已保存到: {current_ckpt}")

        # 6. 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = config.MODELS_DIR / "best_model.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
            print(f"最优模型更新，已保存到: {best_path}")
        else:
            print("最优模型未更新")

    writer.close()


if __name__ == "__main__":
    train()

# 1.定义Dataset
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import config


# 2.获取DataLoader的方法
def get_dataloader(train=True):
    data_path = str(config.PROCESSED_DIR / ('train' if train else 'test'))
    dataset = load_from_disk(data_path)

    # 设置为 PyTorch 格式
    dataset.set_format(type='torch', columns=['up', 'down'])

    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=train)


if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    print(f'train batch个数：{len(train_dataloader)}')

    test_dataloader = get_dataloader(train=False)
    print(f'test batch个数：{len(test_dataloader)}')

    # 检查一个 batch
    for batch in train_dataloader:
        print("up.shape:", batch['up'].shape)        # [batch, seq_len]
        print("down.shape:", batch['down'].shape)    # [batch, seq_len]
        break

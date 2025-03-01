import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from GPT import GPT, UTRDataset
import pandas as pd

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]  # 提取所有样本的input_ids
    labels = [item["labels"] for item in batch]  # 提取所有样本的labels
    padding_masks = [item["padding_mask"] for item in batch]  # 提取所有样本的padding_mask

    # 使用pad_sequence对序列进行填充，确保批次内所有序列长度一致
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)  #
    padding_masks_padded = torch.nn.utils.rnn.pad_sequence(padding_masks, batch_first=True, padding_value=1)

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "padding_mask": padding_masks_padded
    }


def train_GPT(sequences, word_vectors_path, model_save_path):

    dataset = UTRDataset(sequences, word_vectors_path, 200, 3)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(dataset.word_vectors.key_to_index) + 1
    max_len = 200 # 最大序列长度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT(vocab_size=vocab_size, max_len=max_len).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充位置
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    best_loss = float('inf')

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)  # 获取输入序列
            labels = batch["labels"].to(device)  # 获取标签
            padding_mask = batch["padding_mask"].to(device)  # 获取填充掩码
            # 输入序列和目标序列
            inputs = input_ids[:, :-1]  # 输入序列（去掉最后一个词）
            targets = labels[:, 1:]  # 目标序列（右移一位）

            optimizer.zero_grad()
            logits = model(inputs, padding_mask=padding_mask[:, :-1])  # 前向传播

            # 计算损失并反向传播
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            avg_loss = running_loss / len(data_loader)
            print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}')

            scheduler.step(avg_loss)
            print(f'Learning rate after epoch {epoch + 1}: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), model_save_path)
                print(f'New best model saved with loss: {avg_loss:.4f}')

        print(f'Training complete. Best model saved at: {model_save_path}')


file_path = "five_prime_utr_test.csv"
UTR_column = "Sequence"
df = pd.read_csv(file_path)
utr_seqs = df[UTR_column]

train_GPT(utr_seqs, 'RNA_word_vec.model', "best_GPT.pth")


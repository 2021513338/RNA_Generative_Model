import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from GPT import GPT, UTRDataset
from sklearn.metrics import accuracy_score
import math

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    padding_masks = [item["padding_mask"] for item in batch]

    # 填充到批次内最大长度
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    padding_masks_padded = pad_sequence(padding_masks, batch_first=True, padding_value=1)

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "padding_mask": padding_masks_padded
    }


def train_GPT(sequences, word_vectors_path, model_save_path, num_epochs):

    k_folds = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 K 折交叉验证
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
        print(f"Training fold {fold + 1}/{k_folds}")
        # 分割数据集
        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        # 创建数据集和DataLoader
        train_dataset = UTRDataset(train_seqs, word_vectors_path, 200, 3)
        val_dataset = UTRDataset(val_seqs, word_vectors_path, 200, 3)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

        vocab_size = len(train_dataset.word_vectors.key_to_index)
        model = GPT(vocab_size=vocab_size).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略填充位置
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        best_loss = float('inf')


        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
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

                avg_loss = running_loss / len(train_loader)
                print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}')

                scheduler.step(avg_loss)
                print(f'Learning rate after epoch {epoch + 1}: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), model_save_path)
                    print(f'New best model saved with loss: {avg_loss:.4f}')

            print(f'Training complete. Best model saved at: {model_save_path}')

        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        val_loss = 0.0
        total_tokens = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                padding_mask = batch["padding_mask"].to(device)


                inputs = input_ids[:, :-1]
                targets = labels[:, 1:]
                mask = (targets != 0)  # 过滤填充位置的掩码

                # 前向传播
                logits = model(inputs, padding_mask=padding_mask[:, :-1])
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss += loss.item() * targets.numel()  # 累加总损失（考虑批量）
                total_tokens += mask.sum().item()

                # 获取预测结果并过滤填充位置
                preds = logits.argmax(dim=-1)  # (batch_size, seq_len)
                preds_np = preds.cpu().numpy().flatten()
                targets_np = targets.cpu().numpy().flatten()
                mask_np = mask.cpu().numpy().flatten()

                # 仅保留非填充位置的数据
                valid_preds = preds_np[mask_np]
                valid_targets = targets_np[mask_np]

                all_preds.extend(valid_preds.tolist())
                all_targets.extend(valid_targets.tolist())


        val_accuracy = accuracy_score(all_targets, all_preds)
        avg_val_loss = val_loss / total_tokens
        perplexity = math.exp(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Perplexity: {perplexity:.4f}")



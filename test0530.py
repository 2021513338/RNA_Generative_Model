import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from GRU import GRUModel, GRUDataset
from sklearn.metrics import roc_auc_score
import joblib

def collate_classification(batch):
    word_embeds = [item['word_embeds'] for item in batch]
    word_embeds_padded = pad_sequence(
        word_embeds, batch_first=True, padding_value=0
    )

    segment_ids = [item['segment_ids'] for item in batch]
    segment_ids_padded = pad_sequence(
        segment_ids, batch_first=True, padding_value=0
    )

    labels = torch.stack([item['label'] for item in batch])

    return {
        'word_embeds': word_embeds_padded,
        'segment_ids': segment_ids_padded,
        'label': labels,
    }

def train_gru_classification(CDS_sequences, Five_UTR_sequences, Three_UTR_sequences, labels, k, word_vectors_path, epochs,
                         num_classes=2, resume_from_checkpoint=None):
    batch_size = 16
    model_path = "result/best_gru_cls.pth"  # 修改模型保存名称
    result_save_path = f"result/GRU_cls_ep{epochs}.txt"
    result_file = open(result_save_path, "a" if resume_from_checkpoint else "w")
    device = 'cuda'  # 强制使用CPU

    # 确保标签为长整型
    labels = torch.LongTensor(labels.values if isinstance(labels, pd.Series) else labels)

    # 数据集分割（新增类别分层分割）
    if resume_from_checkpoint is None:
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)),
            test_size=0.1,
            stratify=labels.numpy(),
            random_state=42
        )
        joblib.dump(train_idx, "result/train_idx.joblib")
        joblib.dump(val_idx, "result/val_idx.joblib")
    else:
        train_idx = joblib.load("result/train_idx.joblib")
        val_idx = joblib.load("result/val_idx.joblib")

    # 创建GRU数据集
    train_dataset = GRUDataset(
        [CDS_sequences[i] for i in train_idx],
        [Five_UTR_sequences[i] for i in train_idx],
        [Three_UTR_sequences[i] for i in train_idx],
        k, word_vectors_path,
        task_type="classification",
        labels=[labels[i] for i in train_idx]
    )

    val_dataset = GRUDataset(
        [CDS_sequences[i] for i in val_idx],
        [Five_UTR_sequences[i] for i in val_idx],
        [Three_UTR_sequences[i] for i in val_idx],
        k, word_vectors_path,
        task_type="classification",
        labels=[labels[i] for i in val_idx]
    )

    # 初始化GRU模型
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        print(f"使用 {num_gpus} 张GPU进行训练")
        model = nn.DataParallel(GRUModel(
            num_classes=num_classes
        )).to(device)
    else:
        print("使用单GPU训练")
        model = GRUModel(
            num_classes=num_classes
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_loss = float('inf')

    # 检查点恢复逻辑
    if resume_from_checkpoint:
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', checkpoint['loss'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        start_epoch = 0

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_classification)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_classification)

    for epoch in range(start_epoch, epochs):
        time_start = time.time()
        model.train()
        total_loss = 0

        for batch in train_loader:
            word_embeds = batch['word_embeds'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            targets = batch['label'].to(device)

            optimizer.zero_grad()
            # GRU模型调用（注意参数顺序）
            outputs = model(word_embeds, segment_ids, task_type='classification')
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        time_end = time.time()
        epoch_time = time_end - time_start
        scheduler.step(avg_loss)
        result_file.write(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Time: {epoch_time:.2f}s | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}\n"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f'New best model saved with loss: {avg_loss:.4f}')

        # 每10个epoch验证一次（减少验证频率以加快训练）
        if (epoch + 1) % 20 == 0:
            model.eval()
            all_probs = []
            all_targets = []
            with torch.no_grad():
                for batch in val_loader:
                    word_embeds = batch['word_embeds'].to(device)
                    segment_ids = batch['segment_ids'].to(device)
                    targets = batch['label'].cpu().numpy()

                    outputs = model(word_embeds, segment_ids, task_type='classification')
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()

                    all_probs.extend(probs[:, 1] if num_classes == 2 else probs)
                    all_targets.extend(targets)

            # 计算AUC
            try:
                auc_score = roc_auc_score(all_targets, all_probs) if num_classes == 2 else \
                    roc_auc_score(all_targets, all_probs, multi_class='ovo')
            except ValueError:
                auc_score = 0.5

            # 保存检查点
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'auc_score': auc_score
            }
            torch.save(checkpoint, f"result/checkpoint_gru_epoch_{epoch + 1}.pth")
            result_file.write(f"Checkpoint saved: checkpoint_gru_epoch_{epoch + 1}.pth\n")
            result_file.write(f"The AUC at epoch {epoch + 1} is {auc_score:.4f}\n")
            result_file.flush()

    result_file.close()

# 调用示例
cell_type = 'DC'
file_path = f"/Users/xcw/Desktop/BERT_CLA_TEST/Homo_{cell_type}_2.csv"
cds_column = "cds_seq"
utr_5_colum = "utr5_seq"
utr_3_colum = "utr3_seq"
label_colum = f'Label_{cell_type}_Intensity_per_fpkm'
df = pd.read_csv(file_path)
CDS_sequences = df[cds_column]
Five_UTR_sequences = df[utr_5_colum]
Three_UTR_sequences = df[utr_3_colum]
labels = df[label_colum]
train_gru_classification(CDS_sequences, Five_UTR_sequences, Three_UTR_sequences,
                     labels, k=3, word_vectors_path='result/word_vectors.model',
                     epochs=200, num_classes=2)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Transformer import AminoAcidDataset, TransformerModel
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


def collate_fn(batch):
    seq_lengths = [item['seq_length'] for item in batch]
    max_seq_length = max(seq_lengths)

    padded_seqs = []
    padded_labels = []
    for item in batch:
        seq = item['seq']  # seq 的形状为 (seq_length, embedding_dim)
        label = item['label']  # label 的形状为 (seq_length,)

        if len(seq) < max_seq_length:
            pad_length = max_seq_length - len(seq)
            # 对 seq 进行填充
            padded_seq = torch.cat([seq, torch.full((pad_length, seq.size(1)), -1.0)], dim=0)
            # 对 label 进行填充，保持为一维
            padded_label = torch.cat([label, torch.full((pad_length,), 62, dtype=torch.long)], dim=0)
        else:
            padded_seq = seq[:max_seq_length]
            padded_label = label[:max_seq_length]

        padded_seqs.append(padded_seq)
        padded_labels.append(padded_label)

    # 堆叠 padded_seqs，形状为 (max_seq_length, batch_size, embedding_dim)
    padded_seqs = torch.stack(padded_seqs, dim=1)

    # 堆叠 padded_labels，形状为 (max_seq_length, batch_size)
    padded_labels = torch.stack(padded_labels, dim=1)

    # 生成 src_padding_mask
    src_padding_mask = (padded_seqs == -1.0).all(dim=2).transpose(0, 1)
    print(src_padding_mask.dtype)

    return {
        'seq': padded_seqs,
        'label': padded_labels,
        'src_padding_mask': src_padding_mask,
        'seq_lengths': torch.tensor(seq_lengths, dtype=torch.long)
    }

def train_Transformer(amino_acid_seqs, labels, nhead, num_encoder_layers, dim_feedforward, num_epochs):

    # 模型参数
    k_folds = 5
    tgt_vocab_size = 63
    d_model = 128  # 模型维度，多头注意力的头数，需要能被d_model整除
    num_decoder_layers = num_encoder_layers  # 解码器层数
    max_seq_length = 1000  # 最大序列长度
    trans_dropout = 0.1  # Transformer 的 dropout
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "best_Transformer.pth"
    word_vectors_path = "protein_word_vec.model"
    result_save_path = f"result/TF_nh_{nhead}_el_{num_encoder_layers}_ff_{dim_feedforward}_ep_{num_epochs}.txt"

    # 初始化 K 折交叉验证
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_auc_scores = []

    result_file = open(result_save_path, "w")
    result_file.write("Fold\tACC\tAUC\n")  # 写入表头

    for fold, (train_idx, val_idx) in enumerate(kf.split(amino_acid_seqs)):
        print(f"Training fold {fold + 1}/{k_folds}")

        # 划分训练集和验证集
        train_seqs = [amino_acid_seqs[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_seqs = [amino_acid_seqs[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # 创建训练集和验证集的数据加载器
        train_dataset = AminoAcidDataset(train_seqs, train_labels, word_vectors_path)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataset = AminoAcidDataset(val_seqs, val_labels, word_vectors_path)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

        # 初始化模型、优化器和损失函数
        model = TransformerModel(tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                 dim_feedforward, max_seq_length, trans_dropout).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=62)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        best_loss = float('inf')

        # 训练模型
        for epoch in range(num_epochs):
            model.train()  # 确保模型在训练模式下
            running_loss = 0.0
            for batch in train_loader:
                src = batch['seq'].to(device)  # 已经转置为 (seq_len, batch_size, embedding_dim)
                tgt = batch['label'].to(device)  # 已经转置为 (seq_len, batch_size)
                src_padding_mask = batch['src_padding_mask'].to(device)  # 从 collate_fn 中获取填充掩码

                tgt_input = tgt[:-1, :]  # 解码器输入
                tgt_label = tgt[1:, :]  # 解码器目标

                optimizer.zero_grad()
                output = model(src, tgt_input, src_padding_mask=src_padding_mask)

                loss = criterion(output.view(-1, tgt_vocab_size), tgt_label.view(-1))
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

        # 验证模型
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                seq = batch['seq'].to(device)
                label = batch['label'].to(device)
                src_padding_mask = batch['src_padding_mask'].to(device)
                predictions = model.predict(seq, src_padding_mask)
                all_preds.extend(predictions)
                all_labels.extend(label.numpy())

        # 将预测结果和标签展平
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # 确保标签和预测结果是二维数组
        unique_labels = np.unique(all_labels)
        all_labels_binarized = label_binarize(all_labels, classes=unique_labels)
        all_preds_binarized = label_binarize(all_preds, classes=unique_labels)

        # 计算 ACC 和 AUC
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels_binarized, all_preds_binarized, average='macro')
        fold_accuracies.append(acc)
        fold_auc_scores.append(auc)
        print(f"Fold {fold + 1} - ACC: {acc:.4f}, AUC: {auc:.4f}")
        result_file.write(f"{fold + 1}\t{acc:.4f}\t{auc:.4f}\n")

    # 计算平均 ACC 和 AUC
    avg_acc = np.mean(fold_accuracies)
    avg_auc = np.mean(fold_auc_scores)
    print(f"Average ACC: {avg_acc:.4f}, Average AUC: {avg_auc:.4f}")
    print(f'Training complete. Best model saved at: {model_save_path}')
    result_file.write(f"Average\t{avg_acc:.4f}\t{avg_auc:.4f}\n")

    result_file.close()

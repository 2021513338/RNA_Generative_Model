import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from RNN_CRF import AminoAcidDataset, BiGRU_CRF
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np



def collate_fn(batch):
    seqs = [item['seq'] for item in batch]
    labels = [item['label'] for item in batch]
    seq_lengths = [len(seq) for seq in seqs]

    # 动态填充
    seqs = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return {
        'seq': seqs,
        'label': labels,
        'seq_lengths': torch.tensor(seq_lengths, dtype=torch.long)
    }


def train_RNN_CRF(amino_acid_seqs, labels, word_vectors_path, model_save_path, k_folds=5):
    # 数据加载器
    dataset = AminoAcidDataset(amino_acid_seqs, labels, word_vectors_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # 模型参数
    tag_size = 62
    input_dim = 100  # 词向量维度
    hidden_dim = 128  # RNN隐藏层维度
    output_dim = tag_size

    # 初始化 K 折交叉验证
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(amino_acid_seqs)):
        print(f"Training fold {fold + 1}/{k_folds}")

        # 划分训练集和验证集
        train_seqs = [amino_acid_seqs[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_seqs = [amino_acid_seqs[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # 创建训练集和验证集的数据加载器
        train_dataset = AminoAcidDataset(train_seqs, train_labels, word_vectors_path)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        val_dataset = AminoAcidDataset(val_seqs, val_labels, word_vectors_path)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        # 初始化模型、优化器和学习率调度器
        model = BiGRU_CRF(input_dim, hidden_dim, tag_size, output_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        best_loss = float('inf')

        # 训练模型
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                seq = batch['seq']
                label = batch['label']
                seq_lengths = batch['seq_lengths']

                optimizer.zero_grad()
                output = model(seq, seq_lengths)
                loss = model.loss(output, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f'Epoch {fold + 1}, {epoch + 1}, Avg Loss: {avg_loss:.4f}')
            scheduler.step(avg_loss)

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
                seq = batch['seq']
                label = batch['label']
                seq_lengths = batch['seq_lengths']
                predictions = model.predict(seq, seq_lengths)
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

    # 计算平均 ACC 和 AUC
    avg_acc = np.mean(fold_accuracies)
    avg_auc = np.mean(fold_auc_scores)
    print(f"Average ACC: {avg_acc:.4f}, Average AUC: {avg_auc:.4f}")
    print(f'Training complete. Best model saved at: {model_save_path}')



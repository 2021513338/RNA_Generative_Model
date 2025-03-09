import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from BERT import BERT, BERTDataset
from sklearn.metrics import r2_score

def collate_mlm(batch):
    # 对 word_embeds, segment_ids, seq_labels 进行填充
    word_embeds = [item['word_embeds'] for item in batch]
    word_embeds_padded = pad_sequence(
        word_embeds, batch_first=True, padding_value=0
    )

    segment_ids = [item['segment_ids'] for item in batch]
    segment_ids_padded = pad_sequence(
        segment_ids, batch_first=True, padding_value=0
    )

    seq_labels = [item['seq_labels'] for item in batch]
    seq_labels_padded = pad_sequence(
        seq_labels, batch_first=True, padding_value=-1
    )

    # 处理 masked_positions（原有逻辑，填充为 -1）
    masked_positions = [item['masked_positions'] for item in batch]
    masked_positions_padded = pad_sequence(
        masked_positions, batch_first=True, padding_value=-1
    )

    masked_labels = torch.cat([item['masked_labels'] for item in batch])

    return {
        'word_embeds': word_embeds_padded,
        'segment_ids': segment_ids_padded,
        'seq_labels': seq_labels_padded,
        'masked_positions': masked_positions_padded,
        'masked_labels': masked_labels,
    }

def collate_regression(batch):
    # 对 word_embeds, segment_ids 进行填充
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

def train_bert_mlm(CDS_sequences, Five_UTR_sequences, Three_UTR_sequences, k, word_vectors_path, num_epochs):

    model_save_path = "best_bert_mlm.pth"

    batch_size = 8
    # 创建 DataLoader
    dataset = BERTDataset(CDS_sequences, Five_UTR_sequences, Three_UTR_sequences, k, word_vectors_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_mlm)
    # 将模型移动到设备（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT(dataset.word_vectors).to(device)
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 用于 MLM 任务的交叉熵损失
    best_loss = float('inf')
    # 训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        total_loss = 0
        total_correct = 0
        total_masked = 0

        for batch in dataloader:
            # 将数据移动到设备
            word_embeds = batch['word_embeds'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            masked_positions = batch['masked_positions']
            masked_labels = batch['masked_labels'].to(device)

            # 前向传播
            logits = model(word_embeds, segment_ids, masked_positions, task_type="mlm")

            # 计算损失
            loss = criterion(logits.view(-1, logits.size(-1)), masked_labels.view(-1))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和正确率
            total_loss += loss.item()

            # 计算掩码重建的正确率
            preds = torch.argmax(logits, dim=-1)  # 获取预测的 token ID
            correct = (preds == masked_labels).sum().item()
            total_correct += correct
            total_masked += masked_labels.numel()

        # 计算平均损失和正确率
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_masked
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Masked Accuracy: {accuracy:.4f}")
        scheduler.step(avg_loss)
        print(f'Learning rate after epoch {epoch + 1}: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'New best model saved with loss: {avg_loss:.4f}')


def train_regression( CDS_sequences, Five_UTR_sequences, Three_UTR_sequences, labels, k, word_vectors_path, epochs):
    # 初始化 K 折交叉验证
    batch_size = 8
    model_save_path = "best_bert_mlm.pth"
    reg_model_path = "best_bert_reg.pth"
    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    result_save_path = f"result/BERT_ep{epochs}.txt"
    result_file = open(result_save_path, "w")
    result_file.write("Fold\tR^2\n")  # 写入表头

    for fold, (train_idx, val_idx) in enumerate(kf.split(CDS_sequences)):
        print(f"Training fold {fold + 1}/{k_folds}")

        # 创建训练集和验证集的数据加载器
        train_CDS_seqs = [CDS_sequences[i] for i in train_idx]
        train_five_seqs = [Five_UTR_sequences[i] for i in train_idx]
        train_three_seqs = [Three_UTR_sequences[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]

        val_CDS_seqs = [CDS_sequences[i] for i in val_idx]
        val_five_seqs = [Five_UTR_sequences[i] for i in val_idx]
        val_three_seqs = [Three_UTR_sequences[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_dataset = BERTDataset(train_CDS_seqs, train_five_seqs, train_three_seqs, k, word_vectors_path, task_type="regression", labels=train_labels)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_regression)
        val_dataset = BERTDataset(val_CDS_seqs, val_five_seqs, val_three_seqs, k, word_vectors_path, task_type="regression", labels=val_labels)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_regression)


        criterion = nn.MSELoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BERT(train_dataset.word_vectors).to(device)
        model.load_state_dict(torch.load(model_save_path))
        for param in model.parameters():
            param.requires_grad = False

        for param in model.regression_head.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),  # 只选择需要梯度的参数
            lr=1e-5
        )
        model.train()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        best_loss = float('inf')

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                word_embeds = batch['word_embeds'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                train_batch_labels = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(word_embeds, segment_ids, task_type='regression')
                loss = criterion(outputs.squeeze(), train_batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Regression Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)
            print(f'Learning rate after epoch {epoch + 1}: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), reg_model_path)
                print(f'New best model saved with loss: {avg_loss:.4f}')

        # 在每个 fold 结束后，使用最佳模型计算验证集的 R²
        model.load_state_dict(torch.load(reg_model_path))
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                word_embeds = batch['word_embeds'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                val_batch_labels = batch['label'].to(device)

                outputs = model(word_embeds, segment_ids, task_type='regression')
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(val_batch_labels.cpu().numpy().flatten())

        r2 = r2_score(all_labels, all_preds)
        print(f"Fold {fold + 1} Validation R^2: {r2:.4f}")
        result_file.write(f"{fold + 1}\t{r2:.4f}\n")

    result_file.close()
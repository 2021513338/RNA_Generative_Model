import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from BERT import BERT, BERTDataset

def train_bert(CDS_sequences, Five_UTR_sequences, Three_UTR_sequences, k, word_vectors_path, batch_size=32, num_epochs=10, learning_rate=1e-4, device='cuda'):

    # 创建 DataLoader
    dataset = BERTDataset(CDS_sequences, Five_UTR_sequences, Three_UTR_sequences, k, word_vectors_path)  # 创建 BERTDataset 对象
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BERT(dataset.word_vectors)
    # 将模型移动到设备（GPU 或 CPU）
    model = model.to(device)
    # 定义优化器和损失函数
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()  # 用于 MLM 任务的交叉熵损失

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
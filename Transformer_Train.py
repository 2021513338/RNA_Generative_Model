from Transformer import AminoAcidDataset, TransformerModel
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
    src_padding_mask = (padded_seqs == -1).all(dim=2).transpose(0, 1)

    return {
        'seq': padded_seqs,
        'label': padded_labels,
        'src_padding_mask': src_padding_mask
    }

def train_Transformer(amino_acid_seqs, labels, word_vectors_path, model_save_path="best_Transformer.pth"):
    # 数据加载器
    dataset = AminoAcidDataset(amino_acid_seqs, labels, word_vectors_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    tgt_vocab_size = 63
    d_model = 128  # 模型维度
    nhead = 8  # 多头注意力的头数
    num_encoder_layers = 6  # 编码器层数
    num_decoder_layers = 6  # 解码器层数
    dim_feedforward = 2048  # 前馈网络维度
    max_seq_length = 1000  # 最大序列长度
    trans_dropout = 0.1  # Transformer 的 dropout

    # 初始化模型、优化器和损失函数
    model = TransformerModel(tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                             dim_feedforward, max_seq_length, trans_dropout)
    criterion = nn.CrossEntropyLoss(ignore_index=62)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    best_loss = float('inf')

    # 训练模型
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()  # 确保模型在训练模式下
        running_loss = 0.0
        for batch in data_loader:
            src = batch['seq']  # 已经转置为 (seq_len, batch_size, embedding_dim)
            tgt = batch['label']  # 已经转置为 (seq_len, batch_size)
            src_padding_mask = batch['src_padding_mask']  # 从 collate_fn 中获取填充掩码

            tgt_input = tgt[:-1, :]  # 解码器输入
            tgt_label = tgt[1:, :]  # 解码器目标

            optimizer.zero_grad()
            output = model(src, tgt_input, src_padding_mask=src_padding_mask)

            loss = criterion(output.view(-1, tgt_vocab_size), tgt_label.view(-1))
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




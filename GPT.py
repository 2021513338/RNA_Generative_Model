import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from gensim.models import Word2Vec

# 定义GPT模型
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, max_len=100):
        super(GPT, self).__init__()
        self.positional_encoding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.positional_encoding.weight, mean=0, std=0.02)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
        )

        self.transformer = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x, padding_mask=None):
        # x: (batch_size, seq_len)
        batch_size, seq_len, _ = x.size()
        print(f"Input shape: {x.shape}")  # 打印输入的形状
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).to(x.device)
        x = x + self.positional_encoding(positions)  # (batch_size, seq_len, d_model)
        print(f"Positional encoding shape: {x.shape}")
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        print(f"Permuted shape for Transformer: {x.shape}")

        # 生成自回归掩码和填充掩码
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        if padding_mask is not None:
            src_key_padding_mask = padding_mask  # (batch_size, seq_len), True表示填充位置
        else:
            src_key_padding_mask = None

        # Transformer前向传播
        x = self.transformer(
            x, x,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=src_key_padding_mask
        )
        print(f"Transformer decoder output shape: {x.shape}")
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        print(f"Permuted back shape: {x.shape}")
        x = self.fc_out(x)
        print(f"Logits shape: {x.shape}")
        return x


# 自定义数据集
class UTRDataset(Dataset):
    def __init__(self, sequences, word_vectors_path, max_len, k=3):
        self.k = k
        self.sequences = sequences
        self.word_vectors_path = word_vectors_path
        self.max_len = max_len
        self.word_vectors = self.load_word_vectors()

    def load_word_vectors(self):
        if os.path.exists(self.word_vectors_path):
            print(f"Loading pre-trained word vectors from {self.word_vectors_path}...")
            return Word2Vec.load(self.word_vectors_path).wv
        else:
            print(f"No pre-trained word vectors found. Training new model...")
            k = self.k
            k_mer_sequences = [[seq[i:i + k] for i in range(len(seq) - k + 1)] for seq in self.sequences]
            w2v_model = Word2Vec(sentences=k_mer_sequences, vector_size=128, window=5, min_count=1, workers=4)
            w2v_model.save(self.word_vectors_path)
            print(f"Model saved to {self.word_vectors_path}.")
            return w2v_model.wv

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        k = self.k  # 设置k值，可以根据需要调整
        k_mers = [seq[i:i + k] for i in range(len(seq) - k + 1)]  # 将RNA序列分割为k-mers
        seq_vectors = [self.word_vectors[word] for word in k_mers]  # 获取词向量
        seq_labels = [self.word_vectors.key_to_index[word] for word in k_mers]  # 获取k-mer对应的整数标签
        pad_mask = [0] * len(seq_vectors)  # 初始化填充掩码

        if len(seq_vectors) < self.max_len:
            pad_length = self.max_len - len(seq_vectors)
            seq_vectors += [torch.zeros(self.word_vectors.vector_size)] * pad_length  # 使用零向量填充
            seq_labels += [0] * pad_length  # 填充位置标签为0
            pad_mask += [1] * pad_length  # 填充位置标记为1（True）
        else:
            seq_vectors = seq_vectors[:self.max_len]
            seq_labels = seq_labels[:self.max_len]
            pad_mask = pad_mask[:self.max_len]

        return {
            "input_ids": torch.tensor(seq_vectors, dtype=torch.float),  # 词向量张量
            "labels": torch.tensor(seq_labels, dtype=torch.long),  # k-mer对应的整数标签
            "padding_mask": torch.tensor(pad_mask, dtype=torch.bool)  # True表示填充位置
        }






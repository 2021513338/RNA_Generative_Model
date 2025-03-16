import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from gensim.models import Word2Vec

# 定义GPT模型
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, max_len=200):
        super(GPT, self).__init__()
        self.positional_encoding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.positional_encoding.weight, mean=0, std=0.02)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            batch_first=True
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
        print(f"Permuted shape for Transformer: {x.shape}")

        # 生成自回归掩码和填充掩码
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(
            x, x,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask
        )
        print(f"Transformer decoder output shape: {x.shape}")

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
            k_mer_sequences = []
            for seq in self.sequences:
                # 分割原始序列的k-mers
                k_mers = [seq[i:i + self.k] for i in range(len(seq) - self.k + 1)]
                k_mers.append('<eos>')  # 追加<eos>
                k_mer_sequences.append(k_mers)
            w2v_model = Word2Vec(sentences=k_mer_sequences, vector_size=128, window=5, min_count=1, workers=4)
            w2v_model.save(self.word_vectors_path)
            print(f"Model saved to {self.word_vectors_path}.")
            return w2v_model.wv

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        k = self.k
        k_mers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        k_mers.append('<eos>')
        seq_vectors = [self.word_vectors[word] for word in k_mers]
        seq_labels = [self.word_vectors.key_to_index[word] for word in k_mers]
        pad_mask = [0] * len(seq_vectors)  # 实际位置标记为0（非填充）

        return {
            "input_ids": torch.tensor(seq_vectors, dtype=torch.float),
            "labels": torch.tensor(seq_labels, dtype=torch.long),
            "padding_mask": torch.tensor(pad_mask, dtype=torch.bool)
        }






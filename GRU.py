import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import numpy as np


class GRUModel(nn.Module):
    def __init__(self, word_vec_size=128, hidden_size=768, num_layers=3, bidirectional=True, num_classes=2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        # 分段嵌入
        self.segment_embed = nn.Embedding(3, 16)

        # GRU编码器
        self.gru = nn.GRU(
            input_size=word_vec_size + 16,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # 分类输出层
        factor = 2 if bidirectional else 1
        self.attention = nn.Linear(hidden_size * factor, 1)
        if num_classes > 0:
            self.classifier = nn.Linear(hidden_size * factor, num_classes)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化所有权重"""
        # 初始化 GRU 权重
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                init.orthogonal_(param)
            elif 'bias' in name:
                init.zeros_(param)

        # 初始化分段嵌入 (新增)
        init.xavier_uniform_(self.segment_embed.weight)

        # 初始化注意力层 (新增)
        init.xavier_uniform_(self.attention.weight)
        if self.attention.bias is not None:
            init.zeros_(self.attention.bias)

        # 初始化分类器
        if hasattr(self, 'classifier') and isinstance(self.classifier, nn.Module):
            if isinstance(self.classifier, nn.Linear):
                init.xavier_uniform_(self.classifier.weight)
                if self.classifier.bias is not None:
                    init.zeros_(self.classifier.bias)

    def forward(self, word_embeds, segment_ids, task_type="classification"):
        # 分段嵌入
        seg_emb = self.segment_embed(segment_ids)
        combined = torch.cat([word_embeds, seg_emb], dim=-1)
        # GRU编码
        output, _ = self.gru(combined)
        # 注意力池化
        attn_weights = torch.softmax(self.attention(output), dim=1)
        context = torch.sum(attn_weights * output, dim=1)
        if task_type == "classification":
            if self.num_classes <= 0:
                raise ValueError("Model not configured for classification")
            # (batch_size, hidden_size)
            return self.classifier(context)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")


class GRUDataset(Dataset):
    def __init__(self, CDS_sequences, Five_UTR_sequences, Three_UTR_sequences, k,
                 word_vectors_path, task_type="classification", labels=None):

        self.k = k
        self.CDS_sequences = CDS_sequences
        self.Five_UTR_sequences = Five_UTR_sequences
        self.Three_UTR_sequences = Three_UTR_sequences
        self.sequences = [
            five + cds + three
            for five, cds, three in zip(Five_UTR_sequences, CDS_sequences, Three_UTR_sequences)
        ]
        self.word_vectors_path = word_vectors_path
        self.task_type = task_type
        self.labels = labels
        self.word_vectors = self.load_word_vectors()

        if self.task_type == "classification" and self.labels is None:
            raise ValueError("Labels must be provided for classification task")

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
        k = self.k
        # 将文本转换为词向量序列
        k_mers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        seq_vectors = np.array([self.word_vectors[word] for word in k_mers], dtype=np.float32)
        seq_vectors = torch.tensor(seq_vectors, dtype=torch.float32)
        # 生成分段ID
        five_len = len(self.Five_UTR_sequences[idx])
        cds_len = len(self.CDS_sequences[idx])
        three_len = len(self.Three_UTR_sequences[idx])

        five_kmers = five_len - 1
        cds_kmers = cds_len
        three_kmers = three_len - 1

        segment_ids = []
        segment_ids += [0] * five_kmers  # 5' UTR
        segment_ids += [1] * cds_kmers  # CDS
        segment_ids += [2] * three_kmers  # 3' UTR

        if self.task_type == "classification":
            return {
                "word_embeds": seq_vectors,
                "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
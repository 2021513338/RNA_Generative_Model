import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import numpy as np


class BERT(nn.Module):
    def __init__(self, word2vec_model, word_vec_size=128, hidden_size=768, num_layers=12, num_heads=8,
                 max_seq_len=16384, num_segments=4, num_classes=2):
        super(BERT, self).__init__()
        # CLS头嵌入
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, word_vec_size))
        nn.init.normal_(self.cls_embedding, mean=0, std=0.02)
        # 位置嵌入
        self.position_embeddings = nn.Embedding(max_seq_len, word_vec_size)
        # 分段嵌入（Segment Embeddings）
        self.segment_embeddings = nn.Embedding(num_segments, word_vec_size)
        # Layer Normalization 和 Dropout
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(word_vec_size)
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=word_vec_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # MLM 输出层
        self.vocab_size = word2vec_model.vector_size
        self.mlm_head = nn.Sequential(
            nn.LayerNorm(word_vec_size),  # Layer Normalization
            nn.Linear(word_vec_size, hidden_size),  # Hidden layer
            nn.GELU(),  # GELU Activation
            nn.Dropout(0.1),  # Dropout
            nn.Linear(hidden_size, self.vocab_size),  # Output layer to vocab size
        )
        # 分类输出层
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(word_vec_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )
        # 回归输出层
        self.regression_head = nn.Sequential(
            nn.LayerNorm(word_vec_size),
            nn.Linear(word_vec_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ 初始化权重 """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 使用正交初始化
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            else:
                init.xavier_uniform_(module.weight)
                # init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, word_embeds, segment_ids, masked_positions=None, task_type="mlm"):
        # (batch_size, seq_len, word_vec_size)
        # 添加[CLS]标记到输入序列
        batch_size = word_embeds.size(0)
        cls_embeds = self.cls_embedding.expand(batch_size, 1, -1)
        word_embeds = torch.cat([cls_embeds, word_embeds], dim=1)

        # 位置编码
        position_ids = torch.arange(
            word_embeds.size(1),
            dtype=torch.long,
            device=word_embeds.device
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)  # (1, seq_len, hidden_size)

        # 分段嵌入
        cls_segments = torch.zeros(batch_size, 1, dtype=torch.long, device=segment_ids.device)
        segment_ids = torch.cat([cls_segments, segment_ids], dim=1)
        segment_embeds = self.segment_embeddings(segment_ids)  # (batch_size, seq_len, hidden_size)

        # 组合嵌入
        embeddings = word_embeds + position_embeds + segment_embeds
        # (batch_size, seq_len, word_vec_size)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Transformer 编码
        padding_mask = (word_embeds == 0).all(dim=-1)
        transformer_output = self.transformer_encoder(embeddings, src_key_padding_mask=padding_mask)
        # (seq_len, batch_size, hidden_size)

        # 根据任务类型返回输出
        if task_type == "mlm":
            if masked_positions is None:
                raise ValueError("masked_positions must be provided for MLM task")
            # 获取有效位置（忽略填充值-1）
            mask = (masked_positions != -1)
            valid_positions = masked_positions[mask]  # 有效的位置索引
            # 生成样本索引（确定每个位置属于哪个样本）
            batch_size, max_masked = masked_positions.shape
            sample_indices = torch.arange(batch_size, device=word_embeds.device)[:, None].expand(-1, max_masked)
            sample_indices = sample_indices[mask]
            # 提取对应的输出
            transformer_output = transformer_output.contiguous()
            masked_outputs = transformer_output[sample_indices, valid_positions, :]
            # 计算logits
            logits = self.mlm_head(masked_outputs)
            return logits

        elif task_type == "classification":
            if self.num_classes is None:
                raise ValueError("num_classes must be specified for classification task")
            # 使用 [CLS] 位置的输出进行分类
            cls_output = transformer_output[:, 0, :]  # (batch_size, hidden_size)
            logits = self.classifier(cls_output)  # (batch_size, num_classes)
            return logits

        elif task_type == "regression":
            # 使用 [CLS] 位置的输出进行回归
            cls_output = transformer_output[:, 0, :]  # (batch_size, word_vec_size)
            output = self.regression_head(cls_output)  # (batch_size, 1)
            return output
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")



class BERTDataset(Dataset):
    def __init__(self, CDS_sequences, Five_UTR_sequences, Three_UTR_sequences, k, word_vectors_path, mask_prob=0.15,
                 num_segments=2, task_type="mlm", labels=None):

        self.k = k
        self.CDS_sequences = CDS_sequences
        self.Five_UTR_sequences = Five_UTR_sequences
        self.Three_UTR_sequences = Three_UTR_sequences
        self.sequences = [
            five + cds + three
            for five, cds, three in zip(Five_UTR_sequences, CDS_sequences, Three_UTR_sequences)
        ]
        self.word_vectors_path = word_vectors_path
        self.mask_prob = mask_prob
        self.num_segments = num_segments
        self.task_type = task_type
        self.labels = labels
        self.word_vectors = self.load_word_vectors()

        # 检查分类任务的标签
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

    # 修改后代码（删除填充逻辑）
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        k = self.k
        k_mers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        seq_vectors = np.array([self.word_vectors[word] for word in k_mers], dtype=np.float32)
        seq_vectors = torch.tensor(seq_vectors, dtype=torch.float32)
        seq_labels = torch.tensor(
            [self.word_vectors.key_to_index[word] for word in k_mers],
            dtype=torch.long
        )

        # 确定分段ID（无需填充）
        five_len = len(self.Five_UTR_sequences[idx])  # 假设Five_UTR_sequences存储原始字符串
        cds_len = len(self.CDS_sequences[idx])
        three_len = len(self.Three_UTR_sequences[idx])

        # 生成分段ID：Five UTR=0, CDS=1, Three UTR=2
        segment_ids = []
        segment_ids += [1] * (five_len - 1)  # Five UTR的k-mer数量
        segment_ids += [2] * cds_len  # CDS的k-mer数量
        segment_ids += [3] * (three_len - 1)  # Three UTR的k-mer数量

        # 直接返回未填充的数据
        if self.task_type == "mlm":
            masked_positions = []
            masked_labels = []
            input_word_embeds = seq_vectors.clone()
            for i in range(len(seq_vectors)):
                if np.random.rand() < self.mask_prob:
                    masked_positions.append(i)
                    masked_labels.append(seq_labels[i].item())
                    input_word_embeds[i] = torch.rand(self.word_vectors.vector_size)

            return {
                "word_embeds": input_word_embeds,
                "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
                "seq_labels": seq_labels,
                "masked_positions": torch.tensor(masked_positions, dtype=torch.long),
                "masked_labels": torch.tensor(masked_labels, dtype=torch.long),
            }

        elif self.task_type == "classification":
            # 为分类任务准备的输入是完整的词向量和 labels
            return {
                "word_embeds": seq_vectors,
                "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }

        elif self.task_type == "regression":
            # 为回归任务准备的输入是完整的词向量和连续值标签
            return {
                "word_embeds": seq_vectors,
                "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.float32)  # 连续值标签
            }

        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
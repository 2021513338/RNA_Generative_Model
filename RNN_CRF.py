import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from torchcrf import CRF
from gensim.models import Word2Vec


class AminoAcidDataset(Dataset):
    def __init__(self, amino_acid_seqs, labels, word_vectors_path):
        self.amino_acid_seqs = amino_acid_seqs
        self.labels = labels
        self.word_vectors_path = word_vectors_path
        self.word_vectors = self.load_word_vectors()

    def load_word_vectors(self):
        """
        加载预训练的词向量模型。
        如果模型不存在，则训练并保存。
        """
        if os.path.exists(self.word_vectors_path):
            print(f"Loading pre-trained word vectors from {self.word_vectors_path}...")
            return Word2Vec.load(self.word_vectors_path).wv
        else:
            print(f"No pre-trained word vectors found. Training new model...")
            w2v_model = Word2Vec(sentences=self.amino_acid_seqs, vector_size=128, window=5, min_count=1, workers=4)
            w2v_model.save(self.word_vectors_path)
            print(f"Model saved to {self.word_vectors_path}.")
            return w2v_model.wv

    def __len__(self):
        return len(self.amino_acid_seqs)

    def __getitem__(self, idx):
        seq = self.amino_acid_seqs[idx]
        label = self.labels[idx]
        seq_vec = np.array([self.word_vectors[word] for word in seq])
        seq_length = len(seq)
        return {
            'seq': torch.tensor(seq_vec, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long),
            'seq_length': seq_length
        }


# BiLSTM-CRF模型
class BiGRU_CRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, tag_size, output_dim, num_layers):
        super(BiGRU_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim // 2, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.crf = CRF(tag_size, batch_first=True)

    def forward(self, x, seq_lengths):

        x = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.gru(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        x = self.fc(x)

        return x

    def loss(self, emissions, tags):
        # 使用CRF计算损失
        mask = tags != 0
        return -self.crf(emissions, tags, mask=mask, reduction='mean')

    def predict(self, sentence, seq_lengths):
        emissions = self.forward(sentence, seq_lengths)
        # 使用CRF进行解码，得到最优标签序列
        return self.crf.decode(emissions)




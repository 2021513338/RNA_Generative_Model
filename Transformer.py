import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
import numpy as np
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

        # 不再进行填充，保留原始序列长度
        seq_vec = np.array([self.word_vectors[word] for word in seq])
        return {
            'seq': torch.tensor(seq_vec, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long),
            'seq_length': len(seq)  # 返回序列的实际长度
        }



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # 形状为 (max_len, 1, d_model)
        self.register_buffer('pe', pe)
        print(f"PositionalEncoding pe.shape: {self.pe.shape}")

    def forward(self, x):
        # 确保位置编码的长度与输入张量的序列长度一致
        print(f"PositionalEncoding input x.shape: {x.shape}")  # 输入张量形状
        pe_slice = self.pe[:x.size(0), :]
        print(f"PositionalEncoding pe_slice.shape: {pe_slice.shape}")  # 切片后的形状
        return x + pe_slice  # 添加位置编码


class TransformerModel(nn.Module):
    """
    Transformer 模型，包含编码器和解码器。
    """
    def __init__(self, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, trans_dropout):
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        # 编码器
        self.src_pos_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=trans_dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 解码器
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)  # 添加解码器的嵌入层
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=trans_dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 输出层
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):

        # 打印输入张量形状
        print(f"src.shape: {src.shape}")  # 编码器输入形状 (seq_length, batch_size, d_model)
        print(f"tgt.shape: {tgt.shape}")  # 解码器输入形状 (tgt_seq_length, batch_size, d_model)
        print(f"tgt max value: {torch.max(tgt)}, tgt min value: {torch.min(tgt)}")
        # 编码器
        src_emb = src * math.sqrt(self.d_model)  # 直接使用输入的词向量
        src_emb = self.src_pos_encoding(src_emb)
        print(f"src_emb.shape after positional encoding: {src_emb.shape}")
        encoder_output = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        print(f"encoder_output.shape: {encoder_output.shape}")  # 应该是 (seq_length, batch_size, d_model)

        # 解码器
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)
        print(f"tgt_emb.shape after positional encoding: {tgt_emb.shape}")

        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask=tgt_mask,
                                      memory_key_padding_mask=src_padding_mask,
                                      tgt_key_padding_mask=tgt_padding_mask)
        print(f"decoder_output.shape: {decoder_output.shape}")  # 应该是 (tgt_seq_length, batch_size, d_model)
        output = self.out(decoder_output)
        print(f"output.shape: {output.shape}")  # 应该是 (tgt_seq_length, batch_size, tgt_vocab_size)
        return output

    def predict(self, src, src_padding_mask=None, tgt_padding_mask=None):
        # 预测时，解码器的输入需要逐步生成
        batch_size = src.size(1)
        tgt = torch.zeros(1, batch_size, dtype=torch.long)  # 假设 0 是起始符

        for i in range(src.size(0)):
            current_len = tgt.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_len)
            output = self.forward(src, tgt, tgt_mask, src_padding_mask, tgt_padding_mask)
            next_token = output.argmax(dim=2)[-1, :].unsqueeze(0)  # 选择最后一步的输出
            tgt = torch.cat([tgt, next_token], dim=0)

        return tgt[1:, :]  # 去掉起始符


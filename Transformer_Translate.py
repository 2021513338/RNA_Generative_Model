from Transformer import TransformerModel
import torch
from gensim.models import Word2Vec
from Sequence_Convert import load_data_from_csv

def transform_protein_sequence(model_path, sequence, word_vectors_path):
    word_vectors = Word2Vec.load(word_vectors_path).wv

    tgt_vocab_size = 63
    d_model = 128  # 模型维度
    nhead = 8  # 多头注意力的头数
    num_encoder_layers = 6  # 编码器层数
    num_decoder_layers = 6  # 解码器层数
    dim_feedforward = 2048  # 前馈网络维度
    max_seq_length = 1000  # 最大序列长度
    trans_dropout = 0.1  # Transformer 的 dropout

    model = TransformerModel(tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                             dim_feedforward, max_seq_length, trans_dropout)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 将氨基酸序列转换为词向量
    seq_vec = [word_vectors[word] for word in sequence]
    seq_tensor = torch.tensor(seq_vec, dtype=torch.float).unsqueeze(1)
    print("Input shape", seq_tensor.shape)
    # 进行预测
    with torch.no_grad():
        prediction = model.predict(seq_tensor)
        print("Prediction shape:", prediction.shape)

    # 将预测结果转换为标签
    predicted_labels = prediction.squeeze(1).tolist()
    return predicted_labels


file_path = "CCDS_Seq_test.csv"
protein_column = "Sequence"
cds_column = "CDS"

amino_acid_seqs, labels = load_data_from_csv(file_path, protein_column, cds_column)

train_Transformer(amino_acid_seqs, labels, "word_vectors.model", "best_Transformer.pth")
test_sequence = ['A', 'C', 'C', 'D']
predicted_labels = transform_protein_sequence("best_Transformer.pth", test_sequence, word_vectors_path="word_vectors.model")
print("Predicted labels for the test sequence:", predicted_labels)
import torch
import pandas as pd
from Autoencoder import Autoencoder  # 导入Autoencoder类
from Sequence_Convert import convert_cds_to_numbers


def get_mRNA_low_dim(rna_seq, model_path):
    input_size = len(rna_seq)  # 假设所有序列长度相同
    encoding_dim = 64  # 低维表示的维度
    hidden_size = 256
    embedding_dim = 32
    model = Autoencoder(input_size, encoding_dim, hidden_size, embedding_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    sequence_tensor = torch.tensor(rna_seq, dtype=torch.long).unsqueeze(0)
    # 使用模型的encoder部分获取低维表示
    with torch.no_grad():  # 不需要计算梯度
        #low_dim_output = model(sequence_tensor)
        low_dim_output = model.encode(sequence_tensor)
    return low_dim_output

file_path = 'trimmed_sequence.csv'
df = pd.read_csv(file_path)
sequence = df['CDS_Sequence'].sample(n=1).iloc[0]
print(sequence)
rna_sequences = convert_cds_to_numbers(sequence)
low_dim_output = get_mRNA_low_dim(rna_sequences, 'best_autoencoder.pth')
print("Low-dimensional representation:")
print(low_dim_output.numpy())
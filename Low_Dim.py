import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Autoencoder import Autoencoder  # 导入Autoencoder类

def load_model(model_path, input_size, hidden_size, output_size):
    model = Autoencoder(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_low_dim_representation(model, sequence):
    model.eval()
    with torch.no_grad():
        return model.encoder(torch.tensor(sequence).float())

def get_mRNA_low_dim(rna_seq):
    input_size = len(rna_seq)
    hidden_size = 128
    output_size = 32
    # 加载模型
    model_path = './model.pth'  # .pth文件的路径
    model = load_model(model_path, input_size, hidden_size, output_size)

    # 获取低维表示
    low_dim_representation = get_low_dim_representation(model, rna_seq)
    print(low_dim_representation)

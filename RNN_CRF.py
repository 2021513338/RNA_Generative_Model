import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torchcrf import CRF
from gensim.models import Word2Vec

# 假设的氨基酸序列数据和对应的标签（密码子）
# 这里只是示例数据，你需要用实际的数据来训练模型
amino_acid_seqs = [['A', 'B', 'C', 'D'], ['B', 'C', 'D', 'E'], ['C', 'D', 'E', 'F']]
labels = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]  # 假设的标签，实际应为密码子索引

# 词向量预训练
w2v_model = Word2Vec(sentences=amino_acid_seqs, vector_size=100, window=5, min_count=1, workers=4)
word_vectors = w2v_model.wv

# 自定义数据集
class AminoAcidDataset(Dataset):
    def __init__(self, amino_acid_seqs, labels, word_vectors):
        self.amino_acid_seqs = amino_acid_seqs
        self.labels = labels
        self.word_vectors = word_vectors

    def __len__(self):
        return len(self.amino_acid_seqs)

    def __getitem__(self, idx):
        seq = self.amino_acid_seqs[idx]
        label = self.labels[idx]
        seq_vec = np.array([self.word_vectors[word] for word in seq])
        return {
            'seq': torch.tensor(seq_vec, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

# BiLSTM-CRF模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, tag_size, output_dim):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.crf = CRF(tag_size, batch_first=True)

    def forward(self, x):
        print("Input shape", x.shape)
        x, _ = self.gru(x)
        print("GRU shape", x.shape)
        x = self.fc(x)
        print("Output shape", x.shape)
        return x

    def loss(self, emissions, tags):
        # 使用CRF计算损失
        return -self.crf(emissions, tags)

    def predict(self, sentence):
        emissions = self.forward(sentence)
        # 使用CRF进行解码，得到最优标签序列
        return self.crf.decode(emissions)

def train_RNN_CRF(amino_acid_seqs, labels, word_vectors, model_save_path="best_RNN_CRF.pth"):
    # 数据加载器
    dataset = AminoAcidDataset(amino_acid_seqs, labels, word_vectors)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # 模型参数
    tag_size = 6
    input_dim = 100  # 词向量维度
    hidden_dim = 128  # RNN隐藏层维度
    output_dim = tag_size

    # 初始化模型、优化器和损失函数
    model = BiLSTM_CRF(input_dim, hidden_dim, tag_size, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    best_loss = float('inf')

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # 确保模型在训练模式下
        running_loss = 0.0
        for batch in data_loader:
            seq = batch['seq']# 增加一个维度以匹配LSTM输入
            label = batch['label']
            optimizer.zero_grad()
            output = model(seq)
            print("output shape", output.shape, "label shape:", label.shape)
            loss = model.loss(output, label)  # 去掉一个维度以匹配标签维度
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


# 预测函数
def transform_protein_sequence(model_path, sequence, word_vectors):
    tag_size = 6
    input_dim = 100  # 词向量维度
    hidden_dim = 128  # LSTM隐藏层维度
    output_dim = tag_size
    model = BiLSTM_CRF(input_dim, hidden_dim, tag_size, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 将氨基酸序列转换为词向量
    seq_vec = np.array([word_vectors[word] for word in sequence])
    seq_tensor = torch.tensor(seq_vec, dtype=torch.float).unsqueeze(0)  # 增加批次和序列维度
    print("Input shape", seq_tensor.shape)
    # 进行预测
    with torch.no_grad():
        prediction = model.predict(seq_tensor)

    # 将预测结果转换为标签
    predicted_labels = prediction[0]  # 取出批次中的第一个序列的预测结果
    return predicted_labels


# 示例使用
#train_RNN_CRF(amino_acid_seqs, labels, word_vectors, "best_RNN_CRF.pth")

test_sequence = ['A', 'B', 'C', 'D']
predicted_labels = transform_protein_sequence("best_RNN_CRF.pth", test_sequence, word_vectors)
print("Predicted labels for the test sequence:", predicted_labels)
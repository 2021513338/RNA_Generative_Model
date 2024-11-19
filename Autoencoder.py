import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Sequence_Convert import read_csv_to_numbers
class RNADataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim, embedding_dim):
        super(Autoencoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, encoding_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, embedding_dim),
            nn.ReLU(True),
            nn.Linear(embedding_dim, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def train_autocoder(rna_sequences, model_save_path="best_autoencoder.pth"):
    dataset = RNADataset(rna_sequences)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    input_size = len(rna_sequences[0])  # 假设所有序列长度相同
    encoding_dim = 64  # 低维表示的维度
    embedding_dim = 61
    model = Autoencoder(input_size, encoding_dim, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')

    epochs = 100
    for epoch in range(epochs):
        model.train()  # 确保模型在训练模式下
        running_loss = 0.0
        for sequences in data_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, sequences)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'New best model saved with loss: {avg_loss:.4f}')

    print(f'Training complete. Best model saved at: {model_save_path}')

def main():
    file_path = 'path_to_your_csv_file.csv'
    column_name = 'your_column_name'
    rna_sequences = read_csv_to_numbers(file_path, column_name)
    train_autocoder(rna_sequences)

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Sequence_Convert import read_csv_to_numbers
class RNADataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        item_tensor = torch.tensor(item, dtype=torch.long)
        #item_tensor = torch.tensor(item, dtype=torch.float32)
        return item_tensor

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim, hidden_size, embedding_dim):
        super(Autoencoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.encode_gru = nn.GRU(embedding_dim, hidden_size, num_layers=1, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size // 2),
            #nn.ReLU(True),
            nn.Linear(hidden_size // 2, encoding_dim),
            #nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_size // 2),
            #nn.ReLU(True),
            nn.Linear(hidden_size // 2, hidden_size),
            #nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(True),
        )
        self.decode_gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.decoder_final = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            #nn.Sigmoid()
        )

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        print("Embedding shape:", x.shape)
        x, h = self.encode_gru(x)  # x.shape=[batch_size, seq_len, hidden_size], h.shape=[num_layers , batch_size, hidden_size]
        h = h[0]
        print("GRU shape:", x.shape)
        x = self.encoder(h)  # 编码 [batch_size, encoding_dim]
        print("Encoder shape:", x.shape)
        x = self.decoder(x)  # 解码 [batch_size, hidden_size]
        print("Decoder shape:", x.shape)
        x = x.unsqueeze(1).repeat(1, x.size(1), 1)  # [batch_size, hidden_size, hidden_size]
        x, h = self.decode_gru(x)  # [batch_size, hidden_size, hidden_size]
        h = h[0]
        print("GRU_decoder shape:", x.shape)
        x = self.decoder_final(h)  # [batch_size, input_size]
        print("Decoder_final shape:", x.shape)
        return x

    def encode(self, x):
        x = self.embedding(x)
        x, h = self.encode_gru(x)
        h = h[0]
        x = self.encoder(h)
        return x

def train_autocoder(rna_sequences, model_save_path="best_autoencoder.pth"):
    dataset = RNADataset(rna_sequences)
    data_loader = DataLoader(dataset, batch_size=144, shuffle=True)

    input_size = len(rna_sequences[0])  # 假设所有序列长度相同
    encoding_dim = 64  # 低维表示的维度
    hidden_size = 256
    embedding_dim = 32
    model = Autoencoder(input_size, encoding_dim, hidden_size, embedding_dim)
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    best_loss = float('inf')

    epochs = 100
    for epoch in range(epochs):
        model.train()  # 确保模型在训练模式下
        running_loss = 0.0
        for sequences in data_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, sequences.float())
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

#def main():
    #file_path = 'trimmed_sequence.csv'
    #column_name = 'CDS_Sequence'
    #rna_sequences = read_csv_to_numbers(file_path, column_name)
    #train_autocoder(rna_sequences)


#if __name__ == '__main__':
    #main()
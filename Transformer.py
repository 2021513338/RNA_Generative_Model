import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
# 定义超参数
INPUT_DIM = len(EN.vocab.stoi)  # 英文词汇表大小
OUTPUT_DIM = len(DE.vocab.stoi)  # 德文词汇表大小
ENC_EMB_DIM = 256  # 编码器嵌入维度
DEC_EMB_DIM = 256  # 解码器嵌入维度
HID_DIM = 512  # 隐藏层维度
N_LAYERS = 3  # 层数
ENC_DROPOUT = 0.5  # 编码器dropout
DEC_DROPOUT = 0.5  # 解码器dropout
NUM_EPOCHS = 10 #迭代次数

# 定义模型
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.encoder = nn.Embedding(input_dim, emb_dim)
        self.decoder = nn.Embedding(output_dim, emb_dim)
        self.enc_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=8)
        self.dec_layer = nn.TransformerDecoderLayer(d_model=hid_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.enc_layer, num_layers=n_layers)
        self.transformer_decoder = nn.TransformerDecoder(self.dec_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, src, trg):
        src_emb = self.encoder(src)
        trg_emb = self.decoder(trg)
        output = self.transformer_encoder(src_emb)
        output = self.transformer_decoder(output, trg_emb)
        output = self.fc_out(output)
        return output

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg[:, :-1])

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1])

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def train_save_model(train_iterator, valid_iterator, clip, num_epochs, save_path):
    best_valid_loss = float('inf')
    model = Transformer(INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iterator, criterion)

        # 打印当前周期的信息
        print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}')

        # 保存具有最低验证损失的模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at Epoch {epoch + 1}")

def load_from_csv(X_filename='X.csv', y_filename='y.csv'):
    X = pd.read_csv(X_filename, header=None).values
    y = pd.read_csv(y_filename, header=None).values.ravel()
    return X, y

if __name__ == "__main__":
    '''
    X, y = make_classification(n_samples=100, n_features=303, n_classes=2, random_state=42)

    '''
    RNA_seq, Protein_seq = load_from_csv('RNA_seq.csv', 'Protein_seq.csv')

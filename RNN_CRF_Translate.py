import torch
from gensim.models import Word2Vec
from RNN_CRF import BiGRU_CRF
from Sequence_Convert import load_data_from_csv
import numpy as np

def transform_protein_sequence(model_path, sequence, word_vectors_path):
    word_vectors = Word2Vec.load(word_vectors_path).wv
    tag_size = 62
    input_dim = 100  # 词向量维度
    hidden_dim = 128  # LSTM隐藏层维度
    output_dim = tag_size
    model = BiGRU_CRF(input_dim, hidden_dim, tag_size, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 将氨基酸序列转换为词向量
    seq_vec = np.array([word_vectors[word] for word in sequence])
    seq_tensor = torch.tensor(seq_vec, dtype=torch.float).unsqueeze(0)  # 增加批次和序列维度
    seq_lengths = torch.tensor([len(sequence)], dtype=torch.long)
    print("Input shape", seq_tensor.shape)
    # 进行预测
    with torch.no_grad():
        prediction = model.predict(seq_tensor, seq_lengths)

    # 将预测结果转换为标签
    predicted_labels = prediction[0]  # 取出批次中的第一个序列的预测结果
    return predicted_labels

# 假设的氨基酸序列数据和对应的标签（密码子）
# 这里只是示例数据，你需要用实际的数据来训练模型
#amino_acid_seqs = [['A', 'B', 'C', 'D'], ['B', 'C', 'D', 'E'], ['C', 'D', 'E', 'F'], ['A', 'B'], ['A', 'B', 'C', 'D', 'E']]
#labels = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [1, 2], [1, 2, 3, 4, 5]]  # 假设的标签，实际应为密码子索引
file_path = "CCDS_Seq_test.csv"
protein_column = "Sequence"
cds_column = "CDS"

amino_acid_seqs, labels = load_data_from_csv(file_path, protein_column, cds_column)
# 示例使用
#train_RNN_CRF(amino_acid_seqs, labels, "word_vectors.model", "best_RNN_CRF.pth")

test_sequence = ['A', 'C', 'C', 'D']
predicted_labels = transform_protein_sequence("best_RNN_CRF.pth", test_sequence, "word_vectors_path.model")
print("Predicted labels for the test sequence:", predicted_labels)

'''
def train_RNN_CRF(amino_acid_seqs, labels, word_vectors_path, model_save_path):
    # 数据加载器
    dataset = AminoAcidDataset(amino_acid_seqs, labels, word_vectors_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    # 模型参数
    tag_size = 62
    input_dim = 100  # 词向量维度
    hidden_dim = 128  # RNN隐藏层维度
    output_dim = tag_size

    # 初始化模型、优化器和损失函数
    model = BiGRU_CRF(input_dim, hidden_dim, tag_size, output_dim)
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
            seq_lengths = batch['seq_lengths']

            optimizer.zero_grad()
            output = model(seq, seq_lengths)
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
'''
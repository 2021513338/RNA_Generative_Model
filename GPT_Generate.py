import torch
from gensim.models import Word2Vec
from GPT import GPT
def generate_GTP(model_save_path, start_sequence, word_vectors_path, max_len=100, temperature=1.0, top_k=50):
    word_vectors = Word2Vec.load(word_vectors_path).wv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(word_vectors.key_to_index) + 1
    max_len = 200  # 最大序列长度
    model = GPT(vocab_size=vocab_size, max_len=max_len)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()  # 将模型设置为评估模式
    device = next(model.parameters()).device  # 获取模型所在的设备

    # 将初始序列转换为 k-mer 并获取对应的词向量
    k = 3  # 假设 k=3
    k_mers = [start_sequence[i:i + k] for i in range(len(start_sequence) - k + 1)]
    input_vectors = [word_vectors[word] for word in k_mers]
    input_vectors = torch.tensor(input_vectors, dtype=torch.float).unsqueeze(0).to(device)  # 添加 batch 维度并移动到设备

    generated_sequence = start_sequence

    with torch.no_grad():  # 禁用梯度计算
        for _ in range(max_len - len(k_mers)):
            # 获取模型的输出
            output = model(input_vectors)
            logits = output[:, -1, :] / temperature  # 获取最后一个时间步的输出并应用温度

            # 应用 top-k 过滤
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                logits[logits < top_k_values[:, -1].unsqueeze(1)] = -float('Inf')

            # 从 logits 中采样下一个词
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 将生成的词添加到序列中
            next_kmer = word_vectors.index_to_key[next_token.item()]
            generated_sequence += next_kmer

            # 更新输入向量
            next_vector = word_vectors[next_kmer]
            next_vector = torch.tensor(next_vector, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
            input_vectors = torch.cat([input_vectors, next_vector], dim=1)

    return generated_sequence

# 假设 model 是已经训练好的 GPT 模型，word_vectors 是预训练的词向量模型
start_sequence = "CTC"
generated_sequence = generate_GTP('best_GPT.pth', start_sequence, 'RNA_word_vec.model', max_len=50, temperature=0.7, top_k=50)
print(generated_sequence)
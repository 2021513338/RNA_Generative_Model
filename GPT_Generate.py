import torch
from gensim.models import Word2Vec
from GPT import GPT
def generate_GTP_ktop(model_save_path, start_sequence, word_vectors_path, temperature=1.0, top_k=50):
    word_vectors = Word2Vec.load(word_vectors_path).wv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(word_vectors.key_to_index)
    max_len = 200  # 最大序列长度
    model = GPT(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()  # 将模型设置为评估模式

    # 将初始序列转换为 k-mer 并获取对应的词向量
    k = 3
    k_mers = [start_sequence[i:i + k] for i in range(len(start_sequence) - k + 1)]
    input_vectors = [word_vectors[word] for word in k_mers]
    input_vectors = torch.tensor(input_vectors, dtype=torch.float).unsqueeze(0).to(device)  # 添加 batch 维度并移动到设备

    eos_index = word_vectors.key_to_index['<eos>']
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
            if next_token.item() == eos_index:
                break  # 检测到<eos>则终止
            generated_sequence += next_kmer

            # 更新输入向量
            next_vector = word_vectors[next_kmer]
            next_vector = torch.tensor(next_vector, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
            input_vectors = torch.cat([input_vectors, next_vector], dim=1)

    return generated_sequence


def generate_GTP_beam(model_save_path, start_sequence, word_vectors_path, temperature=1.0, beam_width=5):
    # 加载词向量和模型（同原有代码）
    word_vectors = Word2Vec.load(word_vectors_path).wv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(word_vectors.key_to_index)
    max_len = 200  # 最大序列长度
    model = GPT(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    k = 3  # k-mer长度
    eos_token = '<eos>'
    eos_index = word_vectors.key_to_index[eos_token]

    # 初始序列处理（生成k-mers并转换为输入向量）
    initial_k_mers = [start_sequence[i:i + k] for i in range(len(start_sequence) - k + 1)]
    input_vectors = torch.tensor([word_vectors[word] for word in initial_k_mers], dtype=torch.float).unsqueeze(0).to(device)

    # 初始化束搜索候选池
    candidates = [
        {
            "input_vectors": input_vectors,
            "sequence": start_sequence,
            "log_prob": 0.0,  # 对数概率累积
            "is_finished": False
        }
    ]

    for _ in range(max_len - len(initial_k_mers)):
        # 扩展所有未终止的候选
        new_candidates = []
        for candidate in candidates:
            if candidate["is_finished"]:
                new_candidates.append(candidate)  # 已终止的候选直接保留
                continue

            # 模型预测
            with torch.no_grad():
                output = model(candidate["input_vectors"])
                logits = output[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)

            # 取概率最高的前beam_width个词
            top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)
            top_probs = top_probs.log().cpu().numpy().flatten()  # 转换为对数概率
            top_indices = top_indices.cpu().numpy().flatten()

            # 生成新候选
            for i in range(beam_width):
                next_token = top_indices[i]
                next_log_prob = candidate["log_prob"] + top_probs[i]
                next_kmer = word_vectors.index_to_key[next_token]
                new_sequence = candidate["sequence"] + next_kmer

                # 构建新的输入向量
                next_vector = torch.tensor(word_vectors[next_kmer], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
                new_input_vectors = torch.cat([candidate["input_vectors"], next_vector], dim=1)

                # 判断是否终止
                is_finished = (next_token == eos_index) or (len(new_sequence) >= max_len)

                new_candidates.append({
                    "input_vectors": new_input_vectors,
                    "sequence": new_sequence,
                    "log_prob": next_log_prob,
                    "is_finished": is_finished
                })

        # 按对数概率排序，保留前beam_width个候选
        new_candidates.sort(key=lambda x: x["log_prob"], reverse=True)
        candidates = new_candidates[:beam_width]

        # 检查所有候选是否终止
        if all(c["is_finished"] for c in candidates):
            break

    # 选择对数概率最高的候选（去除<eos>）
    best_candidate = max(candidates, key=lambda x: x["log_prob"])
    final_sequence = best_candidate["sequence"].split(eos_token)[0]  # 截取到<eos>之前

    return final_sequence


# 假设 model 是已经训练好的 GPT 模型，word_vectors 是预训练的词向量模型
start_sequence = "CTC"
generated_sequence = generate_GTP_ktop('best_GPT.pth', start_sequence, 'UTR_RNA_vec.model', temperature=0.7, top_k=50)
print(generated_sequence)
generated_sequence = generate_GTP_beam('best_GPT.pth', start_sequence, 'UTR_RNA_vec.model', temperature=0.7, beam_width=5)
print(generated_sequence)
import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, shared_embedding=None):
        super(Encoder, self).__init__()
        if shared_embedding is None:
            self.embeddings = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embeddings = shared_embedding
    def forward(self, input_ids):
        embedded_input = self.embeddings(input_ids)
        # 假设这里是编码器的其他计算
        return embedded_input
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, shared_embedding=None):
        super(Decoder, self).__init__()
        if shared_embedding is None:
            self.embeddings = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embeddings = shared_embedding
    
    def forward(self, input_ids, encoder_outputs):
        embedded_input = self.embeddings(input_ids)
        # 假设这里是解码器的其他计算
        return embedded_input
class EncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, tie_word_embeddings=True, tie_encoder_decoder=False):
        super(EncoderDecoderModel, self).__init__()
        self.tie_word_embeddings = tie_word_embeddings
        self.tie_encoder_decoder = tie_encoder_decoder
        self.shared_embedding = nn.Embedding(vocab_size, embed_dim) # 共享的嵌入
        # 编码器和解码器用同样的词嵌入
        self.encoder = Encoder(vocab_size, embed_dim, shared_embedding=self.shared_embedding)
        self.decoder = Decoder(vocab_size, embed_dim, shared_embedding=self.shared_embedding)
        # 如果输出层也共享词嵌入
        if self.tie_word_embeddings:
            self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)
            # 将嵌入层的权重矩阵转置赋值给输出层的权重
            # 由于 nn.Linear 层的权重默认形状是 (out_features, in_features)，而嵌入层的权重形状是
            # (vocab_size, embed_dim)，因此 self.output_layer 的权重实际上已经是正确的形状，不需要
            # 额外的转置操作。
            self.output_layer.weight = self.shared_embedding.weight
        else:
            self.output_layer = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids, target_ids):
        encoder_outputs = self.encoder(input_ids)
        decoder_outputs = self.decoder(target_ids, encoder_outputs)
        # print(decoder_outputs.shape) [2, 10, 512]
        return self.output_layer(decoder_outputs)
vocab_size = 10000
embed_dim = 512
model = EncoderDecoderModel(vocab_size, embed_dim, tie_word_embeddings=True, tie_encoder_decoder=True)
input_ids = torch.randint(0, vocab_size, (2, 10))  # 批次大小为2，序列长度为10
target_ids = torch.randint(0, vocab_size, (2, 10))
output = model(input_ids, target_ids)
print(output.shape)  # 输出形状应该是 (2, 10, 10000)
# 当编码器和解码器处理相同语言的文本时，例如在摘要生成任务中，共享编码器和解码器的某些组件（如自注意力层或前馈层
# ）确实有一定的合理性。这种共享策略可以在某些情况下带来一些好处：
# 共享自注意力层或前馈层的好处
# 减少参数量：
#     共享权重可以显著减少模型的参数量。这对于资源受限的设备或需要快速推理的应用特别有用。
# 知识共享：
#     共享权重意味着编码器和解码器可以共享已经学习到的特征。例如，在摘要生成任务中，编码器学到的句子结构和上下文关系也可以被解码器所利用。
# 泛化能力增强：
#     共享权重可以帮助模型更好地泛化到未见过的数据，因为它迫使模型学习更加通用的表示。
# 减少过拟合风险：
#     在数据量较小的情况下，共享权重可以减少过拟合的风险，因为模型参数更少，更容易训练。
# 以下是几种可能适用共享编码器和解码器组件的场景：
# 同一种语言的任务：
#     当编码器和解码器处理相同语言的文本时，可以考虑共享自注意力层或前馈层。例如，在摘要生成、问答系统等任务中，源语言和目标语言相同。
# 多任务学习：
#     在多任务学习中，多个任务可能会受益于共享的底层特征。共享自注意力层或前馈层可以帮助提取跨任务的共同特征。
# 迁移学习：
#     在迁移学习中，可以先在一个任务上训练模型，然后将学到的特征迁移到另一个相关任务上。在这种情况下，共享部分层可以帮助新任务更快地学习。
# 示例代码：共享自注意力层或前馈层
# 假设我们在一个摘要生成任务中，源语言和目标语言相同，并且希望共享编码器和解码器的部分层。下面是一个示例代码
# ，展示了如何定义编码器和解码器，并共享部分层：
import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        return src

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
    
    def forward(self, src):
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        return src

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = SelfAttentionLayer(d_model, nhead, dropout)
        self.feed_forward = FeedForwardLayer(d_model, dim_feedforward, dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.self_attention(src, src_mask, src_key_padding_mask)
        src = self.feed_forward(src)
        return src

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, transformer_layer):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.transformer = transformer_layer
    
    def forward(self, input_ids):
        embedded_input = self.embeddings(input_ids)
        encoder_output = self.transformer(embedded_input)
        return encoder_output

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, transformer_layer):
        super(Decoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.transformer = transformer_layer
    
    def forward(self, input_ids, encoder_outputs, tgt_mask=None, memory_mask=None):
        embedded_input = self.embeddings(input_ids)
        decoder_output = self.transformer(embedded_input)
        return decoder_output

class EncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, tie_word_embeddings=False, share_transformer_layers=False):
        super(EncoderDecoderModel, self).__init__()
        self.tie_word_embeddings = tie_word_embeddings
        self.share_transformer_layers = share_transformer_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        if share_transformer_layers:
            self.transformer_layer = TransformerLayer(embed_dim, 8)
            self.encoder = Encoder(vocab_size, embed_dim, self.transformer_layer)
            self.decoder = Decoder(vocab_size, embed_dim, self.transformer_layer)
        else:
            self.encoder = Encoder(vocab_size, embed_dim, TransformerLayer(embed_dim, 8))
            self.decoder = Decoder(vocab_size, embed_dim, TransformerLayer(embed_dim, 8))
        
        if self.tie_word_embeddings:
            self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)
            self.output_layer.weight = self.embedding.weight
        else:
            self.output_layer = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids, target_ids):
        encoder_outputs = self.encoder(input_ids)
        tgt_mask = self.generate_square_subsequent_mask(target_ids.size(1)).to(target_ids.device)
        decoder_outputs = self.decoder(target_ids, encoder_outputs)
        return self.output_layer(decoder_outputs)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
# 创建模型实例
vocab_size = 10000
embed_dim = 512
model = EncoderDecoderModel(vocab_size, embed_dim, tie_word_embeddings=True, share_transformer_layers=True)
# 创建输入数据
input_ids = torch.randint(0, vocab_size, (2, 10))  # 批次大小为2，序列长度为10
target_ids = torch.randint(0, vocab_size, (2, 10))
# 前向传播
output = model(input_ids, target_ids)
print(output.shape)  # 输出形状应该是 (2, 10, 10000)
# 滑动窗口（Sliding Window）是一种算法技术，通常用于处理序列数据，如时间序列、文本序列等。
# 它通过在数据集上移动一个固定大小的窗口来进行计算或处理，每次移动一定步长后，窗口内的数据会
# 被用来执行特定的操作。这种方法广泛应用于计算机视觉、自然语言处理（NLP）、信号处理等领域。
# 滑动窗口机制在注意力机制中的应用主要体现在第二步（计算注意力分数）之前。具体来说，滑动窗口机制通过将序列
# 分割成较短的段落，使得每个段落内的查询向量只与该段落内的键向量计算注意力分数，而不是与整个序列的键向量计算。
# 工作流程
# 分割序列：将长序列分割成多个较短的段落。
# 处理每个段落：对于每个段落，计算其内部的查询、键和值向量。
# 计算注意力分数：只在每个段落内部计算注意力分数。
# 应用注意力机制：在每个段落内应用注意力机制，包括 softmax 和加权求和。
# 整合结果：将各个段落的注意力输出整合在一起，形成最终的注意力输出。
# 假设我们有一个长度为 100 的序列，使用滑动窗口大小为 20，步长为 10：
# 初始化窗口：窗口从位置 0 开始，处理序列 [0, 20]。
# 计算注意力：只在 [0, 20] 范围内计算注意力分数。
# 移动窗口：窗口移动到位置 10，处理序列 [10, 30]。
# 重复步骤2和3：直到处理完整个序列。
# 注意事项
# 边界处理：对于最后一个不完整窗口，可以采用补零或其他方法来处理不足的部分。
# 信息连续性：滑动窗口的重叠部分有助于保持信息的连续性，避免断层效应。
import numpy as np
# 定义序列长度和滑动窗口参数
sequence_length = 100
window_size = 20
stride = 10
# 创建一个示例序列
sequence = np.arange(sequence_length)  # 创建一个从 0 到 99 的序列
# 初始化窗口起始位置
start_pos = 0
# 存储每个窗口的数据
window_results = []
# 移动窗口并处理数据
while start_pos + window_size <= sequence_length:
    end_pos = start_pos + window_size
    current_window = sequence[start_pos:end_pos]
    # 处理当前窗口的数据
    print(f"处理窗口 [{start_pos}, {end_pos}) 数据：{current_window}")
    # 存储处理结果
    window_results.append(current_window)
    start_pos += stride
# 处理最后一个不完整的窗口（如果有的话）
if start_pos < sequence_length:
    end_pos = sequence_length
    current_window = sequence[start_pos:end_pos]
    print(f"处理最后一个窗口 [{start_pos}, {end_pos}) 数据：{current_window}")
    window_results.append(current_window)
# 输出所有窗口的数据
print("所有窗口的数据：")
for i, window in enumerate(window_results):
    print(f"窗口 {i}: {window}")
# 假设我们有一个长度为 100 的序列，并且我们选择一个大小为 20 的窗口，步长为 10。我们将分别处理查询向量和键向量
# 定义序列长度和滑动窗口参数
sequence_length = 100
window_size = 20
stride = 10
# 创建一个示例序列
sequence = np.random.rand(sequence_length, 32)  # 假设序列每个位置有 32 维特征
# 初始化窗口起始位置
start_pos = 0
# 存储每个窗口的处理结果
window_results = []
while start_pos + window_size <= sequence_length:
    end_pos = start_pos + window_size
    current_window = sequence[start_pos:end_pos]
    # 分割查询和键
    queries = current_window[:, :8]  # 假设查询向量是前 8 维
    keys = current_window[:, 8:16]  # 假设键向量是接下来的 8 维
    values = current_window[:, 16:]  # 假设值向量是最后的 16 维
    # 计算注意力分数
    attention_scores = np.dot(queries, keys.T) / np.sqrt(keys.shape[-1])
    # 应用 softmax
    attention_probs = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
    # 加权求和得到注意力输出
    context_vector = np.dot(attention_probs, values)
    # 存储处理结果
    window_results.append(context_vector)
    # 更新窗口起始位置
    start_pos += stride
# 处理最后一个不完整的窗口（如果有的话）
if start_pos < sequence_length:
    end_pos = sequence_length
    current_window = sequence[start_pos:end_pos]
    queries = current_window[:, :8]
    keys = current_window[:, 8:16]
    values = current_window[:, 16:]
    attention_scores = np.dot(queries, keys.T) / np.sqrt(keys.shape[-1])
    attention_probs = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
    context_vector = np.dot(attention_probs, values)
    window_results.append(context_vector)
# 整合结果
final_output = np.concatenate(window_results, axis=0)
# 用来演示子类覆盖父类(object)的__setattr__和__getattribute__
class Person:
     # attribute_map:类属性
    attribute_map = {
        'age': 'years_old',
    }
    # 在Python中，所有对对象属性的操作实际上都是通过特殊的方法（称为“魔术方法”或“特殊方法”）来完成的。
    # 这些方法允许开发者自定义对象的行为，例如如何设置或获取属性。
    def __init__(self, name, age):
        # print(self.attribute_map)
        # 在这里设置属性时,实际上调用的是父类的__setattr__方法,因为这里子类重写了这个方法,调用的会是
        # 子类中的__setattr__方法,这里会判断当前属性是否在attribute_map的键里面,如果在,就获取它对应的
        # 值,这里是years_old,设置的时候,设置的其实是years_old
        self.name = name
        self.age = age
    # 重写 __setattr__ 方法
    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)
    # 重写 __getattribute__ 方法
    # 当调用某个key对应的值时,它首先会判断key是否在attribute_map的健里面,如果在,就获取它对应的值
    # 之后调用父类的__getattribute__方法获取对应的值
    def __getattribute__(self, key):
        print(key,key in super().__getattribute__("attribute_map"))
        # 如果key在attribute_map的键里面
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            # 获取attribute_map中key对应的值
            key = super().__getattribute__("attribute_map")[key]
        print(key)
        return super().__getattribute__(key)
# 匹配配置文件
_re_configuration_file = re.compile(r"config\.(.*)\.json")
p = Person('Alice', 30)
print(p.age)  # 输出 30
# 使用 years_old 访问 (别名)
print(p.years_old)  # 输出 30
# 使用 years_old 设置 (别名)
p.years_old = 31
# 验证 age 是否更新
print(p.age)  # 输出 31
type(getattr(torch,'float32'))
# 在前馈层前的拆分特征维度
class Model:
    def __init__(self, **kwargs):
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
    def process_input(self, input_data):
        if self.chunk_size_feed_forward > 0:
            batch_size, sequence_length, feature_dim = input_data.shape
            # 初始化一个空列表来存储处理后的结果
            processed_results = []
            # 遍历每个样本
            for batch_idx in range(batch_size):
                sample = input_data[batch_idx] # (s,d)
                # 将每个样本的特征向量按照 chunk_size_feed_forward 进行分割
                chunks = [
                    sample[:, i:i + self.chunk_size_feed_forward]
                    for i in range(0, feature_dim, self.chunk_size_feed_forward)
                ]
                # 对每个特征块进行前馈网络处理
                chunk_results = [self.feed_forward(chunk) for chunk in chunks]
                # 合并处理后的特征块
                final_result = self.merge_results(chunk_results)
                # 添加到最终结果列表中
                processed_results.append(final_result)
            # 将处理后的结果堆叠成最终输出
            final_output = np.stack(processed_results, axis=0)
        else:
            final_output = self.feed_forward(input_data)
        return final_output
    def feed_forward(self, data_chunk):
        # 前馈网络处理逻辑
        # 示例中简单返回输入
        return data_chunk
    def merge_results(self, chunk_results):
        # 合并各个块的结果
        # 示例中简单拼接
        return np.concatenate(chunk_results, axis=-1)
# 创建模型实例
model = Model(chunk_size_feed_forward=16)
# 处理输入数据
input_data = np.random.rand(64, 128, 256)  # 假设有 64 个样本，每个样本有 128 个序列长度，每个序列有 256 个特征
result = model.process_input(input_data)
print(result.shape)  # 应该输出 (64, 128, 256)
def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2) # q_len,k_len
    # 缩放因子
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype) # 初始化的掩码
    # 如果传入使用因果掩码
    if is_causal:
        # 断言attn_mask is None,这时不该有填充掩码
        assert attn_mask is None
        # 因果掩码
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        # 设置因果掩码中当前token之后的值为-inf,很大的负数
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    # 如果传入了mask
    if attn_mask is not None:
        # 填充注意力掩码中False的位置为-inf
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
boxes1 = torch.tensor([
    [10, 10, 20, 20],
    [20, 20, 30, 30]
], dtype=torch.float32)

boxes2 = torch.tensor([
    [15, 15, 25, 25],
    [25, 25, 35, 35]
], dtype=torch.float32)
# 将 boxes1 的形状从 [N, 4] 变为 [N, 1, 4]
boxes1_expanded = boxes1[:, None, :]  # 形状变为 [N, 1, 4]
# 保留 boxes2 的左上角坐标，形状为 [M, 2]
boxes2_left_top = boxes2[:, :2]  # 形状为 [M, 2]
# 计算最大左上角坐标
left_top = torch.max(boxes1_expanded[:, :, :2], boxes2_left_top)
left_top
# 假设的形状
batch_size = 32
n_heads = 8
seq_length = 10
key_length = 10

# 创建位置偏差
position_bias = torch.randn(batch_size, n_heads, seq_length, key_length)

# 创建修剪掩码
self.pruned_heads = [2]
mask = torch.ones(n_heads)
mask[list(self.pruned_heads)] = 0
position_bias_masked = position_bias[:, mask.bool()]

# 创建头部掩码
layer_head_mask = torch.ones(n_heads)
layer_head_mask[[3, 4]] = 0  # 关闭第 3 和第 4 个头

# 计算注意力权重
scores = torch.randn(batch_size, n_heads, seq_length, key_length)
scores += position_bias_masked
attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

# 应用头部掩码
if layer_head_mask is not None:
    attn_weights = attn_weights * layer_head_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
# 创建一个随机张量
x = torch.randn(3, 3)
print("Original x:", x)
# 使用 inplace=False
relu_non_inplace = nn.ReLU(inplace=False)
y = relu_non_inplace(x)
print("Non-inplace operation (y):", y)
print("x after non-inplace operation:", x)
# 使用 inplace=True
relu_inplace = nn.ReLU(inplace=True)
z = relu_inplace(x)
print("Inplace operation (z):", z)
print("x after inplace operation:", x) # x的值被修改
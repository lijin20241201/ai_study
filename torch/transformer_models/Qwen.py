class Qwen2Config(PretrainedConfig):
    model_type = "qwen2"
    # 如果你在模型配置中设置了 keys_to_ignore_at_inference = ["past_key_values"]，那么无论在什么情况下，
    # 模型在推理时都会忽略 past_key_values 的存在。这意味着即使你传入了 past_key_values，模型也会当作没有提供这
    # 些值一样工作，从而不会利用任何先前计算的结果来进行加速。
    # 如果你希望在实际部署的应用程序中利用 past_key_values 来加速推理过程并保持上下文一致性，那么你应该确保不忽略这些
    # 键值。具体来说，你可以采取以下措施之一：
    # 移除或注释掉设置：直接删除或注释掉 keys_to_ignore_at_inference = ["past_key_values"] 的设置。
    # 动态配置：如果你的应用程序需要在不同场景下切换是否使用 past_key_values，可以考虑动态配置这一选项，根据运行时的参
    # 数来决定是否忽略 past_key_values。
    # keys_to_ignore_at_inference = ["past_key_values"] if ignore_past else []
    # 使用模型参数控制：在调用模型方法时，可以通过传入参数来控制是否使用 past_key_values。例如，在调用 model.generate()
    # 方法时，可以通过设置参数来控制是否使用 past_key_values。
    # outputs = model.generate(input_ids, past_key_values=past_key_values if use_past else None)
    # 这样做的好处是，你可以根据实际需求灵活地控制是否使用缓存的 key-value 对，同时保留代码的可读性和可维护性。
    # 总结来说，如果你的应用需要利用连续上下文并通过 past_key_values 加速推理，那么你不应该设置 keys_to_ignore_at_inference 
    # = ["past_key_values"]，而是根据实际情况来决定是否使用这些键值对。
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        # 用途：用于初始化嵌入层（embedding layer），以及作为最终全连接层（fully connected layer）的输出维度。
        vocab_size=151936,# 词汇表的大小，即模型可以识别的不同单词或标记的数量。
        hidden_size=4096,# 含义：隐藏层的维度，即每个Transformer编码器或解码器层的输出向量的大小。
        # 决定了模型内部状态的表示能力
        intermediate_size=22016,# 前馈神经网络（feed-forward network, FFN）中间层的维度。
        # FFN通常由两个线性层组成，第一个线性层的输出维度为 intermediate_size，用于提升模型的学习能力。
        num_hidden_layers=32,# Transformer模型中编码器或解码器堆叠的层数。增加模型的深度，以增强其捕捉复杂特征的能力。
        num_attention_heads=32,# 含义：每个Transformer层中多头注意力机制（multi-head attention mechanism）的头数。
        # 允许多个并行的注意力机制运行，从而捕捉不同的特征。
        num_key_value_heads=32,# 每层中用于计算键（Key）和值（Value）的注意力头的数量
        # 优化计算资源，有时候为了节省计算成本，可以设置 num_key_value_heads 小于 num_attention_heads。
        hidden_act="silu",# 隐藏层使用的激活函数。引入非线性，使模型能够学习复杂的映射关系
        max_position_embeddings=32768,# 模型支持的最大位置嵌入的长度。决定了模型能够处理的最大序列长度。
        initializer_range=0.02,# 模型权重初始化的标准差范围。控制模型参数初始化时的随机性。
        rms_norm_eps=1e-6,# RMSNorm 层中使用的数值稳定性项。防止除法运算中的除零错误。
        use_cache=True,# 是否使用缓存机制来存储过去计算的结果。在生成任务中，可以加速推理过程
        tie_word_embeddings=False,# 是否共享输入嵌入层（input embedding）和输出嵌入层（output embedding）的权重。
        # 减少模型参数数量，有时可以提高模型性能。
        rope_theta=10000.0,# 旋转位置嵌入（Rotary Positional Embedding）的基本周期。
        # 帮助模型理解不同位置的相对关系。
        use_sliding_window=False,# 含义：是否使用滑动窗口机制。
        # 用途：在处理长序列时，可以减少内存消耗。
        sliding_window=4096, # 含义：滑动窗口的大小。定义了滑动窗口覆盖的序列长度。
        max_window_layers=28,# 含义：最多可以有多少层使用滑动窗口机制。
        # 用途：限制滑动窗口机制使用的层数，平衡计算效率和模型性能。
        attention_dropout=0.0,# 含义：注意力机制中的Dropout概率。随机丢弃一些注意力权重来防止过拟合。
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        # 向后兼容
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6): # 隐藏层的大小
        super().__init__()
        # 一个可学习的权重参数，初始化为全 1 张量。
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 用于防止除零错误的小常数。
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        # 记录输入张量的数据类型，以便最终转换回原始类型。
        input_dtype = hidden_states.dtype
        # 转换为 torch.float32 类型，以确保数值稳定性。
        hidden_states = hidden_states.to(torch.float32)
        # 这里计算了方差，并且使用了 torch.rsqrt 函数来计算方差加上一个小的常数 ϵϵ 的倒数平方根
        # 。接着，这个值被用来对输入进行归一化。
        # 这里，E[(x2)]表示 x 的元素平方的均值，也就是方差的一个变形版本。因此，RMSNorm 本质
        # 上是对输入张量的每个元素进行除以其自身的 RMS 值的操作。
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # rsqrt是开方的倒数
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 应用可学习的权重,其中 weight是一个可学习的参数，用于缩放规范化后的张量。
        return self.weight * hidden_states.to(input_dtype)

# 用于生成旋转位置嵌入。这种嵌入方法在 Transformer 模型中用于捕捉序列中的位置信息，尤其适用于长序列任务。
# 通过旋转的方式将位置信息编码到嵌入向量中。具体步骤如下:
# 生成频率：通过指数函数生成一系列频率值。计算正弦和余弦：利用生成的频率计算正弦和余弦值
# ,旋转嵌入：将输入向量按一定规则旋转，以嵌入位置信息。
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        # 最大位置嵌入的长度，默认为 2048,base：基数，默认为 10000。。
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # inv_freq：计算频率的逆值。
        # 位置列表先归一化(从绝对位置变成相对位置),之后取指数(1--接近10000),之后取倒数,位置从1--越来越小
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        # register_buffer：将 inv_freq 注册为缓冲区，以便在模型保存和加载时保持不变。
        # register_buffer 方法用于注册一个非训练的缓冲区（buffer），这意味着它不会被梯度更新。当你使用 register_buffer 注册一个缓
        # 冲区时，它会被保存在模型的状态字典（state dict）中，并且在模型保存和加载时也会被序列化。
        # persistent=True：缓冲区会出现在模型的状态字典中，并且会被序列化和加载。
        # persistent=False：缓冲区不会出现在模型的状态字典中，但在实际保存和加载时，仍然会被序列化并加载。
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Build here to make `torch.jit.trace` work.生成正弦和余弦缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # t 是一个包含位置索引的张量，形状为 (seq_len,)。
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        # torch.outer：计算外积，得到一个形状为 (seq_len, dim/2) 的张量
        freqs = torch.outer(t, self.inv_freq) # 计算频率。
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # 拼接频率。emb 的形状为 (seq_len, dim)。
        # 在旋转位置嵌入（RoPE）中，我们通常将嵌入向量分为两个部分，并分别应用正弦和余弦变换。具体来说：
        # 对于每个位置 tt，计算频率 ff，得到一个形状为 (seq_len, dim/2) 的张量。
        # 将频率张量拼接两次，得到一个形状为 (seq_len, dim) 的张量。
        # 这样做的原因是，我们将嵌入向量分为两部分，每部分对应一个频率值。
        emb = torch.cat((freqs, freqs), dim=-1) 
        # cos_cached 和 sin_cached：注册正弦和余弦缓存。
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    def forward(self, x, seq_len=None): # x：输入张量。
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 如果 seq_len 大于已缓存的最大长度，则重新生成缓存。
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return ( # 返回正弦和余弦缓存的切片。
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size # d
        self.intermediate_size = config.intermediate_size # hd
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) # d-->hd
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)# d-->hd
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False) # hd-->d
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self, hidden_state): # (h,s,d)
        # 门控信号生成：gate_proj(hidden_state) 生成门控信号
        # 特征调整：gate_output 与 up_output 相乘，将门控信号应用于特征表示。
        # 门控机制的作用：通过门控信号动态调整哪些特征应该通过哪些特征应该被抑制。
        # 激活函数的选择：如果 config.hidden_act 是 "sigmoid"，那么激活函数将是 sigmoid
        # 选择 Silu/Swish 作为门控激活函数是因为它结合了 Sigmoid 函数的平滑性和平滑非线性，同时避免了 ReLU 
        # 和 Sigmoid 的一些缺点。
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
# 返回结果：apply_rotary_pos_emb 函数返回的是带有位置信息的嵌入向量 q_embed 和 k_embed，它们包含了原始词嵌入加上位置信息后的结果。
# 旋转位置嵌入：通过旋转位置嵌入（RoPE）技术，将位置信息嵌入到原始的词嵌入中，增强了模型对位置信息的捕捉能力。
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim) # [1, 1024, 128]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim) # [1, 1024, 128]
    # 取负号和重新组合：通过取负号和重新组合嵌入向量的不同部分，使得嵌入向量的不同部分之间产生了相互作用。
    # 应用正弦和余弦变换：通过应用正弦和余弦变换，将旋转后的向量与原始向量结合，从而更好地捕捉位置信息的变化。
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 功能：扩展一个张量的第二维度，使得每个键值头在第二维度上重复 n_rep 次。
# 形状变换：从 (batch, num_key_value_heads, seqlen, head_dim) 变为 
# (batch, num_attention_heads, seqlen, head_dim)。用于q和k,v的头数不同的情况
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这个函数等价于 torch.repeat_interleave(x, dim=1, repeats=n_rep)。
    隐藏状态的形状从 (batch, num_key_value_heads, seqlen, head_dim) 变为 
    (batch, num_attention_heads, seqlen, head_dim)。
    """
    # 获取输入张量的形状
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape # (b,k_heads,s,dk)
    if n_rep == 1: # 这是q和k,v的头数相同的情况
        return hidden_states
    # 新增一个维度,之后扩展
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim) #reshape

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__() # 调用父类的初始化方法
        self.config = config # 配置类实例
        self.layer_idx = layer_idx # 层索引
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        
        self.hidden_size = config.hidden_size # d
        self.num_heads = config.num_attention_heads # q_h
        self.head_dim = self.hidden_size // self.num_heads # dk
        self.num_key_value_heads = config.num_key_value_heads # kv_h
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # 比例
        self.max_position_embeddings = config.max_position_embeddings # p
        self.rope_theta = config.rope_theta # base
        self.is_causal = True # 是否用因果掩码
        self.attention_dropout = config.attention_dropout # dropout
        # 嵌入维度必须能被整除
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # 线性投影
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        #需要注意的是这里的投影维度可能和q的投影维度不同
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # 最后一个线性转换层
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # 旋转位置嵌入层
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim, # dk
            max_position_embeddings=self.max_position_embeddings,# max_position
            base=self.rope_theta, # base
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,# 可选
        position_ids: Optional[torch.LongTensor] = None,# 可选
        past_key_value: Optional[Cache] = None, # 可选参数:缓存
        output_attentions: bool = False,# 是否输出注意力权重
        use_cache: bool = False, # 是否使用缓存
        cache_position: Optional[torch.LongTensor] = None, # 缓存位置
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size() # b,s,d
        # 投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # (b,q_len,q_h,dk)-->(b,q_h,q_len,dk),transpose:换轴(转置)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # (b,k_h,k_len,dk)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2] # k_len
        # 如果上个时间步的key,value表示存在
        if past_key_value is not None: # 如果设置了缓存
            if self.layer_idx is None: # 就必须有layer_idx,不然报错
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # 旋转位置嵌入,传kv_len
        # 键/值序列长度：kv_seq_len 是键和值向量的长度，这是因为key和value向量代表的是相同的序列。
        # 查询序列长度：q_len 是查询向量的长度，这可能不同于键/值向量的长度。
        # 旋转位置嵌入：在计算旋转位置嵌入时，使用键/值序列长度是为了确保位置信息与键和值向量一致。
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # 返回带位置信息的嵌入表示
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # 如果past_key_value is not None
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            # 更新当前的key,value表示,应该要加上之前时间步的表示
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # repeat k/v heads if n_kv_heads < n_heads
        # 如果键值头数量少于查询头数量，则重复键值头以匹配查询头数量。
        key_states = repeat_kv(key_states, self.num_key_value_groups) 
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # (b,q_h,q_len,dk)@(b,k_h,dk,k_len)-->(b,h,q_len,k_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        # 切片,在最后一个维度切出q_len的长度
        if attention_mask is not None:  # 不管长度是多少，我们用切片切出需要的k_len
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            # 相加,一般遮挡的地方是很大的负数
            attn_weights = attn_weights + causal_mask 
        # upcast attention to fp32
        # 在k_len上归一化,得到query序列中每个token对应key中token的一系列权重,这些权重中较大的值表示和当前query中的token
        # 相似度较近,较小的表示离当前query中token较远
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # (b,h,q_len,k_len)@(b,h,v_len,dk)-->(b,h,q_len,dk)
        attn_output = torch.matmul(attn_weights, value_states) 
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # (b,h,q_len,dk)-->(b,q_len,h,dk),之后.contiguous()转为内存连续存储
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (b,q_len,h,dk)-->(b,q_len,d)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # 最后经过线性转换
        attn_output = self.o_proj(attn_output)
        # 如果不输出注意力权重
        if not output_attentions:
            attn_weights = None
        # 返回多头注意力的输出，注意力权重，上个时间步的key_value的缓存
        return attn_output, attn_weights, past_key_value
class Qwen2FlashAttention2(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # 调用父类初始化方法
        # 如果大于2_10,这个是False
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    def forward(
        self, # 当前实例
        hidden_states: torch.Tensor,# 上一层的输入,或者第一次传人的嵌入
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size() # b,s,d
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # (b,h,q_len,dk)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2] # k_len
        # 如果有k_v缓存,就必须有层索引
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            # 应该是加上了past_key_value之前时间步的seq_len
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # 因为可以填充输入，所以绝对序列长度取决于最大位置id。
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        # 获取cos,sin
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        # 获取带位置信息的q,k表示
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # 使用滑动窗口的条件是这么几个,配置中有这个属性,kv长度大于窗口大小等
        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and self.config.use_sliding_window
        )
        # 不支持就报警告信息
        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )
        # 如果设置了past_key_value 
        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads,设置q,k,v的头相同
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # 如果是训练模式,就用dropout,评估模式不用dropout
        dropout_rate = 0.0 if not self.training else self.attention_dropout
        # 在PEFT（Prompt-Encoder Fine-Tuning）中，通常我们将层归一化（LayerNorm）用浮点数32位（float32）进行训练以
        # 保证稳定性。因此，输入的隐藏状态会被默默地转换为浮点数32位。所以，我们需要将它们转换回浮点数16位（float16），
        # 只是为了确保一切按预期工作。
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2) # (b,q_len,h,dk)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )
        # (b,s,d)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Decide whether to use SWA or not by layer index.
        # 超过配置的使用滑动窗口的最大层数的话,设置use_sliding_windows = False
        if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
            use_sliding_windows = False
        # 在序列中至少包含一个填充标记
        if attention_mask is not None:
            batch_size = query_states.shape[0] # b
            # query_states:(len(indices_q),h,dk)
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens 
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens 
            # 不用滑动窗口的情况
            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )
            # attn_output_unpad:(len(indices_q),h,dk),indices_q:一维张量,非填充token索引
            # 这个可以把去填充的张量恢复成去填充之前的张量
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else: # 没有设置attention_mask的情况
            if not use_sliding_windows: # 不用滑动窗口的情况
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else: # 用滑动窗口的情况
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )
        # 返回注意力输出,形状(b,s,h,dk)
        return attn_output
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # b,k_len,h,dk
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
        # 在第一次迭代时，我们需要通过在正确的位置进行切片来正确地重新创建填充掩码
        if kv_seq_len != attention_mask.shape[-1]:
            # 获取掩码的最后一维的大小
            attention_mask_num_tokens = attention_mask.shape[-1]
            # 切出kv_len的大小
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]
        # 非填充的token索引,每一批次中每个序列长度的累积和,表示当前批次中最长序列的长度。
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # index_first_axis = IndexFirstAxis.apply
        # 调用 .apply 方法：IndexFirstAxis.apply(input_tensor, indices) 实际上调用的是 forward 方法。
        # 前向传播：forward 方法执行具体的计算，并返回结果。
        # 反向传播：当计算梯度时，PyTorch 自动调用 backward 方法来计算梯度。
        # 返回的key_layer形状是(len(indices_k),h,dk),去了填充token的嵌入
        # len(indices_k)：表示去除了填充后的有效序列长度。
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        if query_length == kv_seq_len:
            # 索引在第一个轴
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            # 这时给q符相应的值
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        
        elif query_length == 1: 
            max_seqlen_in_batch_q = 1
            # 每一批次中每个序列长度的累积和
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            # 非填充token索引
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1) # (b,h,dk)
        else:
            # The -q_len: 切片操作假设是左填充
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,# (len(indices_q),h,dk),
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
class Qwen2SdpaAttention(Qwen2Attention):
    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 不支持输出注意力权重的回退警告,和回退到父类的forward调用
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        bsz, q_len, _ = hidden_states.size() # b,q_len,d
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # (b,s,d)-->(b,s,h,dk)-->(b,h,s,dk)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2] # k_len
        # 如果使用key_value缓存
        if past_key_value is not None:
            # 设置新的kv_seq_len
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # 旋转位置嵌入
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # 带上位置信息的嵌入
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # 设置缓存的情况
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # 设置k,v和q具有相同的头数
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        causal_mask = attention_mask
        # 如果有传人掩码,切取k_len长度
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            # 设置内存连续状态
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        # q_len等于1时不需要因果掩码,编码器自注意力也不需要因果掩码
        is_causal = True if causal_mask is None and q_len > 1 else False
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        # (b,h,s,dk)-->(b,s,h,dk)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value
# 千问解码器层,继承自nn.Module
class Qwen2DecoderLayer(nn.Module):
    # 构造函数参数:self,当前实例对象,config,Qwen2Config实例对象,layer_idx,层索引
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__() # 调用父类的初始化方法
        self.hidden_size = config.hidden_size # d
        # 如果设置了使用滑动窗口,就必须设置_attn_implementation为"flash_attention_2,不然会警告
        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        # 多头注意力层
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        # 前馈全连接层
        self.mlp = Qwen2MLP(config)
        # 改良的标准化层
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(
        self, # 当前实例对象
        hidden_states: torch.Tensor, # 上次的解码器层输出或者第一次的嵌入层
        # 可选参数:注意力掩码
        attention_mask: Optional[torch.Tensor] = None,
        # 位置ids
        position_ids: Optional[torch.LongTensor] = None,
        # 可选参数:元组(里面元素是Tensor)类型,一般用于推理时,指当前token
        # 之前的所有token表示,这里是作为key,value
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # 可选参数:是否输出注意力权重,布尔类型
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False, # 是否使用缓存
        # 缓存的位置ids,可选,是LongTensor类型
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs, #其他参数,以上是类型注解,规范性的代码就应该有这个
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states #残差前段,如果第一次,是嵌入,否则是上一次解码器的输出
        hidden_states = self.input_layernorm(hidden_states) # 标准化
        # 目标序列自注意力
        # 自注意力输出,自注意力权重,返回的key_value
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, 
            attention_mask=attention_mask, # 注意力掩码
            position_ids=position_ids, # 位置ids
            past_key_value=past_key_value, #上个时间步的key_value
            output_attentions=output_attentions, 
            use_cache=use_cache,
            cache_position=cache_position,
        )
        # 自注意力前后残差连接
        hidden_states = residual + hidden_states
        #重新设定残差连接前段
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states) #标准化
        hidden_states = self.mlp(hidden_states) # 前馈全连接层
        hidden_states = residual + hidden_states # 前馈前后残差
        #解码器输出,放入一个元组中
        outputs = (hidden_states,)
        if output_attentions:  # 如果要输出注意力权重
            outputs += (self_attn_weights,)
        if use_cache: # 如果使用缓存
            outputs += (present_key_value,)
        return outputs # 返回元组
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config # config
    base_model_prefix = "model" # 模型前缀
    supports_gradient_checkpointing = True #是否支持梯度检查
    # 这个属性指定了哪些模块不应该被切分。在 PyTorch 中，当使用模型并行或者分布式训练时，模型的各个层（模块）
    # 可能会被分配到不同的设备上。但是，有些模块内部可能依赖于紧密的交互，不适合被拆分。例如，Qwen2DecoderLayer 
    # 可能包含多个子层（如自注意力层和前馈神经网络层），这些子层需要作为一个整体存在于同一设备上，以确保正确的计算
    # 顺序和数据一致性。
    _no_split_modules = ["Qwen2DecoderLayer"]
    # 这个属性指定了哪些键（keys）在进行设备放置（device placement）时应该被跳过。在 Transformer 
    # 模型中，尤其是对于自回归模型（如解码器），会有历史的键值对（past_key_values）被传递到模型中，
    # 以便在后续的序列预测中使用。这些键值对通常需要保存在内存中，并且在每次调用模型时传递。由于它们是
    # 预先计算的，所以在进行设备分配时，不需要将它们放置到特定的设备上，而是应该保持原样。
    # 在生成任务中，past_key_values 是动态更新的,它们通常仍然存储在 GPU 内存中，以保证高速访问和高效
    # 计算。由于它们在整个序列生成过程中相对稳定，通常不需要在每一步都重新分配到不同的设备上。
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True #是否支持flash_attn_2
    _supports_sdpa = True # 是否支持sdpa
    # 这个属性表明当前模型是否支持使用缓存类（cache class）。在序列模型（如语言模型）中，特别是在生成文本时，
    # 需要重复使用前面生成的键值对（key-value pairs）。支持缓存类意味着模型可以有效地管理和重用这些键值对，
    # 从而提高推理速度和内存使用效率。
    # 哪些模块不应该被切分；
    # 哪些键在进行设备放置时应该被忽略；
    # 是否支持更高效的注意力机制实现；
    # 是否支持缓存机制以提高推理速度。
    _supports_cache_class = True
    #初始化层权重
    # 为什么初始化很重要？
    # 减少训练初期的随机性：合理的初始化可以帮助模型更快地收敛，并且减少训练初期的随机性。
    # 防止梯度消失或爆炸：通过控制权重的大小，可以减轻梯度消失或爆炸的问题。
    # 提高模型性能：良好的初始化策略有助于提高模型的最终性能。
    def _init_weights(self, module):
        # 获取配置中的标准差,这个值通常很小，比如 0.02，目的是为了防止权重过大而导致训练不稳定。
        std = self.config.initializer_range 
        # 如果是线性层实例
        if isinstance(module, nn.Linear):
            # 标准正太分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None: # 如果设置了要bais,初始化为0
                module.bias.data.zero_()
        # 如果是嵌入层的实例
        elif isinstance(module, nn.Embedding):
            # 标准正太分布初始化w
            module.weight.data.normal_(mean=0.0, std=std)
            #如果设置了填充id
            if module.padding_idx is not None: 
                #设置填充id那一行全0
                module.weight.data[module.padding_idx].zero_()
#千问模型,继承自Qwen2PreTrainedModel
class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id # pad_idx
        self.vocab_size = config.vocab_size # v
        #嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 解码器层列表
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 判断究竟用哪种注意力机制
        self._attn_implementation = config._attn_implementation 
        # 标准化层
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False #是否梯度检查
        # 初始化权重
        self.post_init()
    def get_input_embeddings(self): # 获取嵌入
        return self.embed_tokens
    def set_input_embeddings(self, value): #设置嵌入
        self.embed_tokens = value
    # @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self, # 当前模型实例对象
        input_ids: torch.LongTensor = None, # input_ids
        # 注意力掩码,可选(tensor张量)
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.LongTensor] = None, # pids
        # 之前的key_value表示
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # input_embeds
        use_cache: Optional[bool] = None, # 是否使用缓存
        output_attentions: Optional[bool] = None, # 是否输出注意力权重
        # 是否输出最后一个隐藏状态
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # 是否返回字典
        # 缓存的位置ids
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]: 
        # 先看传参,再看配置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 缓存k,v
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果input_ids和inputs_embeds都不是None,抛出错误
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        #如果设置了需要梯度检查,并且是训练模式
        if self.gradient_checkpointing and self.training:
            #如果设置了使用缓存,抛出不兼容警告,并且改变设置use_cache = False
            if use_cache: 
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        use_legacy_cache = False
        #如果设置了use_cache=True,但是past_key_values不是Cache的实例
        if use_cache and not isinstance(past_key_values, Cache):
            # 那就设置使用legacy_cache
            use_legacy_cache = True
            # 获取缓存
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            # 警告:让你把past_key_values设置成一个Cache类的实例
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        # 如果inputs_embeds为None
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) # (b,s,d)
        # 如果没指定缓存的pids
        if cache_position is None:
            # 设置past_seen_tokens 
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            # 设置缓存位置
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        # 如果还没有设定pids
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) # 变成(1,s)
        # 根据掩码参数,ids,缓存的位置,之前的key_values,是否输出注意力权重设定因果掩码
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        # 设定初始隐藏状态为嵌入
        hidden_states = inputs_embeds 
        # decoder layers
        # 所有层的隐藏状态输出
        all_hidden_states = () if output_hidden_states else None
        # 所有的自注意力权重
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None # 初始化缓存
        # 遍历每一个decoder层
        for decoder_layer in self.layers:
            # 把隐藏状态加进all_hidden_states元组,第一次是嵌入,之后是上一层的解码状态
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # 如果设置了梯度检查,并且是训练模式
            if self.gradient_checkpointing and self.training:
                # 那就用梯度检查,这可以减少gpu内存消耗
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__, # 设定的调用函数
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else: 
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            # 重新设定隐藏状态为当前解码器的输出
            hidden_states = layer_outputs[0]
            # 如果使用缓存
            if use_cache:
                # 设定下个缓存
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                # 加入权重元组
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states) # 标准化
        # add hidden states from the last decoder layer
        if output_hidden_states: # 如果设置输出最后的解码器输出,就加入元组
            all_hidden_states += (hidden_states,)
        next_cache = None 
        if use_cache:
            # 设定next_cache
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict: # 如果设置不返回字典形式,就返回元组
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor, # ids
        cache_position: torch.Tensor, # 位置
        past_key_values: Cache, # 缓存
        output_attentions: bool,
    ):
        # 如果要用flash_attention_2
        if self.config._attn_implementation == "flash_attention_2":
            # 如果设定了注意力掩码,并且0在里面
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask # 直接返回
            return None # 不满足上面条件的返回None
        # 当前token之前的token序列
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        # 根据past_key_values是否是StaticCache的实例设定
        using_static_cache = isinstance(past_key_values, StaticCache)
        # 当设定要输出注意力权重时,会回退到使用eager的注意力
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None
        # 获取输入的类型,和设备
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min # 获取当前类型的最小值
        sequence_length = input_tensor.shape[1] # s
        # 如果使用静态缓存
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        # 如果attention_mask是tensor的实例,那就取它最后一维的长度,否则是
        # past_seen_tokens + sequence_length + 1
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        # 如果传人的掩码是4维张量
        if attention_mask is not None and attention_mask.dim() == 4:
            # 这种情况已经设定遮挡位置是很小的负数,所以未遮挡的应该是0
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask # 设定因果掩码
        # 这种是要么没传人掩码,要么不是4维的情况
        else:
            # 先初始化,填充为最大的负数(s_q,s_k)
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            # q_len如果是1,没必要用掩码
            if sequence_length != 1:
                # torch.triu 将矩阵转换为上三角矩阵，对角线以上的元素保留，对角线及以下的元素设置为 0。
                causal_mask = torch.triu(causal_mask, diagonal=1) # 因果掩码
            # cache_position 表示当前的位置信息。
            # 计算一个布尔矩阵，其中每一行对应 cache_position 的值与 target_length 范围内的索引比较。
            # 如果当前位置小于等于目标位置，则保持 min_dtype；否则设置为 0。
            # 这一步确保了掩码仅应用于当前位置之前的位置。
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            # 将掩码扩展到与输入张量 input_tensor 形状相匹配的维度。
            # 新的维度用于适应批量大小（input_tensor.shape[0]）和头的数量（这里为 1）。
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            # 这段代码仅在存在 attention_mask 时执行。如果不存在，则直接使用原始的因果掩码。
            if attention_mask is not None:
                # clone() 方法创建了一个新的张量副本，这样可以在原始张量上进行就地操作（in-place
                # operation），而不影响原始数据。这样做是为了确保掩码修改不会影响到其他地方引用的相同内存地址。
                causal_mask = causal_mask.clone() 
                mask_length = attention_mask.shape[-1] # 序列的长度。
                # 结合因果掩码和注意力掩码：
                # causal_mask[:, :, :, :mask_length] 表示因果掩码的一部分，这部分与 attention_mask 的维度对齐。
                # attention_mask[:, None, None, :] 通过添加两个新维度来调整 attention_mask 的形状，使其与因果掩码
                # 的形状匹配。
                # 将两者逐元素相加，得到的结果 padding_mask 是一个布尔矩阵，其中 True 表示需要被掩码的位置。
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                # 使用 masked_fill() 方法将 causal_mask 中对应 padding_mask 为 True 的位置设置为 min_dtype
                # （通常是最大的负数，确保这些位置在softmax操作后接近于0）。
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask
class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        # 输出层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
    def get_input_embeddings(self):
        return self.model.embed_tokens
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    def get_output_embeddings(self):
        return self.lm_head
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    def set_decoder(self, decoder):
        self.model = decoder
    def get_decoder(self):
        return self.model
    # @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0] # (h,s,d)
        logits = self.lm_head(hidden_states) # (h,s,v)
        logits = logits.float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # 获取模型对前i个token的预测token的置信度分数,因为一般输入序列是[:,:-1]
            # 最后一个不是对应的预测,所以切掉
            shift_logits = logits[..., :-1, :].contiguous()
            # 真实的标签,通常是[:,1:]
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss() # 交叉熵损失
            # 变形成(b*s,v)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1) # 变形成(b*s)
            # Enable model parallelism
            # 把标签传给logits所在设备
            shift_labels = shift_labels.to(shift_logits.device)
            # 计算损失
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict: # 不返回字典的情况
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss, # 损失
            logits=logits, # 模型对下个token的置信度分数
            past_key_values=outputs.past_key_values, # 缓存
            hidden_states=outputs.hidden_states, # 隐藏状态
            attentions=outputs.attentions, # dec_attn
        )
    # 准备生成的输入
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        return model_inputs
    @staticmethod # 静态方法
    def _reorder_cache(past_key_values, beam_idx):
        # 更新cache
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
class Qwen2ForSequenceClassification(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
class Qwen2ForTokenClassification(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
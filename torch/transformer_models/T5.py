# 存储一个T5Model或TFT5Model配置的类。它用于根据指定的参数实例化一个T5模型，
# 定义模型的架构。使用默认值实例化一个配置将产生与T5的google-t5/t5-small架构相似的配置。
class T5Config(PretrainedConfig):
    model_type = "t5" 
    # keys_to_ignore_at_inference = ["past_key_values"] 的设置主要用于简化模型的输出，
    # 使输出更加专注于用户真正关心的结果部分，如生成的文本。在内部计算过程中，模型仍然会使用 
    # past_key_values 来加速解码。如果您需要管理 past_key_values，可以显式地获取并在后续
    # 调用中传递它们。
    keys_to_ignore_at_inference = ["past_key_values"]
    # 定义的属性别名,实际设置的是对应的值
    attribute_map = {"hidden_size": "d_model",
                     "num_attention_heads": "num_heads", 
                     "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=32128, # 词汇表大小。
        d_model=512, # 嵌入维度
        d_kv=64, # dk
        d_ff=2048, # 前馈层中间隐层的大小
        num_layers=6, # 层数
        num_decoder_layers=None, # 解码器中的隐藏层数。如果不设置，将使用与num_layers相同的值。
        num_heads=8, # h
        # 每个注意力层使用的桶的数量。
        relative_attention_num_buckets=32, 
        # 较长序列的桶分离的最大距离。
        relative_attention_max_distance=128,
        dropout_rate=0.1, # 所有丢弃层的比例。
        layer_norm_epsilon=1e-6, # 防止除0错误
        # 在T5模型中，initializer_factor 主要用于控制模型内部参数的初始化尺度。如果设置了不同
        # 于1的值，那么所有参数的初始化都会按这个因子进行缩放。这通常是在某些特殊实验或调试情况下
        # 使用的，例如，当想要测试不同的初始化策略对模型性能的影响时，可以调整这个因子。
        initializer_factor=1.0, 
        feed_forward_proj="relu", # 前馈层激活函数
        is_encoder_decoder=True, # 是否是编码器解码器结构 
        use_cache=True, # 是否使用缓存
        pad_token_id=0, # 填充token id
        eos_token_id=1, # 结束token id
        classifier_dropout=0.0,# 分类器的丢弃比例。
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        # 将 feed_forward_proj 字符串按照 - 符号分割成多个部分，并将结果存储在 act_info 列表中。
        act_info = self.feed_forward_proj.split("-")
        # 将 act_info 列表的最后一部分赋值给 self.dense_act_fn，这通常是实际的激活函数名称
        self.dense_act_fn = act_info[-1]
        # 这行代码检查 act_info 的第一部分是否为 "gated"。如果是，则表明这是一个带有门控机制的激活函数。
        self.is_gated_act = act_info[0] == "gated"
        # 这段代码检查 feed_forward_proj 的格式是否正确。具体来说：如果 act_info 的长度大于1
        # ，并且第一部分不是 "gated"；或者 act_info 的长度大于2；抛出一个 ValueError 异常
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )
        # 这段代码是为了向后兼容而设置的，特别是针对 feed_forward_proj 参数为 "gated-gelu" 的情况。
        # 在早期的实现中，可能直接将 "gated-gelu" 映射为特定的激活函数，而现在为了保持一致性或遵循新
        # 的标准，这段代码将 self.dense_act_fn 设置为 "gelu_new"。
        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
# T5 模型使用的层归一化只进行缩放而不进行偏移，这被称为均方根层归一化（Root Mean
# Square Layer Normalization）。通过这种方式实现 T5 模型特有的层归一化，可以确
# 保在模型训练和推理过程中，归一化操作既高效又准确。同时，通过在必要的时候转换数据类型
# ，可以更好地支持半精度训练，提高模型训练的效率。
class T5LayerNorm(nn.Module):
    # hidden_size 表示隐藏层的大小，也就是层归一化操作将在最后一个轴上进行。
    # eps 是一个很小的正数，添加到方差中以防止除零错误，默认值为 1e-6
    def __init__(self, hidden_size, eps=1e-6): 
        super().__init__()
        # self.weight 是一个可学习的参数，初始化为全1的向量，用于缩放归一化后的输出。
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # self.variance_epsilon 是一个很小的常数，用于避免在计算方根时出现除零的情况。
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        # 尽管在技术上这不是方差，但在上下文中的确可以将其视为一种替代形式的方差，因为它捕捉了输
        # 入数据的分布特性。这种计算方式简化了归一化的过程，同时保留了必要的统计信息。
        # 确保计算的精度足够高，避免由于半精度（如 float16）带来的数值不稳定问题。
        # 对输入张量的每个元素求平方。沿着最后一个维度计算平均值，并保持输出的维度与输入一致。
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 归一化过程是通过将 hidden_states 乘以 torch.rsqrt(variance + self.variance_epsilon) 来实现的：
        # torch.rsqrt 计算了 variance + self.variance_epsilon 的倒数的平方根。
        # 对输入的 hidden_states 进行缩放，这是均方根层归一化的一部分。
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 精度转换：如果权重 self.weight 的类型是 float16 或 bfloat16，则将归一化的隐藏
        # 状态转换回相应精度。
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        # 这段代码实际上是在计算输入张量 hidden_states 的每个样本在最后一个维度上的平方均值，并将其用于归一化
        # 。虽然术语上称之为“方差”，但实际上是指平方均值，这种简化的方法在 T5 模型中用于实现特定类型的层归一化
        # 最后将归一化后的隐藏状态乘以权重 self.weight，得到最终的输出,这个weight可学习
        return self.weight * hidden_states
class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states): # (b,s,d)
        #(b,s,d)-->(b,s,d_ff),线性层转换
        hidden_states = self.wi(hidden_states)
        # 在线性层之后用非线性激活函数可以增强模型的非线性表示能力
        hidden_states = self.act(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # 如果wo.weight的权重是torch.Tensor的实例,并且hidden_states.dtype != wo.weight.dtype
        # 并且wo.weight.dtype != torch.int8
        # 在深度学习中，权重通常是浮点数（如 torch.float32 或 torch.float16），因为它们需要进行精确的数
        # 学运算来更新模型参数。而 int8 类型通常用于量化后的推理过程中，以减少内存消耗和加速计算。量化是一种
        # 技术，通过它可以在一定程度上牺牲精度以换取更快的推理速度和更低的内存占用。
        # 因此，如果 self.wo.weight.dtype 是 torch.int8，那么很可能这个权重已经被量化了，并且模型的设计者
        # 可能已经考虑到了量化带来的所有影响，包括计算的精度和性能优化。在这种情况下，保持 hidden_states 和 
        # self.wo.weight 数据类型的一致性是不必要的，因为量化过程已经处理了所有相关的细节。
        # 这条检查是为了避免对 int8 类型的权重进行不必要的类型转换，因为这种类型转换在量化场景下既不需要也不合适
        # 。如果你正在处理一个未量化的模型，那么确保 hidden_states 和权重的数据类型一致是很重要的，以防止计算
        # 过程中出现精度损失或其他数值稳定性问题。
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8 # 这种情况说明模型已经量化
        ):
            # 转换hidden_states的类型为wo.weight.dtype的类型
            hidden_states = hidden_states.to(self.wo.weight.dtype的)
        # dff-->d
        hidden_states = self.wo(hidden_states)
        return hidden_states
# 这个模块实现了一个带有门控激活函数的全连接层，通常用于 Transformer 模型中的前馈神经网络部分
class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        # hidden_states 通过两个线性层 wi_0 和 wi_1，然后进行特定的激活操作和元素乘法（即门控机制）
        # 。之后应用 dropout 层，并根据特定条件调整 hidden_states 的数据类型，最后再通过另一个线性层 wo 输出
        # 门控,用来控制进self.wo（输出线性层）之前哪些特征应该保留或增强
        # 这种机制可以看作是一个“门”，其中 hidden_gelu 可以视为一个门的开关信号，它决定了 hidden_linear
        # 中的哪些部分应该通过（保留）并传递给下一个层 self.wo。
        hidden_gelu = self.act(self.wi_0(hidden_states)) # d-->d_ff 
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        # dropout
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            # 量化时的类型一般是int8
            and self.wo.weight.dtype != torch.int8 
        ):
            # 转换,以保证类型一致
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        # d_ff-->d
        hidden_states = self.wo(hidden_states)
        return hidden_states
class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        # 如果设置了使用门控,就用门控前馈层
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else: # 否者使用普通的前馈层
            self.DenseReluDense = T5DenseActDense(config)
        # 标准化层
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    def forward(self, hidden_states): #(h,s,d)
        # 前馈前的标准化
        # 前馈层之前的标准化（Layer Normalization）是非常重要的，因为它有助于解决梯度消失或
        # 爆炸问题，并且有助于加速训练过程。
        # 为什么在前馈层之前标准化？
        # 梯度稳定：标准化可以使得每一层的输入具有相同的分布（均值为 0，方差为 1），这有助于防止
        # 梯度消失或爆炸问题，从而使得训练更加稳定。
        # 通过标准化输入，可以加快训练收敛的速度。这是因为每一步的梯度更新都基于一个相对稳定的基
        # 础，从而减少了训练过程中梯度更新的方向性和大小的变化
        # 数值稳定性：标准化可以改善数值稳定性，减少因数值范围过大或过小导致的数值问题。
        forwarded_states = self.layer_norm(hidden_states)
        # 前馈层
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 前馈前后残差
        # 残差连接是通过将层的输入直接添加到其输出上来实现的。这种做法的主要目的是使模型更容易
        # 学习恒等变换
        # 什么是恒等变换？
        # 恒等变换是指模型学习到的函数 f(x)=x，即输出等于输入。在深度网络中，当层数增加时，网络可
        # 能会变得难以训练，因为梯度可能会消失或爆炸。引入残差连接可以帮助模型学习恒等映射，即使网络
        # 很深也能有效训练。
        # 模型如何从残差中学习恒等变换？
        # 当一个层的输入直接与输出相加时，如果该层的学习结果为零，则输出就是输入本身，即实现了恒等变换
        # 。通过这种方式，即使网络非常深，每一层都可以轻松地学会“什么都不做”（即传递输入不变），这实际
        # 上帮助了网络学习复杂的映射关系
        # 标准化和残差连接都是为了帮助深层网络更好地训练，前者通过标准化输入分布来加速收敛和提高数值稳定
        # 性，后者通过学习恒等变换来缓解梯度消失或爆炸问题。
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states
class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        # 是否是只有解码器
        self.is_decoder = config.is_decoder
        # 是否有相对位置截据
        self.has_relative_attention_bias = has_relative_attention_bias
        # 桶数
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        # max_distance
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model # d
        self.key_value_proj_dim = config.d_kv # dk
        self.n_heads = config.num_heads # h
        self.dropout = config.dropout_rate # dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim #h*dk
        # 线性转换,这里inner_dim有可能与d_model大小不同,在修剪头时
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        # 如果需要设置截据
        if self.has_relative_attention_bias:
            # 这里为每个桶编号嵌入一个头数大小的向量
            # relative_attention_bias是一个位置偏差矩阵，表示查询位置和键位置之间的相对位置信息。
            # 这个矩阵可以看作是注意力矩阵的一个修正项，它使得模型能够更好地捕捉注意力矩阵中不同位置之间的相对关系。
            # 这段代码定义了一个嵌入层 self.relative_attention_bias，它将每个桶编号映射到一个大小等于注意力头数量
            # 的向量。具体来说：
            # 桶化相对位置：首先需要将查询位置和键位置之间的相对位置映射到一系列桶中。
            # 嵌入相对位置：然后为每个桶编号嵌入一个向量，这个向量的大小等于注意力头的数量。
            # 应用嵌入：在注意力计算过程中，将这些相对位置的嵌入向量添加到注意力得分上。
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set() # 已经修剪的头索引集合
        self.gradient_checkpointing = False # 梯度检查标记
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 返回当前修剪过的头索引列表,未修剪的头索引对应的嵌入位置
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # 修改q,k,v的权重和bias以匹配修剪过头之后的嵌入,因为权重形状(out_feat,in_feat)
        # dim=0是在输出上去修剪权重,dim=1是在输入上去修剪
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # 更新当前层的头数
        self.n_heads = self.n_heads - len(heads)
        # 根据当前头数计算inner_dim,即q,k,v的总嵌入
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        # 合并所有已经修剪的头的集合
        self.pruned_heads = self.pruned_heads.union(heads)
    # 用于将相对位置转换成一个桶编号，以便在 Transformer 模型的注意力机制中使用。这种方法可以将相对位置映射到一个固定的桶
    # 编号范围内，使得模型在处理不同长度的序列时更具泛化能力。
    @staticmethod # 静态方法,relative_position:(q_len,k_len)
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        # 初始化了一个张量，用来存储每个位置的桶编号。初始时所有的桶编号都设为 0。
        relative_buckets = 0 # 初始化相对桶编号
        # 如果 bidirectional 参数为 True，表示这是一个双向的相对位置。在这种情况下，我们将
        # 桶编号的数量减半。这是因为在双向的情况下，正向和反向的位置信息需要分开处理，所以桶的数量要减半。
        if bidirectional:
            num_buckets //= 2 # 注意:双向时,桶的编号减半
            # 相对位置<=0的桶编号不变,相对位置>0的,桶编号+桶的数量(这个是桶数减半后的)。
            # .to(torch.long) 将布尔张量转换为整数张量，其中 True 变为 1，False 变为 0。
            # 相对位置>0的在布尔张量中是1,这时会变成num_buckets
            # 这样做的目的是为了区分正向和反向的位置信息。在双向的情况下，正向的桶的编号会加
            # 上桶的数量，而反向的桶的编号保持不变。
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            # 这一行的作用是获取相对位置的绝对值。因为在双向的情况下，我们需要关注的是相对位置的大小而不
            # 是方向。所以，这里取绝对值是为了去除方向的影响，只保留大小信息。
            # 如何理解“桶”？
            # 在相对位置编码中，“桶”是一个抽象的概念，用来对不同的相对位置进行分类或分组。具体来说：
            # 桶编号：每个桶都有一个编号，用来表示一组相似的相对位置。
            # 桶的数量：num_buckets 表示总的桶的数量。在这个例子中，如果 bidirectional 为 True，则桶的数量会减半。
            # 桶的划分：根据相对位置的大小，将不同的位置映射到不同的桶中。这样做是为了简化模型的学习过程，使得模型
            # 能够更有效地利用相对位置信息来进行预测。
            relative_position = torch.abs(relative_position) 
        else:
            # 处理单向情况,单向时桶编号未减半
            # 正向的统一分配了一个桶编号0,负向的取了位置的相反数
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # 现在 relative_position 的范围是 [0, 正无穷)
        # max_exact是位置的一个阈值,用于区分小的相对位置和大的相对位置。当 relative_position小于等于max_exact 
        # 时，我们会使用线性映射；否则，我们会使用对数映射来计算桶编号。
        max_exact = num_buckets // 2
        # 判断相对位置是否小于位置阈值,表示哪些key上的token的相对位置小于 max_exact。这将用于后续选择合适的桶编号。
        is_small = relative_position < max_exact
        # 对于那些相对位置大于 max_exact 的位置，我们使用对数映射来分配桶编号
        relative_position_if_large = max_exact + (
            # 首先，将相对位置除以 max_exact 并取自然对数，得到一个较小的值。这个时候<max_exact
            # 的位置的地方会<0
            torch.log(relative_position.float() / max_exact)
            # 然后，将这个较小的值除以 math.log(max_distance / max_exact)，得到一个标准化的比例。
            / math.log(max_distance / max_exact)
            # 接着，将这个比例乘以剩下的桶数量 num_buckets - max_exact，从而得到一个较大的桶编号。
            * (num_buckets - max_exact) 
        ).to(torch.long)
        # 为了防止桶编号超过最大值 num_buckets - 1。使用 torch.min 确保所有桶编号都不会超过 num_buckets - 1。
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        # 这里使用 torch.where 来选择合适的桶编号：
        # 当 is_small 为 True 时（即相对位置小于 位置阈值max_exact），使用 relative_position 作为桶编号。
        # 当 is_small 为 False 时（即相对位置大于 max_exact），使用 relative_position_if_large 作为桶编号。
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        # 小相对位置的映射：对于相对位置小于 max_exact 的情况，我们直接使用相对位置作为桶编号。
        # 大相对位置的映射：对于相对位置大于 max_exact 的情况，我们使用对数映射来减少桶编号的范
        # 围，避免桶编号过大，同时保持位置信息的连续性。
        # 这种做法的好处在于：小相对位置：保留了位置的精细信息。大相对位置：通过对数压缩，减少了位置
        # 信息的维度，使得模型更容易学习长距离依赖关系。
        return relative_buckets
    def compute_bias(self, query_length, key_length, device=None):
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 表示行上的query中每个token的位置
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 表示列上的key中每个token的位置
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 假设query_length=8,key_length=16
        # 相对位置,对于第一行来说因为query的token位置是0,所以key中每个token对它的相对位置会从0--15
        # 而之后第二行,因为query的token位置是1,所以key中每个token对它的相对位置会从-1--14
        # 之后情况类似,在query和token位置位置相同,也就是如果是方阵的对角线方向,这条线上的相对位置会是0
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder), # 如果不是只有解码器,就设置为True
            num_buckets=self.relative_attention_num_buckets, # 桶的数量
            # relative_attention_max_distance 是一个整数，表示在相对位置编码中考虑的最大距离。
            # 这个参数用来定义在对数缩放部分，最大相对位置的距离,最大距离：在对数缩放部分，相对位置
            # 大于 128 的将会被压缩到桶编号的某个范围内。
            max_distance=self.relative_attention_max_distance, 
        )
        
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values
    def forward(
        self,
        hidden_states, # 输入的隐藏状态张量，形状为 (batch_size, seq_length, dim)。
        mask=None, # 掩码张量，用于指示哪些位置应该被忽略。
        key_value_states=None, # 用于跨注意的情况，当 key_value_states 不为空时，表示这是跨注意。
        position_bias=None, # 位置偏差张量，用于相对位置编码。
        past_key_value=None,# 过去的键和值状态，用于解码时的缓存。
        layer_head_mask=None, # 层头部的掩码，用于遮盖某些头。
        query_length=None,# 查询序列的长度，用于计算真实的序列长度。
        use_cache=False, # 是否使用缓存，通常用于加速解码过程。
        output_attentions=False,# 是否返回注意力权重。
    ):
        # 获取批次大小和序列长度
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length # 初始化
        # 如果past_key_value不是None,继续判断len(past_key_value),不是2会报错
        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            # 如果query_len没设置,就使用真实序列长度,否则使用query_length 
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length
        # 如果key_value_states is None,是自注意力,否则用key_value_states的轴1上的长度
        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]
        def shape(states):# 将隐藏状态重塑为多头形式。
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        def unshape(states): # 将多头形式的隐藏状态恢复为原始形式。
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        # project 函数：根据输入状态投影到键/查询状态。
        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            if key_value_states is None: # 自注意力
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None: # 跨注意力，但是没缓存
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))
            # 有缓存
            if past_key_value is not None:
                # 自注意力
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    # 在序列长度上拼接缓存和hidden_states
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                # 跨注意力
                # 在这种情况下，我们需要检查过去的键值状态的序列长度是否与提供的 key_value_states 的序列长度相同。
                # 如果长度不相同，那么需要重新计算键值状态。
                # 使用 proj_layer 投影 key_value_states，并使用 shape 函数将其重塑为多头形式。
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # Prefix Tuning 是一种技术，其中模型的前缀部分（通常是位置嵌入或其他固定部分）可以在训练
                    # 过程中调整，以适应不同的任务或输入。在这种情况下，past_key_value 可能包含了前缀部分的信息
                    # ，其长度可能与当前 key_value_states 的长度不同。如果 past_key_value 包含了一些额外的前
                    # 缀信息，那么它的序列长度可能会比当前的 key_value_states 更长
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else: # 跨注意力,上述长度相同
                    # 直接使用 past_key_value。
                    hidden_states = past_key_value
            return hidden_states
        query_states = shape(self.q(hidden_states))  # (b,h,s,dk)
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )
        scores = torch.matmul( # (b,h,q_len,k_len)
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        
        if position_bias is None:
            # 如果没有这个属性
            if not self.has_relative_attention_bias:
                # 初始化一个形状(1,h,q_len,k_len)的全0张量
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                # 设置计算梯度
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                # (1, num_heads, query_length, key_length)
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)
            # 如果有缓存的k,v表示
            if past_key_value is not None:
                # 只需要当前隐藏序列长度的位置表示
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]
            # 如果有mask,就加,这时的mask遮挡位置是很大的负数
            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
        # 如果有已经修剪的头
        if self.pruned_heads:
            # 初始化所有头的掩码为1
            mask = torch.ones(position_bias.shape[1])
            # 设置已经修剪的头的掩码为0
            mask[list(self.pruned_heads)] = 0
            # 选取没被修剪的头的位置偏差,这时过滤掉了修剪过的头的位置偏差
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            # 如果没有已经修剪的头,就直接赋值
            position_bias_masked = position_bias
        # 给注意力权重加上位置遮挡,这时掩码中很大的负数的位置在归一化后会是0
        scores += position_bias_masked
        # 在q_len上归一化(b,h,q_len,k_len)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  
        attn_weights = nn.functional.dropout( # dropout
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)
        
        # 如果有头部掩码,遮盖某些头部
        # 头部掩码（Layer Head Mask）
        # 另一方面，头部掩码（layer_head_mask）通常用于临时屏蔽某些头部，而不是永久移除它们。
        # 这种掩码可以在训练或推理过程中动态地改变，而不改变模型结构本身。头部掩码可以在以下场景中使用：
        # 模型训练期间的实验：在训练过程中，可能需要测试不同的头部组合对模型性能的影响。使用头部掩码可以
        # 在不修改模型结构的情况下，临时关闭某些头部。这些头部在当前计算中被屏蔽，但在模型结构上仍然存在。
        # 推理过程中的优化：在推理过程中，可能希望减少计算量，通过暂时关闭某些头部来提高效率
        # 动态调整：头部掩码可以在训练或推理过程中动态调整，以测试不同的头部组合。
        # 灵活性：头部掩码提供了更大的灵活性，可以在不修改模型结构的情况下，关闭或激活特定头部。
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        # 用注意力权重给value_states加权
        # (b,h,q_len,k_len)@(b,h,v_len,dk)-->(b,h,q_len,dk)
        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        # 线性转换到d,因为有些头可能被修剪
        attn_output = self.o(attn_output)
        # 在 Transformer 模型的解码器部分，通常需要保存过去的时间步的键值状态（key-value states），以便在解码过程中重用这些信息
        # 。这是因为解码器在生成每个新的时间步时，需要考虑之前生成的时间步的信息。
        # 如果是解码器：当 self.is_decoder 为 True 时，表示当前层是用于解码器部分的。在这种情况下，如果 use_cache 为
        # True，则会保存当前的时间步的键值状态，以便在后续的时间步中使用。
        # 如果不是解码器：当 self.is_decoder 为 False 时，表示当前层是用于编码器部分的。在这种情况下，通常不需要保存键值
        # 状态，因为编码器处理的是整个输入序列，而不是逐个时间步生成输出。
        # 在训练过程中，由于每个序列的各个时间步是同步处理的，因此通常不需要使用缓存（use_cache 为 False）。而在解码过程中，
        # 每个时间步的生成是逐步进行的，因此需要使用缓存（use_cache 为 True），以便在每一步生成时利用之前时间步的信息。
        # 根据条件设置是否使用缓存
        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        # 输出为注意力输出+缓存的key_value_state元组+权重矩阵位置编码
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        if output_attentions: # 如果设置要输出注意力权重,就输出
            outputs = outputs + (attn_weights,)
        return outputs
# T5自注意力
class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 自注意力
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon) # 标准化层
        self.dropout = nn.Dropout(config.dropout_rate)
    def forward(
        self,
        hidden_states, 
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states) # 标准化
        # 输出为注意力输出+缓存的key_value_state元组+权重矩阵位置编码
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        #自注意力前后残差
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:] 
        return outputs
class T5LayerCrossAttention(nn.Module): # 跨注意力
    def __init__(self, config):
        super().__init__()
        # 跨注意力
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states) # 标准化
        # 跨注意力输出
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0]) # 残差连接
        outputs = (layer_output,) + attention_output[1:]  
        return outputs
class T5Block(nn.Module): # 相当于一个普通transformer中的编码器层或者解码器层
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder 
        self.layer = nn.ModuleList()
        # 添加自注意力
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder: # 如果设置了需要解码器,就追加交叉注意力
            self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config)) # 前馈层
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        # 如果past_key_value存在
        if past_key_value is not None:
            # 如果不是解码器
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            # 如果encoder_hidden_states是None的话，期望的num_past_key_values = 2 ,否则是4
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4
            # 如果不对,报错
            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (key / value) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )
            # 自注意力的past_key_value
            self_attn_past_key_value = past_key_value[:2]
            # 跨注意力的past_key_value
            cross_attn_past_key_value = past_key_value[2:]
        else: # 如果没缓存
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        # 自注意力的输出
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获取隐藏状态输出和缓存
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        # 注意力权重位置嵌入和注意力权重
        attention_outputs = self_attention_outputs[2:]  
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        # 进行交叉注意力的条件
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # 如果之前的present_key_value_state存在
            if present_key_value_state is not None:
                # q_len是缓存中的key的长度
                query_length = present_key_value_state[0].shape[2] 
            else: # 如果缓存不存在,q_len为None
                query_length = None
            # 交叉注意力
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]  # 解码器输出

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # 合并自注意力和交叉注意力key_value_state
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # 加上注意力矩阵位置嵌入和交叉注意力的注意力权重
            attention_outputs = attention_outputs + cross_attention_outputs[2:]
        #  前馈层
        hidden_states = self.layer[-1](hidden_states)
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if use_cache: # 如果使用缓存
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs
        return outputs  
class T5ClassificationHead(nn.Module):
    # 用于句子级别分类任务的头部
    def __init__(self, config: T5Config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)
    # 在 T5ClassificationHead 中，没有显式的池化操作。特征提取通常是通过选择序列中的特定位置
    # （如 <extra_id_0>或 <pad>）来完成的。这种方法假设特定位置的隐藏状态已经包含了足够的信息
    # 来代表整个句子或段落，并且适用于句级分类任务。
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states) # 对输入的隐藏状态应用 dropout，增加模型的泛化能力
        hidden_states = self.dense(hidden_states) # 将隐藏状态的维度从 d_model 投影回 d_model
        hidden_states = torch.tanh(hidden_states) # 使用 Tanh 激活函数对投影后的隐藏状态进行非线性变换。
        hidden_states = self.dropout(hidden_states) # dropout
        # 通过另一个线性层将隐藏状态的维度从 d_model 映射到类别数目 num_labels。
        hidden_states = self.out_proj(hidden_states) 
        return hidden_states
class T5PreTrainedModel(PreTrainedModel):
    config_class = T5Config # config_class 指定了配置类 T5Config。
    # 加载 TF 权重：load_tf_weights 方法用于从 TensorFlow 模型加载权重。
    load_tf_weights = load_tf_weights_in_t5
    # 基模型前缀：base_model_prefix 用于指定模型前缀，便于从字典中加载和保存权重。
    base_model_prefix = "transformer"
    # 是否可并行化：is_parallelizable 表明模型支持并行化。
    is_parallelizable = True
    # 是否支持梯度检查点：supports_gradient_checkpointing 表明模型支持梯度检查点。
    supports_gradient_checkpointing = True
    # 不可分割的模块：_no_split_modules 列出了不应分割的模块，例如 T5Block。
    _no_split_modules = ["T5Block"]
    # 保持 FP32 的模块：_keep_in_fp32_modules 列出了需要保持在 FP32 精度下的模块。
    _keep_in_fp32_modules = ["wo"]
    # 虚拟输入：dummy_inputs 提供了用于调试或测试的虚拟输入数据。
    @property 
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs
    # 初始化权重：_init_weights 方法用于初始化模型的各种参数。
    # 初始化权重方法 _init_weights
    # 这个方法主要用于初始化模型的权重，确保它们按照一定的分布初始化。根据不同的模块类型，使用不同的初始化策略：
    def _init_weights(self, module):
        factor = self.config.initializer_factor  
        # T5LayerNorm：将权重初始化为 factor * 1.0。
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        # T5Model, T5ForConditionalGeneration, T5EncoderModel, T5ForQuestionAnswering：
        # 使用正态分布初始化共享权重，并且如果 tie_word_embeddings 未启用，则初始化语言模型头部的权重。
        elif isinstance(
            module,
            (T5Model, T5ForConditionalGeneration, T5EncoderModel, T5ForQuestionAnswering),
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "qa_outputs"):
                module.qa_outputs.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                module.qa_outputs.bias.data.zero_()
        # T5ForTokenClassification：初始化分类器的权重。
        elif isinstance(module, T5ForTokenClassification):
            if hasattr(module, "classifier"):
                module.classifier.weight.data.normal_(mean=0.0, std=factor * 1.0)
                module.classifier.bias.data.zero_()
        # T5ClassificationHead：初始化密集层和输出投影层的权重。
        elif isinstance(module, T5ClassificationHead):
            module.dense.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.dense, "bias") and module.dense.bias is not None:
                module.dense.bias.data.zero_()
            module.out_proj.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.out_proj, "bias") and module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        # T5DenseActDense 和 T5DenseGatedActDense：初始化 FF 层的权重。
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        # T5Attention：初始化注意力层的权重，包括查询、键、值和输出层的权重。
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
    # 右移输入 ID：_shift_right 方法用于将输入 ID 右移，准备用于解码器的输入。
    # 右移输入 ID 方法 _shift_right
    # 这个方法用于将输入 ID 右移一位，以准备用于解码器的输入。具体做法是：
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        # 检查解码器起始标记 ID 是否已定义：如果未定义，抛出异常。
        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            # 创建右移后的输入 ID：将输入 ID 的最后一个维度向前移动一位，并在起始位置插入解码器起始标记 ID。
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            # 替换无效值：将输入 ID 中的 -100 值替换为填充标记 ID。
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
# @add_start_docstrings(
#     "The bare T5 Model transformer outputting raw hidden-states without any specific head on top.",
#     T5_START_DOCSTRING,
# )
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'encoder.block.0':"
            " 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5Model

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5Model.from_pretrained("google-t5/t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
        >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5EncoderModel(T5PreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight"]
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5EncoderModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5EncoderModel.from_pretrained("google-t5/t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


@add_start_docstrings(
    """
    T5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    T5_START_DOCSTRING,
)
class T5ForSequenceClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.transformer = T5Model(config)
        self.classification_head = T5ClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        self.model_parallel = False

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        # Copied from models.bart.modeling_bart.BartModel.forward different to other models, T5 automatically creates
        # decoder_input_ids from input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        eos_mask = input_ids.eq(self.config.eos_token_id).to(sequence_output.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


@add_start_docstrings(
    """
    T5 Encoder Model with a token classification head on top (a linear layer on top of the hidden-states output)
    e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    T5_START_DOCSTRING,
)
class T5ForTokenClassification(T5PreTrainedModel):
    _tied_weights_keys = ["transformer.encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = T5EncoderModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, outputs[2:-1])
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    T5 Model with a span classification head on top for extractive question-answering tasks like SQuAD (linear layers
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    T5_START_DOCSTRING,
)
class T5ForQuestionAnswering(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.num_labels = config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.model_parallel = False

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqQuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if start_positions is not None and end_positions is not None:
            use_cache = False

        # Copied from models.bart.modeling_bart.BartModel.forward
        #   different to other models, T5 automatically creates decoder_input_ids from
        #   input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + decoder_outputs[1:] + encoder_outputs
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

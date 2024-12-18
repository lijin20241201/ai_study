class Gemma2Config(PretrainedConfig):
    # 指定了该配置的模型类型为 "gemma2"，这是模型的标识符。
    model_type = "gemma2"
    # 在推理过程中，忽略 past_key_values 这个键。通常这个参数用于缓存上一轮的计算结果，避免重复计算。
    # 具体效果包括：
    # 不缓存过去的键和值：
    # past_key_values 是用来缓存计算的结果以加速推理的。但在某些情况下（比如你不需要复用之前的状态），可以选择忽
    # 略这个键，避免模型在推理时存储和使用这些历史信息。
    # 推理过程中不使用过去的计算结果
    # 如果你设置了忽略 past_key_values，模型在每次推理时都不会使用之前的计算结果，而是重新计算当前的注意力信息。
    # 这样做可能会导致推理速度变慢，但也可以保证每次生成时都不依赖于历史缓存。
    # 推理行为的影响：
    # 对于一些不需要进行历史缓存的任务，例如一些不涉及上下文连续生成的任务，忽略 past_key_values 可以使得推理过
    # 程更加简洁。通常，忽略 past_key_values 可能意味着每次都从头开始计算注意力，而不是复用历史信息。
    # 什么时候使用 keys_to_ignore_at_inference：
    # 资源限制：如果在推理时内存有限，缓存可能会导致内存占用过多，使用 keys_to_ignore_at_inference 来避免缓存。
    # 模型特性：在某些模型或者任务中，可能不希望复用过去的计算结果，例如对于短文本或者一些非自回归的任务（如分类任务）
    # ，可以禁用缓存。
    # 实验或调试：在一些特殊的实验或者调试情况下，可能希望观察模型每次推理的独立性，忽略过去的状态。
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        vocab_size=256000, # 词汇表大小，表示模型所能处理的不同词汇的数量。
        hidden_size=3072,# 隐藏层的大小（或称隐层维度）
        intermediate_size=24576,# 用于在前馈神经网络（Feed Forward Network）中增加模型的容量。
        num_hidden_layers=28,# 模型中 Transformer 编码器堆叠的层数
        num_attention_heads=16, # 注意力机制中的头数。
        num_key_value_heads=16, # 每个注意力头中的键（key）和值（value）头的数量
        head_dim=256,# 每个注意力头的维度。
        hidden_activation="gelu_pytorch_tanh",# 隐藏层的激活函数。
        max_position_embeddings=8192, # 最大位置嵌入数，表示模型能够处理的最大输入序列长度。
        initializer_range=0.02, # 权重初始化的范围，决定了模型的权重初始化时值的范围。
        rms_norm_eps=1e-6, # 用来避免数值计算中的除零错误。
        use_cache=True, # 是否使用缓存，通常用于生成任务（如文本生成），缓存可以加速推理。
        pad_token_id=0, # 填充标记的 ID
        eos_token_id=1, # 句子的结束标记（End Of Sentence）的 token ID。
        bos_token_id=2, # 句子的开始标记（Beginning Of Sentence）的 token ID。
        # 是否共享词嵌入（embedding）。当为 True 时，模型会共享词嵌入矩阵和输出层的权重，
        # 通常用来减少模型参数的数量。
        tie_word_embeddings=True,
        # 位置编码的 theta 参数，通常在 RoPE（Rotary Position Embeddings）中使用，调整不同位置之间的关系。
        rope_theta=10000.0,
        attention_bias=False, # 是否在注意力机制中使用偏置。False 表示不使用。
        attention_dropout=0.0, # 注意力层的 dropout 比例，0.0 表示不使用 dropout。
        # 对最终输出 logits 进行软限制（soft capping）。常用于模型输出值的范围控制。
        final_logit_softcapping=30.0,
        # 对注意力 logits 进行软限制（soft capping）。用于控制注意力的输出范围。
        attn_logit_softcapping=50.0,
        # 查询向量在进行注意力计算之前的标量，通常用于调整模型的输入权重。
        query_pre_attn_scalar=224,
        # 滑动窗口的大小，用于控制模型在处理超长输入时分割输入的方式。
        sliding_window=4096,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_activation = hidden_activation
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attn_logit_softcapping = attn_logit_softcapping
        # 调用了父类（PretrainedConfig）的 __init__ 方法，将一些公共的参数传递给父类进行初始化。
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.final_logit_softcapping = final_logit_softcapping
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        # 指定了缓存的实现方式为 "hybrid"，可能表示一种混合缓存策略。
        self.cache_implementation = "hybrid"

from transformers.onnx import OnnxConfig
class ErnieConfig(PretrainedConfig):
    model_type = "ernie" # 模型类型
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072, # 前馈中间层
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        task_type_vocab_size=3,
        use_task_id=False,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute", # 绝对
        use_cache=True,
        classifier_dropout=None,# 分类器dropout
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.task_type_vocab_size = task_type_vocab_size
        self.use_task_id = use_task_id
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
from typing import Mapping
class ErnieOnnxConfig(OnnxConfig):
    # @property 装饰器并不是将方法“变成了一个实例属性”，而是提供了一种方式，使得类的实例可以像访问数据
    # 属性一样来调用被装饰的方法，并且不需要在调用时加上括号（即不需要显式地作为方法来调用）。这种机制使
    # 得属性的访问更加直观和方便，同时保留了方法的灵活性，可以在属性被访问时执行一些计算或逻辑判断。
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # choice 是一个在处理“multiple-choice”（多选）任务时，用于表示不同选项的维度名称
        # 多选任务通常指的是需要从一组给定的选项中选择一个或多个正确答案的任务，比如阅读理解中的
        # 多项选择题。对于多选任务，模型的输入通常包括文本数据（如单词或标记的ID），以及一个或
        # 多个额外的维度来指示如何处理这些数据。
        # 在多选任务中，模型可能需要同时处理多个选项（例如，对于每个问题有多个候选答案），因此这个维度是动态的
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # OrderedDict 的构造函数可以接受一个列表（或者更准确地说，是一个列表的列表，或者更常见的是元
        # 组的列表），其中每个内部列表（或元组）包含两个元素：键和值。
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
                ("task_type_ids", dynamic_axis),
            ]
        )

class ErnieEmbeddings(nn.Module): # 嵌入类
    def __init__(self, config):
        super().__init__()
        # 词嵌入,位置嵌入,token类型嵌入
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.use_task_id = config.use_task_id
        # 任务类型嵌入
        if config.use_task_id:
            self.task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)
        # 使用nn.LayerNorm对嵌入后的表示进行归一化处理，有助于稳定训练过程。
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids是一个张量（tensor），其形状为(1, len position emb)，其中len position emb指的是位置嵌入的最大长度。
        # 这个张量在内存中是连续的（contiguous），并且在模型被序列化（比如保存为文件）时会被导出。这意味着position_ids是一
        # 个静态的、不会随模型训练而改变的数据结构，用于为每个位置分配一个唯一的标识符。
        # 这行代码从配置（config）中获取position_embedding_type的值，如果配置中没有这个值，则默认为"absolute"。这表示模型
        # 将使用绝对位置嵌入，而不是相对位置嵌入或其他类型的位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # register_buffer是PyTorch中nn.Module类的一个方法，用于注册一个不参与模型训练的参数（即，不需要梯度的参
        # 数）。这些参数在模型的前向传播中可能会被使用，但不会通过反向传播进行更新。register_buffer非常适合存储那
        # 些在整个训练过程中保持不变的数据，比如嵌入层（embeddings）的索引、常量权重等。
        # 这行代码创建了一个position_ids张量，其值从0到config.max_position_embeddings - 1，然后将其扩展为
        # (1, config.max_position_embeddings)的形状。这个张量被注册为一个buffer，persistent=False意味
        # 着这个buffer在模型被保存时不会被保存（但在当前的训练会话中是可用的）。然而，需要注意的是，在PyTorch的较新
        # 版本中，persistent参数已经不再被register_buffer方法接受，因为它默认就是非持久的（即，不会被保存）
        # 因为 -1 在 expand 方法中用作占位符，表示该维度的大小应保持不变。
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 这行代码创建了一个全零的token_type_ids张量，其形状与position_ids相同，用于存储标记（token）的类型信息（如果有的
        # 话）。这个张量也被注册为一个buffer
        # 总的来说，这段代码初始化了两个不参与模型训练的参数：position_ids和token_type_ids，它们分别用于表示输入序列中每
        # 个位置的位置信息和类型信息。
        # 这个属性可以通过self来访问和调用，就像访问模型的参数（parameters）一样
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        task_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        # 在某些场景中，position_ids是模型理解和处理序列数据的关键部分，而token_type_ids可能用于提供额外的上下文信息（如区分不同的输入
        # 段）。因此，模型可能更加依赖于position_ids的存在，而对token_type_ids的依赖性较低。
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        # 如果token_type_ids为None，这通常发生在自动生成时，将token_type_ids设置为已注册的缓冲区（buffer），这有助于
        # 用户在追踪模型时无需传递token_type_ids，如果对象有这个属性（即token_type_ids缓冲区已经被注册） 
        # 尽管在大多数情况下，如果你在构造函数中调用了self.register_buffer("token_type_ids", ...)，那么self.token_type_ids
        # 应该总是存在的。但是，编程中总是存在意外情况，比如继承自这个类的子类可能覆盖了构造函数而没有调用基类的构造函数，或者在某些复
        # 杂的初始化逻辑中，register_buffer的调用被意外地跳过了。因此，这个检查提供了一种额外的安全保障。
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"): # 对象有某个属性
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # add `task_type_id` for ERNIE model
        if self.use_task_id:
            if task_type_ids is None:
                task_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            task_type_embeddings = self.task_type_embeddings(task_type_ids)
            embeddings += task_type_embeddings
        # 层标准化和dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Ernie
class ErnieSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        # config：一个配置对象，包含了模型的各种参数，如隐藏层大小,注意力头数,注意力概率丢弃率
        super().__init__()
        # 首先检查隐藏层大小是否是注意力头数的整数倍，如果不是，则抛出 ValueError。
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 隐藏层大小（{config.hidden_size}）不是注意力头数（{config.num_attention_heads}）的整数倍
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads # h
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # dk
        self.all_head_size = self.num_attention_heads * self.attention_head_size # h*dk
        # 这些线性投影层（self.query, self.key, self.value）的参数是独立学习的，因此它们产生的表示（即Q、
        # K、V矩阵）在投影后的空间中仍然是不同的。
        self.query = nn.Linear(config.hidden_size, self.all_head_size) # 线性投影
        self.key = nn.Linear(config.hidden_size, self.all_head_size) # 用于将输入映射到查询（Q）、键（K）、值（V）空间
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 位置嵌入的类型，默认为 None，此时会从 config 中获取
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果使用相对位置嵌入（"relative_key" 或 "relative_key_query"），则还需要一个 distance_embedding 层来编码位置信息。
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.is_decoder = config.is_decoder # 指示当前模块是否作为解码器的一部分。
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 通过 view 方法调整张量的形状，以便在每个序列元素上分割出不同的注意力头(b,s,h,dk)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3) # 使用 permute 方法重新排列维度,(b,h,s,dk)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states) # (b,s,d)
        # 如果这是作为交叉注意力模块实例化的，那么键（keys）和值（values）来自一个编码器；注意力掩码
        # （attention mask）需要是这样设置的，即编码器的填充token（padding tokens）不会被关注到。
        # 判断是否是交叉注意力
        is_cross_attention = encoder_hidden_states is not None 
        # 当is_cross_attention为True且past_key_value不为None时，解码器会重用之前时间步中编码器输出的键（key）和值（value）表示
        # 。这是因为编码器的输出在整个生成过程中是固定的，不需要在每个时间步重新计算。重用这些表示可以节省计算资源并提高生成速度。
        if is_cross_attention and past_key_value is not None: 
            key_layer = past_key_value[0]  
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention: 
            # 如果没有past_key_value（即第一次进行跨注意力计算或past_key_value未被保存），则解码器会
            # 把编码器输出转换成键和值表示。
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states)) # (b,h,s,dk)
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None: 
            # 在自注意力情况下，past_key_value用于保存之前时间步中解码器自身输出的键和值表示。这是因为解码
            # 器在生成过程中需要关注之前已经生成的token。通过重用这些表示，解码器可以增量地生成输出序列
            key_layer = self.transpose_for_scores(self.key(hidden_states)) # (b,h,s,dk)
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            # 在序列长度维度上合并
            # 在每个时间步，我们仍然需要计算新的注意力权重，但通过使用 past_key_value，我们可以避免重复计算之
            # 前时间步已经计算过的部分，从而实现更高效的生成过程
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2) # (b,h,s,dk)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else: 
            # 如果没有past_key_value（例如，在序列的起始位置或past_key_value未被正确传递），
            # 对自身序列做自注意力前的变形表示
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer) # (b,h,s,dk)
        # 当past_key_value is not None时,use_cache为True
        use_cache = past_key_value is not None 
        if self.is_decoder:
            # 这段描述是关于在Transformer模型中，特别是编码器和解码器架构中，如何处理和重用注意力机制中的键（
            # key）和值（value）状态的。这里涉及到交叉注意力（cross-attention）、单向自注意力（如解码器中的
            # 自注意力）和双向自注意力（如编码器中的自注意力）三种情况。
            # 如果使用交叉注意力，那么会保存一个包含所有交叉注意力键和值状态的元组
            # 在后续的交叉注意力层调用中，可以重用这些保存的交叉注意力键和值状态（即第一个“if”情况）。这样做可以
            # 减少计算量，因为不需要在每个时间步都重新计算这些状态。
            # 单向自注意力（如解码器中的自注意力）
            # 在单向自注意力的情况下（如解码器在生成文本时使用的自注意力），会保存一个包含所有之前解码器键和值状态
            # 的元组,在后续的单向自注意力层调用中，可以将之前保存的解码器键和值状态与当前投影得到的键和值状态进行拼
            # 接（即第三个“elif”情况）。这样做是为了让解码器在生成下一个token时能够考虑到之前已经生成的token的信息。
            # 编码器双向自注意力:
            # 对于编码器的双向自注意力，past_key_value 总是为 None
            # 这是因为编码器的自注意力是双向的，即每个token在生成其表示时都可以关注到序列中的其他所有token。因此，在
            # 编码过程中，不需要保存之前的键和值状态来重用，因为每个token的注意力计算都是独立的，并且依赖于整个输入序列。
            # 这段描述解释了在不同类型的注意力机制中，如何处理和重用键和值状态以优化计算效率和模型性能。在交叉注意力和解码器的
            # 单向自注意力中，通过保存和重用之前的键和值状态，可以减少不必要的计算；而在编码器的双向自注意力中，由于每个token
            # 的注意力计算都是独立的，因此不需要保存和重用之前的键和值状
            # 如果是解码器的跨注意力和解码器目标序列,就保存之前的key_layer, value_layer
            past_key_value = (key_layer, value_layer)
        # (b,h,s_q,dk)@(b,h,s_k,dk).T-->(b,h,s_q,s_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 在实现Transformer模型中的相对位置编码（Relative Position Encoding），它是Transformer模型的一个变体，用于增
        # 强模型对序列中元素之间相对位置信息的捕捉能力。
        # 这两种类型都考虑了查询（query）和键（key）之间的相对位置，但后者还额外考虑了键（key）自身的位置信息
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # 获取查询和键的长度：从query_layer和key_layer的形状中获取查询和键的长度
            query_length, key_length = query_layer.shape[2], key_layer.shape[2] # s
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility
            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        # 缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None: 
            # 通过把要忽略的位置设置成很大的负数
            attention_scores = attention_scores + attention_mask
        # 将注意力得分归一化为概率。
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # 在注意力概率上应用dropout实际上是随机地“丢弃”了某些token的注意力权重，这在直觉上可能
        # 有些不寻常，但它是从原始的Transformer论文中继承下来的做法。
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # (b,h,s_q,s_k)@(b,h,s_v,dk)-->(b,h,s_q,dk)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (b,s_q,h,dk)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape) # (b,s_q,d)
        # 输出根据是否输出注意力权重
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder: # 如果是解码器的话,加上past_key_value
            outputs = outputs + (past_key_value,)
        return outputs
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Ernie
class ErnieSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # input_tensor是注意力机制之前的输入，而hidden_states是注意力机制处理后的输出。在SelfOutput层中，这两个张量通
        # 过残差连接被组合在一起，以生成该层的最终输出。
        hidden_states = self.dense(hidden_states) # (b,s,d)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    index = index.to(layer.weight.device) # 确保索引在正确的设备上
    # 使用index_select方法根据dim参数（默认为0）从原始权重中选取指定索引的元素，并克隆（clone
    # ）和分离（detach）这些元素，以创建新的权重矩阵W
    W = layer.weight.index_select(dim, index).clone().detach() 
    if layer.bias is not None:
        #　如果原线性层有偏置，并且dim为1，则直接复制偏置
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            # 根据索引选择偏置的子集
            b = layer.bias[index].clone().detach()
    # 根据修剪后的权重大小（即index的长度）创建新的线性层。注意，这里应该使用new_size[1]作为输
    # 入特征数量，new_size[0]作为输出特征数量
    new_size = list(layer.weight.size())
    new_size[dim] = len(index) 
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    # 复制权重和偏置：将修剪后的权重和偏置复制到新线性层中，并重新启用梯度跟踪。
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer
class ErnieAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # ErnieSelfAttention类
        self.self = ERNIE_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config, position_embedding_type=position_embedding_type
        )
        self.output = ErnieSelfOutput(config)
        self.pruned_heads = set() # 去重集合
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 返回未修剪的头的索引和未修剪的头
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # 更新超参数并存储已修剪的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
class ErnieIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果hidden_act是字符串,找字典映射
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
class ErnieOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
class ErnieLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ErnieAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ErnieAttention(config, position_embedding_type="absolute")
        self.intermediate = ErnieIntermediate(config)
        self.output = ErnieOutput(config)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # ni-directional self-attention: 单向自注意力。自注意力（self-attention）是Transformer模型中的一
        # 个关键组件，它允许模型在处理序列中的每个元素时，都考虑到序列中的其他元素。单向（uni-directional）意味着
        # 在处理序列时，模型只能看到当前位置之前的元素，这有助于保持模型在处理语言任务时的顺序性。
        # 在Transformer的解码过程中，为了加快处理速度并减少重复计算，通常会缓存之前的自注意力层的键（key）和值（value）
        # 。这是因为自注意力机制在计算当前位置的表示时，需要 参考序列中其他位置的表示，而这些表示（通过键和值来体现）在之
        # 前的计算中已经得到过，因此可以直接从缓存中获取，而无需重新计算。
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0] # 注意力输出
        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:] 
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
class ErnieEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config：一个配置对象，包含了模型的各种参数设置
        self.config = config
        # 其中包含多个 ErnieLayer 实例，数量由 config.num_hidden_layers 指定。
        self.layer = nn.ModuleList([ErnieLayer(config) for _ in range(config.num_hidden_layers)])
        # 用于控制是否启用梯度检验点（gradient checkpointing），这是一种减少内存使用的方法，通过重新计算
        # 而非存储中间层的梯度来实现。
        self.gradient_checkpointing = False
    def forward(
        self,
        hidden_states: torch.Tensor, # hidden_states：输入到编码器的隐藏状态。
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # 这些参数用于控制编码器的行为，如是否使用注意力掩码、是否缓存过去的键值对、是否输出注意力权重等。
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        if self.gradient_checkpointing and self.training:
            # 梯度检验和use_cache=True不兼容
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        next_decoder_cache = () if use_cache else None
        # 遍历每一层 ErnieLayer，对输入进行逐层处理。
        for i, layer_module in enumerate(self.layer):
            # 装进元组
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            # 如果启用了梯度检验点且处于训练模式，则使用 _gradient_checkpointing_func 函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            hidden_states = layer_outputs[0] # 更新后的隐藏状态
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class ErniePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 它主要用于处理序列模型（如 Transformer）中的隐藏状态，并通过一个特定的“池化”方法来获
        # 取一个固定大小的表示（representation）。在这个例子中，ErniePooler 类通过简单地选择序列
        # 中第一个标记（token）的隐藏状态作为输入，然后通过一个线性层（nn.Linear）和一个激活函数（
        # 这里使用的是双曲正切函数 nn.Tanh）来进一步处理这个隐藏状态，从而得到一个池化后的输出。
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class ErniePredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
class ErnieLMPredictionHead(nn.Module):
    # 它通常用于自然语言处理任务中的语言模型预测头（Language Model Prediction Head）。
    # 这个类继承自 nn.Module，并在其内部实现了用于预测下一个词或标记的逻辑。
    # 在某些模型架构中，特别是像BERT或ERNIE这样的预训练语言模型中，输出层的权重（即decoder线性层的权重）
    # 会被设置为与输入嵌入层的权重相同。这种设置通常被称为“权重绑定”（weight tying）或“权重共享”（
    # weight sharing）。它的好处包括减少模型参数数量、提高训练效率以及在某些情况下改善模型性能。
    def __init__(self, config):
        super().__init__()
        self.transform = ErniePredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        #  一个线性层（nn.Linear），它将变换后的隐藏状态映射到词汇表大小（config.vocab_size）的输出空间。
        # 这里设置 bias=False 意味着线性层本身不学习偏置项，偏置项将通过 self.bias 单独管理
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.bias: 一个可学习的参数，形状为 [config.vocab_size]，用作输出层的偏置项。通过将 self.decoder.bias 
        # 设置为 self.bias，我们确保了当词汇表大小改变时（通过 resize_token_embeddings 等方法），偏置项也能被正
        # 确地调整大小。
        # 这实际上是在告诉PyTorch：“虽然我没有在nn.Linear中请求偏置项，但我想自己管理一个偏置项，并将其视为这个线性层
        # 的偏置项。”然而，这种做法并不是PyTorch的常规用法，因为它绕过了nn.Linear对偏置项的内部管理。
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 每个标记有一个唯一的输出偏置项：尽管输出权重可能与输入嵌入共享，但每个标记（token）在输出层都有一个唯一的偏置项
        # 。这些偏置项是模型参数的一部分，用于调整模型对每个标记的预测概率
        # 这个注释解释了为什么需要将decoder线性层的偏置项（self.bias）与线性层本身的偏置项（self.decoder.bias）建
        # 立链接。这主要是为了确保当词汇表大小改变时（例如，通过添加新的标记到模型中），偏置项的大小能够正确地被调整。
        self.decoder.bias = self.bias
    # 它的存在是为了在需要时确保 self.decoder.bias 和 self.bias 保持链接。
    def _tie_weights(self):
        self.decoder.bias = self.bias
    # 输入 hidden_states 是一个张量，通常包含了模型对序列中每个标记的隐藏状态表示。
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)  # 首先对隐藏状态进行变换。
        # 然后将变换后的隐藏状态通过线性层（加上偏置项）映射到词汇表大小的输出空间
        hidden_states = self.decoder(hidden_states) 
        # 返回最终的输出 hidden_states
        return hidden_states # (b,s,v)
class ErnieOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = ErnieLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
# Copied from transformers.models.bert.modeling_bert.BertOnlyNSPHead with Bert->Ernie
class ErnieOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score
class ErniePreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = ErnieLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score
class ErniePreTrainedModel(PreTrainedModel):
    config_class = ErnieConfig
    base_model_prefix = "ernie"
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
@dataclass
# Copied from transformers.models.bert.modeling_bert.BertForPreTrainingOutput with Bert->Ernie
class ErnieForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
class ErnieModel(ErniePreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = ErnieEmbeddings(config)
        self.encoder = ErnieEncoder(config)

        self.pooler = ErniePooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertModel.get_input_embeddings
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # Copied from transformers.models.bert.modeling_bert.BertModel.set_input_embeddings
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # Copied from transformers.models.bert.modeling_bert.BertModel._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

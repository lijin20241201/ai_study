from torch.nn import Module
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int # in_features: 输入特征的数量。
    out_features: int # out_features: 输出特征的数量
    weight: Tensor
    # bias: 是否使用偏置项，默认为 True。
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        # factory_kwargs 是一个字典，包含了 device 和 dtype
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 初始化权重 weight，形状为 (out_features, in_features)。
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # 如果 bias 为 True，则初始化偏置 bias，形状为 (out_features)；否则，不使用偏置
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        # 调用 reset_parameters 方法初始化权重和偏置。
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # 使用 kaiming_uniform_ 初始化权重，这是一种常用的初始化方法，适合激活函数如 ReLU。
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # 如果有偏置项，则使用均匀分布进行初始化
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    # 使用 F.linear 函数执行线性变换，返回结果。
    def forward(self, input: Tensor) -> Tensor:
        # 在 PyTorch 的内部，torch.nn.functional.linear 通常是一个 Python 接口，它封装了底层 C++ 
        # 实现的线性变换函数。当你在 Python 中调用 torch.nn.functional.linear 时，实际上是在调用底
        # 层的 C++ 实现，以获得更高的性能。内部会对w转置
        return F.linear(input, self.weight, self.bias)
    # 返回一个字符串，描述线性层的基本属性。
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf # nf：输出特征的数量。nx：输入特征的数量
        # 初始化权重和偏置，并使用nn.init.normal_对权重进行初始化。
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02) # 权重初始化
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,) # size_out：计算输出的形状
        # x.view(-1, x.size(-1))：将输入张量展平，以便进行矩阵乘法。
        # 使用addmm函数进行矩阵乘法，并加上偏置。
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out) # 将结果重塑为原来的形状
        return x

# 什么是连续存储？
# 一个张量在内存中被认为是连续存储的（contiguous），如果它的元素按照其形状和步幅（strides）在内存中紧密排列在一起
# 一维张量的所有元素都是连续的。
# 多维张量按照其形状和步幅在内存中紧密排列，即每个维度的元素都是紧密相连的。
# 何时需要使用contiguous()？
# 当你需要确保张量在内存中是连续存储的，尤其是在以下情况时：
# 将张量传递给某些操作或函数时，这些操作可能要求输入是连续的。
# 使用.view()方法改变张量的形状时，如果原张量不是连续存储的，则需要先调用contiguous()。
# 在进行某些操作（如某些类型的内存拷贝）时，确保数据是连续的可以提高效率。
# 返回新的layer,new_layer就是一个新的Conv1D层，其权重和偏置已被修剪，并且保持了与原层相同的设备和计算梯度的能力。
def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = 1) -> Conv1D:
    index = index.to(layer.weight.device) # 确保索引张量位于与layer相同的设备上
    # layer.weight得到的是形状如[512, 1536]的权重矩阵,index_select(dim, index)
    # 根据提供的索引选择权重张量中的列或行。
    # Conv1D层的权重形状通常为 [input_features,output_features]。修剪时，根据dim参数的不同，可以
    # 选择沿输入特征维度（dim=1）或输出特征维度（dim=0）进行修剪。
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else: # 选中index对应的那些项
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size()) # 获取layer权重的形状[512, 1536]
    new_size[dim] = len(index)  # [512, 1152],改变new_size索引1的维度的长度
    #复制权重和偏置：复制选中的权重和偏置，并关闭梯度计算。
    # 创建新层：创建一个新的Conv1D层，并设置其权重和偏置
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    # 在复制权重和偏置时，先关闭梯度计算，然后再启用，这是为了避免在复制过程中引入不必要的计算图。
    new_layer.weight.requires_grad = False # 设置禁用w梯度
    # W.contiguous() 的使用是为了确保在复制权重数据时，数据是连续存储的。这样做可以确保在进行某些操作时不会出
    # 现问题，并且可以提高操作的效率。
    new_layer.weight.copy_(W.contiguous()) # 复制W的权重数据
    new_layer.weight.requires_grad = True # 启用梯度
    # 启用梯度计算：在复制完权重和偏置后，重新启用梯度计算。
    new_layer.bias.requires_grad = False # 禁用bias梯度
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer
def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer
# 我们希望找到可以被安全移除的heads（也就是那些对模型性能贡献较小或冗余的heads）
# 同时也要确定哪些heads仍然是活跃的（即没有被修剪的）
# 传人参数:heads (List[int])：整数列表，要修剪的头的索引。n_heads (int)：头数
# head_size: int,指的是每一个注意力头的维度大小。already_pruned_heads (Set[int])：这是一个整数集合，包含了已经被修剪
# 掉的heads的索引。集合的数据结构保证了元素的唯一性，所以不会有重复的索引
# 函数的目的是确定哪些heads是可以进一步修剪的（即它们不在already_pruned_heads集合中）,并且计算出所有未被修剪的
# heads的索引。这对于优化模型的大小或者减少计算成本是有帮助的
# 返回的结果是一个元组，包含了两个元素：第一个元素是一个集合 (Set[int])，包含了被修剪的heads的索引。
# 第二个元素是一个torch.LongTensor，包含了所有未被修剪的头的嵌入的索引
# 修剪操作：当我们修剪掉一个或几个head时，实际上减少的是注意力机制的输出维度的一部分。这意味着在修剪之后，注意力机制的输
# 出维度将会是剩余未被修剪的heads的数量乘以每个head的head_size。
# 保持维度一致：为了保持模型其他部分的一致性，通常会有一个额外的线性层来将修剪后的注意力机制输出转换回原来的维度。这样做的
# 目的是确保修剪后的模型可以继续与其它层（如前馈神经网络层等）兼容。
def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    # 创建一个全为1的掩码，用于标记哪些头是可用的（未修剪的）
    mask = torch.ones(n_heads, head_size) 
    # 转换输入的heads列表为集合，并去除已经修剪过的头的索引
    heads = set(heads) - already_pruned_heads 
    # 遍历剩余的、需要修剪的head索引，更新掩码
    for head in heads:
        # 因为要根据这个head索引来设置把mask中哪些嵌入设置为0,这里把在当前头索引之前的
        # 设置为1,之后的为0,为1说明要移位
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0 # 将要修剪的头的对应位置在掩码中设为0
    # 将二维的掩码展平为一维，并只保留值为1的元素（即未修剪的头）的索引,注意：.eq(1) 生成一
    # 个布尔型tensor，其中True表示对应位置的值是1
    mask = mask.view(-1).contiguous().eq(1)
    # 使用torch.arange生成一个从0到len(mask)-1的tensor，然后通过布尔索引选出值为True的索引  
    # 这些索引对应于未修剪的头的位置  
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    # 函数返回当前修剪过的头索引列表,未修剪的头索引对应的嵌入位置
    return heads, index
# 在 Python 字符串前加上 r 前缀，表示该字符串是一个 原生字符串（raw string）。原生字符串的主要用途是忽略所有的转义字符，除了最基本的 \0、\n、\r、\t、\b、\f 和 \a 以及反斜杠本身 \。
# 即使字符串中包含了转义字符 \n、\t 和 \r，它们也会被视为普通的字符串，而不是实际的换行、制表符或回车符。
# .detach() 的作用：创建一个新的张量，这个新的张量从计算图中分离出来，不参与梯度计算。
# 避免影响梯度计算：在进行某些操作时，如重归一化，使用 .detach() 可以确保这些操作不影响到原来的张量的梯度信息。
def _no_grad_embedding_renorm_(weight: Tensor, input: Tensor, max_norm: float, norm_type: float) -> Tuple[Tensor, Tensor]:
    torch.embedding_renorm_(weight.detach(), input, max_norm, norm_type)
def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    if has_torch_function_variadic(input, weight):
        return handle_torch_function(
            embedding,
            (input, weight),
            input,
            weight,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
    
    # 在初始化权重矩阵时，通常会将 padding_idx 对应的行设置为全零向量，以确保其不参与训练。
    # 在使用负索引转换为非负索引时，转换后的索引不应该对应词汇表中的某个已有词汇。这是为了避免混淆和冲突。
    if padding_idx is not None:
        if padding_idx > 0: # 在词汇表的索引范围内
            assert padding_idx < weight.size(0), "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(0), "Padding_idx must be within num_embeddings"
            padding_idx = weight.size(0) + padding_idx
    else:
        padding_idx = -1
    if max_norm is not None:
        # Note [embedding_renorm contiguous]
        # `embedding_renorm_` will call .contiguous() on input anyways, so we
        # call it here and take advantage of the improved locality in the
        # `embedding` call below too.
        input = input.contiguous()
        # Note [embedding_renorm set_grad_enabled]
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.embedding_renorm_
        # remove once script supports set_grad_enabled
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    # 在内部实现中，torch.embedding 函数可能定义在一个 C++ 文件中，并通过 TorchScript 或 JIT 编译器
    # 进行调用。这些函数通常位于 aten/src/ATen/native/Embedding.cpp 文件中。
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
# 实现自定义的前向传播和反向传播逻辑。这类自定义函数在 PyTorch 中非常有用，尤其是在需要特殊的张量操作时
# 自定义的自动梯度函数，可以用于更精细地控制张量操作及其梯度计算
# 这个 IndexFirstAxis 类实现了以下功能：
# 前向传播：
# 使用 torch.gather 沿着第一个轴进行索引操作。
# 保存必要的数据用于反向传播。
# 反向传播：
# 使用 scatter_ 沿着第一个轴进行梯度的散射操作。
# 重塑最终的梯度输入以匹配输入张量的形状。
class IndexFirstAxis(torch.autograd.Function):
    @staticmethod # 静态方法
    def forward(ctx, input, indices):
        # input:(b* s, h,dk),indices:一维的索引列表
        # 保存用于反向传播的数据：保存 indices 用于后续的反向传播。
        ctx.save_for_backward(indices)
        assert input.ndim >= 2 # 确保输入张量至少有两个维度
        # 分离第一个轴的维度和其他轴的形状。
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        # .numel(),计算元素总数
        second_dim = other_shape.numel() # 计算除第一个轴之外的维度的元素总数。
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        # 使用 rearrange 将输入张量的形状从 (b, ..., d) 转换为 (b, d)。
        # 使用 repeat 函数重复 indices 以匹配second_dim的大小。
        # 使用 torch.gather 沿着第一个维度进行索引操作。
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape) # (len(indices),h,dk)
    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None
def unpad_input(hidden_states, attention_mask):
    # hidden_states:(b,q_len,h,dk),attention_mask:(b,s)
    # 计算每个样本的有效长度（非填充部分的长度）。
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 非填充的token索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item() # 批次内最长序列 长度
    # 计算每个序列长度的累积和，并在开头添加一个 0 作为起始索引
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        # 首先将 hidden_states 重塑为 (b * s, ...) 的形状，然后使用 indices 索引去除填充部分。
        # (len(indices),h,dk)
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
def maybe_contiguous(x):
    # x.stride(-1) != 1 检查张量最后一个维度的步幅是否为 1，如果不是，则意味着张量不是按最后一个维度连续存储的。
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x
def _flash_attn_varlen_forward(
    q,# 查询张量。
    k, # 键张量。
    v, # 值张量
    cu_seqlens_q, # 累积序列长度张量，用于索引 Q。
    cu_seqlens_k, # 累积序列长度张量，用于索引 K/V。
    max_seqlen_q, # 最大查询序列长度。
    max_seqlen_k, # 最大键序列长度。
    dropout_p,
    softmax_scale, # 缩放因子，默认为 1 / sqrt(headdim)
    causal, # 是否应用因果掩码。
    window_size=(-1, -1), # 滑动窗口大小，如果不为 (-1, -1)，则实现局部注意力。
    softcap=0.0, # float 软限制（soft-capping）阈值。
    alibi_slopes=None, # (nheads,) 或 (batch_size, nheads) 偏置斜率。
    return_softmax=False, # 是否返回 softmax 概率。
    block_table=None,#　(batch_size, max_seqlen_q) 块表。
    leftpad_k=None,# 左侧填充信息。
    seqused_k=None,# 使用的键序列信息。
):
    # 确保输入的张量 q, k, v 是连续存储的，以便提高 CUDA 计算效率。
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    # 调用 CUDA 前向传播函数
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.varlen_fwd(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        return_softmax,
        None,
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    # out: 注意力操作的结果张量。q, k, v: 输入张量（可能已修改）out_padded: 填充后的输出张量。
    # softmax_lse: Softmax 的 log-sum-exp 结果。S_dmask: Softmax 结果和 dropout 模式。
    # rng_state: 随机状态。
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state
class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        block_table,
    ):
        # 如果 softmax_scale 未提供，则默认设置为 q 的最后一个维度的倒数平方根。
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        # 调用前向传播函数：
        # 函数返回多个结果，包括 out（输出张量）、q、k、v、out_padded（填充后的输出张量）、softmax_lse（softmax 的 logsumexp 结果）
        # 、S_dmask（softmax 结果和 dropout 模式）和 rng_state（随机状态）。
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
        )
        # 保存用于反向传播的数据：
        ctx.save_for_backward(
            q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
        )
        # ctx.save_for_backward(...)：保存 q、k、v、out_padded、softmax_lse、cu_seqlens_q、
        # cu_seqlens_k 和 rng_state 用于反向传播。
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        # 保存其他上下文信息：
        # ctx.dropout_p、ctx.max_seqlen_q、ctx.max_seqlen_k 等属性存储了前向传播时使用的参数
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        # 如果 return_softmax 为 True 并且 dropout_p > 0，则返回 (out, softmax_lse, S_dmask)；
        # 否则只返回 out。
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        # 恢复保存的数据：
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        # 初始化梯度张量
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        # 调用反向传播函数
        # _flash_attn_varlen_backward 用于执行注意力机制的反向传播计算。函数接受 dout（输出张量的梯度）
        # 以及前向传播时保存的所有张量，并计算梯度 dq、dk 和 dv
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        # 裁剪梯度张量：
        # dq, dk, dv 可能会被头部维度填充，因此需要裁剪至与 dout 相同的形状。
        # 返回 dq, dk, dv 以及一系列 None，对应于前向传播时传入的不可学习参数。
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None
def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    """
dropout_p 在评估期间应设置为 0.0。
支持多查询和组查询注意力（Multi-Query Attention/Grouped-Query Attention，简称 MQA/GQA），通过传递具有
少于 Q 头数的 K 和 V 来实现。注意 Q 中的头数必须能够被 K 和 V 中的头数整除。例如，如果 Q 有 6 个头，而 K 和
V 各有 2 个头，那么 Q 的第 0、1、2 头将关注 K 和 V 的第 0 头，而 Q 的第 3、4、5 头将关注 K 和 V 的第 1 头。
如果 causal=True，则因果掩码对齐到注意力矩阵的右下角。例如，如果 seqlen_q = 2 且 seqlen_k = 5，则因果掩码（
1 = 保留，0 = 掩盖）如下：
    1 1 1 1 0
    1 1 1 1 1
如果 seqlen_q = 5 且 seqlen_k = 2，则因果掩码如下：
    0 0
    0 0
    0 0
    1 0
    1 1
如果掩码的某一行全是零，则输出也将为零。

如果 window_size != (-1, -1)，则实现滑动窗口局部注意力。位于位置 i 的查询（query）将仅关注键（key）在范围
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] 内的键。

参数：
    q: (total_q, nheads, headdim)，其中 total_q 是批次中所有查询令牌的总数。
    k: (total_k, nheads_k, headdim)，其中 total_k 是批次中所有键令牌的总数。
    v: (total_k, nheads_k, headdim)，其中 total_k 是批次中所有键令牌的总数。
    cu_seqlens_q: (batch_size + 1,)，dtype 为 torch.int32。批次中序列的累计长度，用于索引 q。
    cu_seqlens_k: (batch_size + 1,)，dtype 为 torch.int32。批次中序列的累计长度，用于索引 kv。
    max_seqlen_q: int。批次中最大查询序列长度。
    max_seqlen_k: int。批次中最大键序列长度。
    dropout_p: float。Dropout 概率。
    softmax_scale: float。应用于 softmax 之前的 QK^T 的缩放因子，默认为 1 / sqrt(headdim)。
    causal: bool。是否应用因果注意力掩码（例如，用于自回归建模）。
    window_size: (left, right)。如果不为 (-1, -1)，则实现滑动窗口局部注意力。
    softcap: float。大于 0 的值激活软限制注意力。
    alibi_slopes: (nheads,) 或 (batch_size, nheads)，fp32。Q 的第 i 个位置和 K 的第 j 个位置之间的注意力得分加上偏置
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)。
    deterministic: bool。是否使用确定性的反向传播实现，这稍微慢一些并且使用更多的内存。前向传播始终是确定性的。
    return_attn_probs: bool。是否返回注意力概率。此选项仅用于测试。返回的概率不一定正确（可能没有正确的缩放）。

返回：
    out: (total, nheads, headdim)。
    softmax_lse [可选，如果 return_attn_probs=True]: (nheads, total_q_seqlen)。矩阵 QK^T * scaling 的 logsumexp。
    S_dmask [可选，如果 return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen)。softmax 的输出
    （可能有不同的缩放）。它还编码了 dropout 模式（负值表示该位置被丢弃，非负值表示被保留）。
"""
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
    )
# 用于高效地将指定索引处的值放入一个新张量的第一轴中。这种方法相比于普通的索引操作更为高效，
# 并且可以更好地控制反向传播时的梯度计算。可以用于更精细地控制张量操作及其梯度计算。
class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        # 保存 indices 用于反向传播。
        ctx.save_for_backward(indices)
        assert indices.ndim == 1 # 确保 indices 是一维张量，values 至少是二维张量。
        assert values.ndim >= 2
        # 创建一个全零张量 output，其形状为 (first_axis_dim, *values.shape[1:])，
        # 并确保设备和数据类型与 values 相同。
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # 使用 indices 索引将 values 的值放入 output 张量的第一轴。
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output # 返回填充好的张量 output。
    @staticmethod # 反向传播 (backward 方法)
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors # 恢复保存的数据：
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # 使用 indices 索引从 grad_output 中提取对应的梯度值。
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        # 返回 grad_values 以及两个 None，分别对应于 indices 和 first_axis_dim 的梯度。
        return grad_values, None, None
def pad_input(hidden_states, indices, batch, seqlen):
    dim = hidden_states.shape[-1] # d
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    # index_put_first_axis = IndexPutFirstAxis.apply
    # 将一个经过“去padding”处理的注意力输出重新填充回原来包含padding的形状
    # 因为这时hidden_states的索引0的轴的长度是len(indices),output[indices] = values
    # 这个操作会把这些非填充token的位置设置,而其他填充位置仍然是0
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


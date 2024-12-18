import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    get_torch_version,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
# 注意力机制的核心思想是在给定的输入序列中，让模型能够关注（或分配权重）不同的位置。在Transformer模型中，注意力机制通常分为以下几个步骤：
# 计算注意力得分：通过查询（query）、键（key）和值（value）之间的矩阵乘法计算注意力得分。
# 缩放注意力得分：通常会对注意力得分进行缩放，以防止数值不稳定。
# 应用掩码（masking）：根据不同的应用场景，可能需要对某些位置的注意力得分进行屏蔽。
# 归一化注意力得分：使用softmax函数将注意力得分转换为概率分布。
# 计算注意力输出：通过将归一化的注意力权重与值向量相乘得到注意力输出。
# 用于封装因果语言模型的输出结果。虽然它不是一个字典，但你可以像访问字典一样访问其属性。如果你需要将
# 这个对象转换为字典形式，可以使用 asdict 函数
@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

def _get_unpad_data(attention_mask):
    # 计算了每个样本的有效长度（即非填充部分的长度）
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 返回了 attention_mask 中所有非零元素（即非填充token）的索引。首先，attention_mask 被展平成一维（flatten()），
    # 然后通过 torch.nonzero(..., as_tuple=False) 获取所有非零元素的位置，并且返回的是一个包含索引的张量。最后 
    # .flatten() 将这些索引从二维转换成一维，方便后续处理。
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 找到批次中的最大有效序列长度。seqlens_in_batch 包含了批次中每个样本的有效长度，通过对这些长度取最大值（max()）
    # ，我们可以知道当前批次中最长的序列有多长,.item() 方法则是将张量转换为Python标量
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # seqlens_in_batch 是一个包含每个样本序列长度的一维张量。torch.cumsum 函数沿着指定维度（在这里是
    # dim=0，表示沿着第一个维度）对张量元素进行累积求和。
    # 累积求和的结果是一个新的张量，其中第 i 个元素是前 i+1 个序列长度的总和。这通常用于快速访问不同样本间非
    # 填充token的起始索引位置。padding 操作在累积求和得到的张量前面添加一个零元素这样做的目的是确保第一个元素为0，
    # 这可以方便地处理索引计算，尤其是在使用某些并行处理算法时，比如在实现 batched sequence 数据的高效处理时。
    # cu_seqlens 提供了每个样本（包括样本自身）及其之前的所有样本的非填充token的累计数量。这对于之后在没有填充元
    # 素的情况下处理变长序列是非常有用的。
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
# 用于从TensorFlow检查点文件加载权重到PyTorch模型中。为了能够成功加载权重，TensorFlow模型与PyTorch模型的
# 架构必须在结构上是一致的，至少在权重对应的层上要有一一对应的关系。
# 确保两个框架下的模型架构一致性是很重要的，特别是在转换预训练权重时。如果你尝试加载的模型有不同版本或者有架
# 构上的差异，那么你需要修改上述代码中的逻辑，以便正确地映射变量名和权重。
def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    # 读取TensorFlow Checkpoint
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    # 列出和加载TensorFlow检查点文件中的变量名和对应的权重数组。
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    # 映射变量名称到PyTorch模型
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())
    # 根据变量名来定位PyTorch模型中的相应层。变量名被解析以确定如何导航到特定的权重或偏置项等。
    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        # 如果找到了匹配的层并且形状一致，则使用 torch.from_numpy 将NumPy数组转换为PyTorch张量，
        # 并将其赋值给模型中的相应参数。
        pointer.data = torch.from_numpy(array)
    return model
class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings 
        # 偏置矩阵 (bias) 和掩码偏置 (masked_bias) 的注册
        # bias 是一个布尔类型的下三角矩阵，用于实现因果掩码（即，防止当前位置看到未来的信息
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # masked_bias 是一个用于掩码操作的极大负数，确保在softmax操作后，被掩码的位置概率接近于0。
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        # 嵌入维度 (embed_dim), 头数 (num_heads), 和每头的维度 (head_dim) 的设置
        self.embed_dim = config.hidden_size # d
        self.num_heads = config.num_attention_heads # h
        self.head_dim = self.embed_dim // self.num_heads # dk
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 注意力缩放和重新排序配置
        self.scale_attn_weights = config.scale_attn_weights # 决定是否在注意力计算时缩放权重。
        self.is_cross_attention = is_cross_attention # 是否是交叉注意力
        # Layer-wise attention scaling, reordering, and upcasting
        # 表示是否按层索引逆序缩放注意力权重。
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        # 表示是否重新排序和提升精度进行注意力计算。
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        # 卷积层定义 (Conv1D)
        # c_attn 和 c_proj 层用于线性变换输入和输出。
        if self.is_cross_attention:
            # 在交叉注意力情况下，c_attn 和 q_attn 分别用于编码键/值对和查询向量
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        # attn_dropout 和 resid_dropout 分别用于注意力和残差连接中的dropout。
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # 因果掩码和已剪枝头部集合 (pruned_heads)：
        self.is_causal = True  # 因果掩码
        self.pruned_heads = set()  # 用于存储已被剪枝的注意力头
    # 剪枝 (prune_heads 方法),通过剪枝减少计算复杂度
    # 此方法用于删除指定的注意力头，减少模型参数数量。它通过调整卷积层的大小来实现这一点，并更新模型的超参数。
    def prune_heads(self, heads):
        # 如果没有要修剪的头,方法返回
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        # 这个是q,k,v对应的嵌入位置
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])
        # Conv1D(3 * self.embed_dim, self.embed_dim),这里第一个是nf,表示输出,
        # 第二个是nx,表示输入,c_attn调整dim=1,是在调整输出特征
        # dim=0是在调整输入特征,因为权重形状是(in_feature,out_feature),
        # c_proj最后还要转换成512维的向量
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # Update hyper params,这里设置新的split_size
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads) # 更新新头数
        # 更新已经修剪的头的索引
        self.pruned_heads = self.pruned_heads.union(heads)
    # 注意力计算 (_attn 方法)
    # 此方法实现了多头注意力机制的核心计算步骤
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 计算注意力权重 (attn_weights),通过矩阵乘法计算原始注意力得分
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        # 根据配置进行注意力权重的缩放。
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            # 生成一个非常小的数值，用于在注意力机制中掩码（mask）那些不应该被考虑的位置。这样做是为了确保在应
            # 用softmax函数之后，这些位置的注意力权重几乎为零
            # torch.finfo 是一个获取浮点数类型的数值信息的方法。它可以返回一个对象，包含了该类型的最小正数、最
            # 大正数、精度等信息。
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            # mask_value是个很小的负数
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            # condition,input,other,根据条件:When True (nonzero), yield input, otherwise yield other
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        # 应用attention_mask,attention_mask 通常用于指示哪些位置在计算注意力得分时应该被忽略
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        # 对注意力权重进行归一化处理。
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        # 在归一化后的注意力权重上应用dropout。
        attn_weights = self.attn_dropout(attn_weights)
        # 如果我们想要禁用某些注意力头，可以使用 head_mask 来指定哪些头应该被忽略。
        # 示例：假设我们有8个头，但只想保留前4个头，那么 head_mask 可能是 [1, 1, 1, 1, 0, 0, 0, 0]
        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        # 通过将注意力权重与值向量相乘得到注意力输出。
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with torch.amp.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size) # (b,s,h,dk)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous() # (b,s,h,dk)
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape) # (b,s,d)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None: # 如果有编码器输出
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            
            query = self.q_attn(hidden_states) # (b,s,d)
            # (b,s,d)-->(b,s,2d),之后再dim=2上拆分
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask # 编码器填充掩码
        else:
            # 在dim=2上拆分成query,key,value
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        # (b,s,d)-->(b,h,s,dk)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        # 在推理阶段,需要用到之前时间步的key,value表示
        # 这种缓存机制是用在推理阶段,因为训练时，推理具有并行性,各个时间步的预测同步进行
        # 而在推理时,因为新预测的token是基于已经生成的token进行预测的,这时候目标序列输入的
        # 自注意力是query是只有上一步新生成的token,key和value是把当前token(只有这么一个)
        # 的表示和之前缓存的token(当前token之前所有token)的表示在序列长度维度进行了合并
        # 当前新生成的token与之前的token进行交互，以确定其在上下文中的位置和意义,之后预测下个token
        # 而解码器跨注意力时,因为推理时,编码器中token嵌入不会再改变,这时候可以缓存编码器输出的表示
        # 目的就是能减少计算开销
        if layer_past is not None: 
            past_key, past_value = layer_past
            # 在序列长度维度合并
            key = torch.cat((past_key, key), dim=-2) 
            value = torch.cat((past_value, value), dim=-2)
        # 如果使用缓存,就保存
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        # 合并头的嵌入为一个整体
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        # 因为有裁剪头,需要把维度投影到embed_dim
        attn_output = self.c_proj(attn_output) 
        attn_output = self.resid_dropout(attn_output) # dropout
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs  # a, present, (attentions)
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists
def is_flash_attn_greater_or_equal_2_10():
    if not _is_package_available("flash_attn"):
        return False
    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
# GPT2闪存注意模块。这个模块继承自GPT2Attention，因为模块的权重保持不变。唯一需要改动的地方是在前向传递过程中，
# 需要正确调用闪存注意的公共API，并在输入中包含任何填充标记的情况下处理这些填充标记
# 你提到的情况是在生成序列时，每次生成一个新的token，并且将这个新生成的token加入到现有的序列中，形成一个新的序列，
# 然后对这个新的序列进行自注意力计算。这是典型的序列生成过程，在每次生成新的token之后，都需要重新计算整个序列的注意力权重。
# 而我之前解释的情况是指另一种优化方法，即在生成过程中保留之前计算过的key和value对，以避免重复计算。这种方法主要用于加速推
# 理过程，特别是在长序列的情况下。在这种情况下，确实只是对当前的token做自注意力计算，并且通过保留之前的部分计算结果来节省计算资源。
# 具体来说，当我们说“将新的key/value与旧的key/value拼接起来”时，实际上是在更新一个缓存（cache），这个缓存包含了之前的计算结果
# 。每次生成新的token时，只需要计算这个新token的key和value，然后将它们添加到缓存中，而不是重新计算整个序列的key和value。
# 两种情况的区别在于：
# 生成新序列并计算自注意力：每次生成新的token后，将它加入到序列中，然后对整个序列重新计算自注意力权重。
# 使用缓存的key/value：只计算当前新token的key和value，并将它们添加到之前的缓存中，以供下一步计算使用。
# 这两种方式都可以用来进行序列生成，但使用缓存的方法在长序列任务中更加高效，因为它减少了重复计算
class GPT2FlashAttention2(GPT2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 一旦针对RoCm的Flash Attention升级到2.1版本，就应该移除这部分
        # 我的是不存在,这个值设置成True,使用左上掩码
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        bsz, _, _ = hidden_states.size()
        # 如果encoder_hidden_states不是None,就是有编码器输出,模型应该做交叉注意力,这时如果q_attn为None,就要抛出异常
        # 因为跨注意力，query和vaue,key不同
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states) # q
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2) # k,v
            attention_mask = encoder_attention_mask # 编码器掩码
        else:
            # 如果encoder_hidden_states is None,这种是自注意力,q=k=v
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        # 之后变形q,k,v,以便他们能做自注意力或交叉注意力
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        # 如果layer_past不为空,这个在训练时是没用的,在推理时的缓存机制,可以缓存解码器自注意力时目标
        # 输入序列之前的token表示,这时传进来的token只有新token,之后query只有一个token,而k,v
        # 和之前缓存的在序列上拼接,之后来预测下个token
        if layer_past is not None:
            past_key = layer_past[0] # 缓存的上次的key,就是当前token之前的token
            past_value = layer_past[1] # 缓存的上次的v
            # 在序列长度维度拼接之前的token表示和当前token表示
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache is True: #如果使用缓存
            present = (key, value) # 缓存当前的key,value

        query_length = query.shape[2] # q_len
        tgt_len = key.shape[2] # k_len和v_len

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        query = query.transpose(1, 2).view(bsz, query_length, self.num_heads, self.head_dim) # (b,q_len,h,dk)
        key = key.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
        value = value.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)

        attn_dropout = self.attn_dropout.p if self.training else 0.0
        
        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        if query.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype() # float16
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.c_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query, key, value, attention_mask, query_length, dropout=attn_dropout
        )

        attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
        attn_output = self.c_proj(attn_weights_reshaped)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights_reshaped,)

        return outputs

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        # 如果不是用左上因果掩码
        if not self._flash_attn_uses_top_left_mask: 
            causal = self.is_causal # 布尔型,True
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            # 如果self.is_causal是True,但是query_length==1的话,这个值是False
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0] # b
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

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

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )
        return attn_output
        # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    # 实现的是一个用于消除输入张量中的填充（padding）的过程，即所谓的“unpadding”，这对于处理变长序列特别有用，因为在实际应用
    # 中，不同序列的长度往往是不一样的，而模型通常需要固定长度的输入，这就需要对较短的序列进行填充以匹配最长序列的长度。然而，在
    # 计算过程中，填充的部分实际上并不参与有效的计算，因此通过这种方式去除这些无效部分可以显著减少计算资源的需求。
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 所有非零元素的位置,之前的所有样本的非填充token的累计数量,批次中最长的序列
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # b,k_len,h,dk
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        
        key_layer = index_first_axis( # (b*k_len,h,dk)
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1: # 如果q_len是1
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# GPT-2 注意力模块现在使用了 PyTorch 提供的 scaled_dot_product_attention 函数来实现注意力机制。
# 局限性：GPT2SdpaAttention 类使用了 torch.nn.functional.scaled_dot_product_attention，但后者不支
# 持 output_attentions=True 或 head_mask 这两个特性。
# 实际用途：如果你的应用场景不需要这两个特性，那么这个类依然是有用的。它提供了基于 scaled_dot_product_attention
# 的实现，通常比手动实现更加高效。
# 回退到手动实现：由于 scaled_dot_product_attention 不支持 output_attentions=True 或 head_mask，因此程序
# 会回退到使用手动实现的注意力机制。
# 手动实现：指的是不使用 GPT2SdpaAttention 类，而是使用传统的手动实现，即完全自定义的注意力机制实现。
# attn_implementation="eager" 是一个参数，用于指定在加载模型时应该使用哪种注意力机制实现。
# 含义：如果你希望强制使用传统的手动实现，而不是尝试使用 scaled_dot_product_attention，可以在加载模型时通过设置 at
# tn_implementation="eeger" 来指定。
# 效果：这样做可以避免因 scaled_dot_product_attention 不支持某些特性而导致的回退警告，并确保始终使用手动实现。
# 通过设置 attn_implementation="eager"，你可以显式地告诉模型使用传统手动实现，从而避免回退警告。
class GPT2SdpaAttention(GPT2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 在使用非连续输入和自定义注意力掩码时，torch==2.1.2 版本中的 SDPA（scaled dot product attention）
        # 带有内存高效后端的功能存在问题，因此我们需要调用 .contiguous()。这个问题在 torch==2.2.0 版本中得到了修复。
        # 在 PyTorch 2.1.2 版本中，使用内存高效的 SDPA 时，如果输入张量是非连续的（即内存布局不是连续的），并且使用了自
        # 定义的注意力掩码，那么可能会出现问题。为了绕过这个问题，可以在调用 SDPA 之前确保输入张量是连续的，即通过调用
        # .contiguous() 方法使输入张量变为连续存储。
        # 在 PyTorch 2.2.0 版本中，这个问题已经被修复，因此在使用 2.2.0 及以上版本时，不需要显式地调用 .contiguous()。
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # 如果要输出注意力权重,或者遮挡头的掩码不是None,调用父类的注意力机制
        if output_attentions or head_mask is not None:
            logger.warning_once(
                "`GPT2SdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
                "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
        
        bsz, q_len, _ = hidden_states.size() # (b,q_len,d)

        # Initial attention projections
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2SdpaAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Optional kv caching
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache is True:
            present = (key, value)

        # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
        if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
        # 内联条件（inline condition）通常是指在代码中直接使用条件表达式（如三元运算符 
        # condition ? true_expr : false_expr）来选择不同的执行路径。在某些情况下，
        # 内联条件可能会导致编译器或优化器无法有效地处理动态形状。
        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # 使用因果掩码的条件:attention_mask是None,并且q_len > 1,并且不是交叉注意力
        is_causal = True if attention_mask is None and q_len > 1 and not is_cross_attention else False
        # 
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.embed_dim)

        # Final projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present, None
class GPT2MLP(nn.Module): # 前馈全连接层
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size # d
        self.c_fc = Conv1D(intermediate_size, embed_dim)  
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)  # d-->hidden_d
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states) # hidden_d-->d
        hidden_states = self.dropout(hidden_states)
        return hidden_states
class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size # d
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size # h_d
        # 如果config._attn_implementation是"eager",这个就是GPT2Attention
        attention_class = GPT2_ATTENTION_CLASSES[config._attn_implementation]
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        #注意力机制
        self.attn = attention_class(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            # 交叉注意力
            self.crossattention = attention_class(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        # 先进行层标准化
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection,自注意力前后残差连接
        hidden_states = attn_output + residual
        # 如果有编码器输出的话
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            # 如果它没有crossattention这个属性的话,就抛出错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            
            residual = hidden_states # 目标序列自注意力的输出残差后的
            hidden_states = self.ln_cross_attn(hidden_states) # 层标准化
            # 获取交叉注意力
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output # 跨注意力前后残差
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states # 前馈前后残差

        if use_cache: # 如果使用缓存,就输出缓存
            outputs = (hidden_states,) + outputs 
        else: # 否则只输出注意力权重
            outputs = (hidden_states,) + outputs[1:]
        return outputs  # hidden_states, present, (attentions, cross_attentions)
# 该类继承自 PreTrainedModel，用于处理模型权重初始化以及预训练模型的下载和加载
class GPT2PreTrainedModel(PreTrainedModel):
    config_class = GPT2Config # 指定模型的配置类，用于存储模型的各种超参数和配置信息。
    # load_tf_weights = load_tf_weights_in_gpt2 # 提供一个方法用于从 TensorFlow 模型加载权重。
    base_model_prefix = "transformer" # 指定模型的前缀，用于在加载或保存模型时标识模型的主体部分
    is_parallelizable = True # 表示模型支持并行化处理
    supports_gradient_checkpointing = True # 表示模型支持梯度检查点（gradient checkpointing），这是一种节省内存的技术
    _no_split_modules = ["GPT2Block"] # 指定不应在这些模块之间分割模型，这对于某些并行化策略很重要。
    _skip_keys_device_placement = "past_key_values" # 指定在放置设备时应跳过的键，这在多GPU设置中很有用。
    _supports_flash_attn_2 = True # 表示模型支持 Flash Attention 的第二版本。
    _supports_sdpa = True # 表示模型支持使用 scaled_dot_product_attention

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs) # 初始化父类 PreTrainedModel 的构造函数，没有额外的操作
    # 权重初始化方法,用于初始化模型的权重,处理不同类型的模块
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, Conv1D)):
            # 线性层（nn.Linear, Conv1D）,使用正态分布初始化权重。
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 嵌入层（nn.Embedding）       
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重。
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，则将填充索引对应的权重初始化为零。
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 归一化层（nn.LayerNorm）：
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_() # 初始化偏置项为零。
            module.weight.data.fill_(1.0) # 初始化权重为 1。
        # 特殊权重初始化
        for name, p in module.named_parameters(): # 循环
            if name == "c_proj.weight": # 对于名为 "c_proj.weight" 的参数，使用特殊的缩放初始化
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # 初始化权重时，考虑到了模型深度的影响，按照 GPT-2 论文中的建议进行缩放。
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))
# 它继承自 ModelOutput 并使用了 dataclass 装饰器来简化类的定义。
# 这个类主要用于封装 GPT-2 双头模型的输出结果，其中包含了模型预测结果、损失、过去的键值对以及其他中间结果
# 使用 dataclass 装饰器的好处
# 使用 dataclass 装饰器可以简化类的定义，自动为类添加一些常用的方法，如 __init__、__repr__、__eq__ 等。此外，
# dataclass 还支持默认值和类型注解，使得类的定义更加简洁明了。
# GPT2DoubleHeadsModelOutput 类主要用于封装 GPT-2 双头模型的输出结果，其中包括了损失、原始输出、过去的键值对以
# 及中间计算的结果。使用 dataclass 装饰器可以简化类的定义，并提供便捷的数据封装功能。通过这种方式，可以方便地管理和
# 访问模型的输出数据。
@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None # 模型的总损失（如果有）
    mc_loss: Optional[torch.FloatTensor] = None # 多分类（Multi-Class Classification）损失
    logits: torch.FloatTensor = None # 模型的原始输出（通常用于后续处理，如分类）
    mc_logits: torch.FloatTensor = None # 多分类任务的原始输出
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None # 过去的键值对（用于解码阶段的缓存)
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None # 每一层的隐藏状态
    attentions: Optional[Tuple[torch.FloatTensor]] = None # 每一层的注意力权重
# GPT-2 双头模型（Double Heads Model）中的“双头”并不是指多头注意力（multi-head attention）中的“头”。在这里，
# “双头”指的是模型具有两个不同的输出头（output heads），分别用于不同的任务。
# 在自然语言处理（NLP）任务中，双头模型通常是指一个模型同时具有两个不同的输出，每个输出头负责一个特定的任务。这样的
# 设计可以让模型在一次前向传播中完成多项任务，从而提高效率并共享底层特征表示。
# 在 GPT-2 的上下文中，双头模型通常包括以下两个输出头：
# LM Head（语言模型头）：用于生成下一个词的概率分布。通常用于语言建模任务，如文本生成。
# MC Head（多分类头）,用于多分类任务，如文本分类或其他监督学习任务。通常用于下游任务，如情感分析、问答等。
# 在许多 NLP 任务中，第一个 token（通常是 BOS 或 [CLS]）被用来表示整个句子的语义信息。
# 在 GPT-2 双头模型中，确实应该取句子的第一个 token（通常是 BOS）的输出作为句子级别的表示。这样可以更好地
# 与句子相关的任务（如文本分类）相结合，并且与许多其他预训练模型的做法保持一致
def get_device_map(n_layers, devices): # {0: [0, 1, 2, 3, 4, 5]}
    layers = list(range(n_layers))
    n_blocks = int(math.ceil(n_layers / len(devices)))
    layers_list = [layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks)]
    return dict(zip(devices, layers_list))
class ModuleUtilsMixin:
    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None
    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
        return None
    def add_memory_hooks(self):
        
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        self.reset_memory_hooks_state()
    def reset_memory_hooks_state(self):
        """
        Reset the `mem_rss_diff` attribute of each module (see [`~modeling_utils.ModuleUtilsMixin.add_memory_hooks`]).
        """
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0
    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        # 
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask

    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
        else:
            device = attention_mask.device
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> Tensor:
        
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
       
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.parameters())

        total_numel = []
        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)

        if is_loaded_in_4bit:
            if is_bitsandbytes_available():
                import bitsandbytes as bnb
            else:
                raise ValueError(
                    "bitsandbytes is not installed but it seems that the model has been loaded in 4bit precision, something went wrong"
                    " make sure to install bitsandbytes with `pip install bitsandbytes`. You also need a GPU. "
                )

        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                # For 4bit models, we need to multiply the number of parameters by 2 as half of the parameters are
                # used for the 4bit quantization (uint8 tensors are stored)
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    if hasattr(param, "element_size"):
                        num_bytes = param.element_size()
                    elif hasattr(param, "quant_storage"):
                        num_bytes = param.quant_storage.itemsize
                    else:
                        num_bytes = 1
                    total_numel.append(param.numel() * 2 * num_bytes)
                else:
                    total_numel.append(param.numel())

        return sum(total_numel)

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
       
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        elif "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        return 0

    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)
# 实现梯度检查点（gradient checkpointing）的核心函数。这个函数负责在前向传播过程中记录必要的信息，并在反向传播过程中重新计
# 算中间激活值，而不是从内存中读取，从而减少内存使用。
# @torch._disable_dynamo：这是一个装饰器，用于禁用 PyTorch 的 Dynamo 编译器。这通常是为了确保 checkpoint
# 函数的行为不受编译器优化的影响。
# 在实际使用中，你可以通过 gradient_checkpointing_func 来包装模型的前向传播函数，从而在训练过程中启用梯度检查点功能。
# 这样可以减少内存使用，尤其是在处理大型模型时非常有用。
@torch._disable_dynamo
def checkpoint(
    function, # 这是要被检查点化的前向传播函数。
    *args, # 传递给 function 的位置参数。
    use_reentrant: Optional[bool] = None, # 一个布尔值，决定是否使用可重入模式。
    # 一个可调用的对象，用于提供上下文管理器。
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    # 一个字符串，指定确定性检查模式。
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,# 一个布尔值，指定是否开启调试模式。
    **kwargs
):
    # 这一段代码检查 use_reentrant 是否为 None。如果是 None，则发出警告，并将 use_reentrant 设为
    # True。未来默认值可能会改为 False。
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint: please pass in use_reentrant=True or "
            "use_reentrant=False explicitly. The default value of use_reentrant "
            "will be updated to be False in the future. To maintain current "
            "behavior, pass use_reentrant=True. It is recommended that you use "
            "use_reentrant=False. Refer to docs for more details on the "
            "differences between the two variants."
        )
        use_reentrant = True
    # 这一段代码从 kwargs 中取出 preserve_rng_state，并检查是否有其他意外的关键字参数。
    # 如果有，并且 use_reentrant 为 True，则抛出错误。
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )
    # 如果 use_reentrant 为 True，则检查 context_fn 和 debug 是否为默认值，并返回 
    # CheckpointFunction.apply 的结果。
    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        # 如果 use_reentrant 为 False，则调用 _checkpoint_without_reentrant_generator 函数，并
        # 通过生成器来处理前向和后向逻辑。
        # checkpoint_without_reentrant_generator：这是一个生成器函数，用于处理非可重入模式下的梯度检查点逻辑。
        # 它负责在前向传播时记录必要的信息，并在反向传播时重新计算中间激活值。
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret
@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
# @add_start_docstrings(
#     "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
#     GPT2_START_DOCSTRING,
# )
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config) # 初始化父类
        self.embed_dim = config.hidden_size # d 
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim) #(v,d)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim) # (p,d)
        self.drop = nn.Dropout(config.embd_pdrop) # dropout
        # 多层GPT2Block(这相当于encoder和decoder的混合类)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # 模型并行化相关设置
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        # 初始化权重
        self.post_init()
    # 如果你使用的是 transformers 库中的模型，并且这些模型提供了 parallelize 和 deparallelize 方法，那么你可以直接使用这些
    # 方法来进行模型的并行化处理。
    # parallelize 方法确实是用于将模型的不同部分分配到不同的 GPU 上，但这并不是传统的多 GPU 训练方法（如 DataParallel 或
    # DistributedDataParallel）。相反，它是一种模型并行（model parallelism）的方式，即将模型的不同部分（如 GPT2Block）
    # 分配到不同的设备上，从而实现更高效的计算。就是模型如果参数太多太大,就把模型中的gptblock均分给设备中的不同的gpu,这个需要
    # 多gpu.模型并行:模型的不同部分被分配到不同的设备上,通过将模型分割成不同的部分并在不同设备上执行，可以有效地利用多个 GPU 的
    # 计算能力
    # 数据并行（Data Parallelism）相同的模型被复制到不同的设备上。不同设备上的模型分别处理不同的数据批次。
    # 最终将各个设备上的结果汇总起来进行更新。适用于模型较小但数据量大的情况
    # 在 GPT2Model 类中定义的 parallelize 方法实现了模型并行。具体来说：
    # 通过 get_device_map 函数得到一个设备映射表，指示每一层应该分配到哪个设备上。
    # 设置 self.model_parallel = True 表示模型已经并行化。移动模型层到指定设备：
    # 将词汇嵌入层 wte 和位置嵌入层 wpe 移动到第一台设备上。将每一层 GPT2Block 分配到相应的设备上。
    # 并行化方法 parallelize
    # @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map,发出警告
        warnings.warn(
            "`GPT2Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        self.device_map = ( # 设置设备映射
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True # 设置并行标志
        # 设置设备
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # 移动嵌入层到设备
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # 移动每一层到相应的设备
        for k, v in self.device_map.items():
            for block in v: # 遍历每一个GPT2Block的索引
                cuda_device = "cuda:" + str(k) # cuda0...
                # 把每一个block加到设备上
                self.h[block] = self.h[block].to(cuda_device)
        # 移动最终的层规范化层
        self.ln_f = self.ln_f.to(self.last_device)
    # 去并行化方法 deparallelize
    # 在训练完成后，或者需要将模型保存下来以便将来使用时，最好将模型恢复到单一设备上，
    # 这样可以简化模型的管理和存储。去并行化可以确保模型的所有部分都在同一个设备上，方便后续的操作
    # 在并行化过程中，模型的部分会被分配到不同的设备上，占用多个设备的内存资源。当不再需要并行化时，去并行化可以释放
    # 这些设备上的内存，以便用于其他任务。
    # 你提到的“单一设备”在这里确实指的是将模型的所有部分整合到一个设备上，通常是单个 GPU。这样做有几个目的：
    # 简化模型管理：当模型的所有部分都在一个设备上时，更容易管理和保存模型。
    # 简化推理过程：在推理阶段，通常不需要多 GPU 的并行计算，因此将模型整合到一个设备上可以简化推理过程。
    # 通过 model.deparallelize() 方法，将模型的所有部分恢复到单个设备上。
    # 使用 model.to(device) 方法将模型移动到指定的设备上。这里的 device 是指第一个设备，通常是第一个 GPU。
    # @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn( # 发出警告
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False # 设置并行标志为 False
        self.device_map = None # 清除设备映射
        self.first_device = "cpu" # 设置所有设备为 CPU
        self.last_device = "cpu"
        # 移动所有层到 CPU
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        # 清空 CUDA 缓存
        torch.cuda.empty_cache() 

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items(): # 遍历每一层需要微调的头索引
            # 调用block的注意力attn层来微调头
            self.h[layer].attn.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPastAndCrossAttentions,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # 是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = ( # 输出隐藏状态(就是普通transformer的编码器输出或者解码器输出)
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 是否启用缓存
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 是否返回字典形式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 不能同时指定input_ids,inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果input_ids存在
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size() # (b,s)
            input_ids = input_ids.view(-1, input_shape[-1]) #(b,s)
            batch_size = input_ids.shape[0] # b
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1] # (b,s) 
            batch_size = inputs_embeds.shape[0] # b
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        # 设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # token_type_ids
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])# (b,s)

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h)) # 多个None的元组
        else: # k_len
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0) # (1,s),在样本维度增加一维
        
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)  # (b,s,d)
        position_embeds = self.wpe(position_ids) # (b,p,d)
        # 给序列中的token带上位置信息,这个位置可学习
        hidden_states = inputs_embeds + position_embeds 
        # Attention mask. 用sdpa的条件,返回布尔值
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if attention_mask is not None: # attention_mask给定的情况下(填充掩码之类的)
            attention_mask = attention_mask.view(batch_size, -1) # (b,s)
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif _use_sdpa:  
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask=attention_mask,
                    input_shape=(batch_size, input_shape[-1]),
                    inputs_embeds=inputs_embeds,
                    past_key_values_length=past_length,
                )
            else:
                # 我们从一个2D的mask创建一个3D的注意力掩码。尺寸是[batch_size, 1, 1, to_seq_length]
                # 因此我们可以广播到[batch_size, num_heads, from_seq_length, to_seq_length]
                # 这个注意力掩码比OpenAI GPT中使用的因果注意力的三角掩码更简单，我们只需要在这里准备广播维度即可。
                attention_mask = attention_mask[:, None, None, :]
                # 由于attention_mask在我们想要关注的位置是1.0，在被屏蔽的位置是0.0，这个操作将会创建一个张量，
                #其中在我们想要关注的位置是0.0,在被屏蔽的位置是该dtype的最小值。由于我们是在softmax之前的原始
                # 得分上添加它，这实际上等同于完全移除这些位置。torch.finfo(self.dtype).min,很大的负数
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size() # (b,s,d)
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length) # (b,s)
            if encoder_attention_mask is None:
                # 全1掩码
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        # 获取掩码头,如果head_mask传人None,返回None列表
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids) # 嵌入
            hidden_states = hidden_states + token_type_embeds  # 加上token_type嵌入

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),) # (b,s,d)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None # 给缓存初始化
        all_self_attentions = () if output_attentions else None # 自注意力权重
        # 交叉注意力
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None # hidden_states
        # 遍历每一层和对应层的缓存
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel: # 并行
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                # 如果用了缓存
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            # 保存每一层的输出和进入第一层之前的嵌入
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 在深度学习模型的训练过程中，特别是在处理大规模模型时，前向传播和反向传播过程中需要
            # 存储大量的中间激活值（intermediate activations）。这些中间激活值主要用于反向传播时计算梯度
            # 梯度检查点通过以下方式减少GPU内存消耗
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else: # 不使用梯度检查点时
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            # block(解码器块)的输出，这个hidden_states处于循环内,一直在变,这次的输出就是下次的输入
            hidden_states = outputs[0]
            if use_cache is True: # 如果使用缓存,这里会依次加每一层的缓存k,v
                presents = presents + (outputs[1],) 

            if output_attentions:
                # 依次加每一层的自注意力权重
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                # 依次加每一层的交叉注意力权重
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
            
            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel: # 如果用了并行
                for k, v in self.device_map.items():
                    # 在并行时，模型各个层被分配给不同的gpu(需要多gpu设备),这里判断如果
                    # 是当前设备上的层都处理过了,就把hidden_states交给下个gpu,这样依次
                    # 经过整个模型的层,直到到最后一个设备的最后一个层
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
        # 这时的hidden_states是经过所有解码器块的输出
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape) # (b,s,d)
        # Add last hidden state,添加最后一个hidden states
        if output_hidden_states: # 如果要输出hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)
        # 返回元组,元组内还是元组
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
def assert_device_map(device_map, num_blocks):
    blocks = list(range(0, num_blocks)) # 层索引列表
    # 获取分配给各个gpu device的所有层索引
    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]
    duplicate_blocks = [] # # 重复块检验,检查是否有重复块索引
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    # Missing blocks
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]
    # 如果有重复块,缺失块,额外块,报错
    if len(duplicate_blocks) != 0:
        raise ValueError(
            "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device."
            " These attention blocks were specified more than once: " + str(duplicate_blocks)
        )
    if len(missing_blocks) != 0:
        raise ValueError(
            "There are attention blocks for this model that are not specified in the device_map. Add these attention "
            "blocks to a device on the device_map: " + str(missing_blocks)
        )
    if len(extra_blocks) != 0:
        raise ValueError(
            "The device_map contains more attention blocks than this model has. Remove these from the device_map:"
            + str(extra_blocks)
        )
# @add_start_docstrings(
#     """
#     The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
#     embeddings).
#     """,
#     GPT2_START_DOCSTRING,
# )
class GPT2LMHeadModel(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config) # gpt model
        # 输出词汇大小的概率分布，表示下个token的预测
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False # 是否并行
        self.device_map = None 

        # Initialize weights and apply final processing
        self.post_init() # 初始化权重
    # 并行
    # @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`GPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        # 设备gpu索引到各个gpt层的映射
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 检查重复层索引,缺失和多余的层索引,有就报错
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map) # 并行
        # 把输出层交给第一个gpu设备
        self.lm_head = self.lm_head.to(self.transformer.first_device) 
        self.model_parallel = True # 设置并行
    # @add_start_docstrings(DEPARALLELIZE_DOCSTRING) # 去并行化方法
    def deparallelize(self): #
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 将并行化的模型恢复到单个设备（通常是 CPU）
        self.transformer.deparallelize() # 调用对象的去并行
        self.transformer = self.transformer.to("cpu") # 交给cpu
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False # 设置非并行
        torch.cuda.empty_cache() # 清空cuda缓存
    # 这两个方法分别用于获取和设置模型的输出嵌入层。
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    # 该方法用于准备生成时的输入。具体步骤如下
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # 从传人的字典中弹出token_type_ids,没有就设置None
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values: # 如果存在
            past_length = past_key_values[0][0].shape[2] # k_len

            # Some generation methods already pass only the last input ID
            # 一些生成方法已经只传人最后一个输入id
            if input_ids.shape[1] > past_length: 
                # 如果传人的长度比缓存的key还长,就设置个移除长度,因为缓存是
                # 当前token前面的所有token,所以这里设置移除长度为缓存长度,这样就只剩当前token
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                # 如果没有k_len长,就设置移除长度为input_ids.shape[1] - 1,就剩一个token
                remove_prefix_length = input_ids.shape[1] - 1
            # 保证只剩一个token,就是当前token
            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None: # 这里input_ids.shape[1]已经是1,其实这里也只是1个token_type
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]
        # 获取attention_mask,没有就默认是None
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None) 
        # 如果存在 attention_mask 且不存在 position_ids，则动态创建 position_ids。
        if attention_mask is not None and position_ids is None:
            # 动态创建position_id以进行批处理生成
            # attention_mask 是一个指示哪些位置应该被模型注意的掩码矩阵。通常情况下，有效位置的值为 1，
            # 无效位置的值为 0。
            # cumsum(-1) 方法沿着指定的维度（这里是 -1，即最后一个维度）计算累计求和。这会产生一个指示当前
            # 位置相对于起始位置的偏移量的张量。
            #如果 attention_mask 为 [1, 1, 1, 0, 0]，则 cumsum(-1) 的结果为 [1, 2, 3, 3, 3]。
            # 由于 cumsum 的结果是从 1 开始的，我们需要减去 1 以使起始位置的 position_id 为 0。
            position_ids = attention_mask.long().cumsum(-1) - 1
            # 这一行代码用于处理无效位置（即 attention_mask 为 0 的位置），将它们的 position_id 设置为 1。
            # 这是因为在某些情况下，无效位置的 position_id 需要设置为一个固定值，以便模型在处理这些位置时不会
            # 产生错误的行为。将无效位置的 position_id 设置为 1 是为了保持位置标识符的一致性和连续性，避免混淆
            # 起始位置，同时确保模型能够正确处理这些位置。设置为 0 或 -1 都可能导致位置信息混乱或模型处理错误。
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values: # 如果有缓存,就只剩一个position_id
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else: #否则设置为None
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # 根据是否存在 inputs_embeds 和 past_key_values 构建输入字典
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else: 
            model_inputs = {"input_ids": input_ids}
        # 把这些键值对装进model_inputs字典
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    # @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=CausalLMOutputWithCrossAttentions,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        # 这段代码主要负责处理生成模型的前向传播过程，包括隐藏状态的处理、损失计算以及在并行环境下的设备管理
        # 设置是否return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0] # 隐藏状态

        # Set device for model parallelism
        if self.model_parallel: #如果并行
            torch.cuda.set_device(self.transformer.first_device)
            # 如果模型是在多个GPU上并行运行的，这段代码会将 hidden_states 移动到 lm_head 
            # 所在的设备上，确保数据和模型在同一设备上进行计算。
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        # 使用 lm_head 层对 hidden_states 进行转换，得到模型对下一个 token 的预测概率分布。b,s,v)
        lm_logits = self.lm_head(hidden_states)
        # 计算损失
        loss = None
        # labels 是作为输入的一部分传递进来的。如果 labels 为 None，则不会计算损失
        # 如果提供了 labels（即真实标签），则计算损失。具体步骤如下：
        if labels is not None:
            # move labels to correct device to enable model parallelism
            # 将 labels 移动到 lm_logits 所在的设备上，以支持模型并行。
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            # 将 lm_logits 和 labels 进行位移（shift），使预测的 token 对应于真实的
            # 下一个 token。
            # shift_logits 和 shift_labels 的处理方式是为了确保模型的预测能够与实际的下一个 token 进行对比。
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss() # 使用交叉熵损失函数（CrossEntropyLoss）计算损失。
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # 根据 return_dict 参数决定返回结果的形式。如果 return_dict 为 False，则返回一个元组，包含 
        # lm_logits 和其他相关信息。如果 return_dict 为 True，则返回一个 CausalLMOutputWithCrossAttentions
        # 对象，其中包含了损失、logits 以及其他相关信息（如 past key values、隐藏状态、注意力等）。
        if not return_dict: 
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    # 重新排序缓存：在使用 beam search 或 beam sample 生成文本时，重新排序 past_key_values 缓存
    # 以匹配当前的 beam_idx。
    # 静态方法是一种不依赖于类实例的状态的方法，
    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        # 此方法用于重新排序 past_key_values 缓存，当使用 beam search 或 beam sample 方法生成文本时，
        # 需要确保 past_key_values 与当前的 beam_idx 匹配。具体步骤如下：
        # 对于每一层的 past_key_values，使用 index_select 方法根据 beam_idx 重新选择对应的状态。
        # 返回重新排序后的 past_key_values。
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
# 这段代码定义了一个名为 SequenceSummary 的类，它继承自 nn.Module，主要用于对序列的隐藏状态进行总结，
# 以生成最终的输出。这个类可以根据不同的配置选项来实现多种总结策略，并可以包含额外的线性投影和激活函数。
class SequenceSummary(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # 用于确定总结隐藏状态的方式，默认为 "last"，表示使用序列的最后一个隐藏状态。
        self.summary_type = getattr(config, "summary_type", "last")
        
        if self.summary_type == "attn": 
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError
        self.summary = Identity()
        # 如果配置中有 summary_use_proj 并且为 True，则使用一个线性层对隐藏状态进行投影。投影的目标维度取决于是否有
        # summary_proj_to_labels 配置项，如果有并且 num_labels > 0，则投影到类别数量；否则，投影到隐藏层大小。
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)
        activation_string = getattr(config, "summary_activation", None)
        # 根据配置中的 summary_activation 选择激活函数。如果没有配置，则使用恒等函数（Identity）
        self.activation: Callable = get_activation(activation_string) if activation_string else Identity()
        # 如果配置中有 summary_first_dropout 或 summary_last_dropout 并且大于 0，则使用相应的丢弃率（dropout rate）
        # 创建 nn.Dropout 层。
        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)
        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)
    # hidden_states：模型的隐藏状态，形状为 (batch_size, sequence_length, hidden_size)。
    # cls_index：可选参数，用于指定分类索引，仅在 summary_type 为 "cls_index" 时使用
    def forward(
        self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        # 根据 summary_type 处理隐藏状态：
        if self.summary_type == "last": # "last"：使用序列的最后一个隐藏状态。
            output = hidden_states[:, -1]
        elif self.summary_type == "first": # 使用序列的第一个隐藏状态。
            output = hidden_states[:, 0]
        elif self.summary_type == "mean": # 计算序列隐藏状态的平均值。
            output = hidden_states.mean(dim=1) # 在序列长度维度求均值
        elif self.summary_type == "cls_index": # 使用指定的分类索引选取隐藏状态
            # 如果cls_index is None:
            # 使用 torch.full_like 方法填充该张量，值为 hidden_states 的倒数第二个维度减一（即最后一个位置的索引）
            # 形状(b,1,d)
            if cls_index is None: 
                cls_index = torch.full_like( 
                    hidden_states[..., :1, :], # (b,1,d)
                    hidden_states.shape[-2] - 1, # 最后一个位置的索引,s-1
                    dtype=torch.long,
                )
                # print(cls_index)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1) # (b,options,1,1)
                # cls_index.dim(),维度数,(-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),)
                # 将会是(-1,-1,-1,d)这样子,cls_index.expand这个扩展时,-1表示不变,而d那个维度要被扩展到d长度
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            # XX：表示 hidden_states 的可选前导维度。这里的 XX 可以代表任意数量的前导维度，取决于实际的数据结构。例如，在多选
            # 题或多序列输入的情况下，可能会有多于一个的前导维度。
            # gather用-2维度(seq_len)上的token_idx,选择指定的表示,形状(b,options,1,d)
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
            # print(output)
        elif self.summary_type == "attn":
            raise NotImplementedError # 使用注意力机制选取隐藏状态（目前未实现）。
        output = self.first_dropout(output) # 应用 dropout 
        output = self.summary(output)  # 线性投影 #(b,options,classes)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output # 返回输出(分类或语义向量抽取)
# @add_start_docstrings(
#     """
# The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
# RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
# input embeddings, the classification head takes as input the input of a specified classification token index in the
# input sequence).
# """,
#     GPT2_START_DOCSTRING,
# )
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config) 
        config.num_labels = 1 
        self.transformer = GPT2Model(config) # gpt model
        # 预测下个token的线性层
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # Initialize weights and apply final processing
        self.post_init()
    # @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None): # 并行
        warnings.warn(
            "`GPT2DoubleHeadsModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should"
            " load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your"
            " own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'transformer.h.0': 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = ( # 给不同gpu分配不同的层
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map) # 把层分给不同gpu
        self.lm_head = self.lm_head.to(self.transformer.first_device) # gpu0
        self.multiple_choice_head = self.multiple_choice_head.to(self.transformer.first_device)
        self.model_parallel = True # 设置并行状态
    # @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.multiple_choice_head = self.multiple_choice_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mc_token_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mc_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, GPT2DoubleHeadsModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0] # (h,s,d)
        # Set device for model parallelism
        # 如果设置了并行,把数据传给lm_head的设备,让数据和模型的层在同一个设备上
        if self.model_parallel: 
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states) # (b,s,v)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1) 
        mc_loss = None
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous() # 模型预测的token
            shift_labels = labels[..., 1:].contiguous() # 真实token
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict: 
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    @staticmethod
    def _reorder_cache( # 更改past_key_values和beam_idx一致
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
@add_start_docstrings(
    """
    The GPT2 Model transformer with a sequence classification head on top (linear layer).

    [`GPT2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint="microsoft/DialogRPT-updown",
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
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

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)  # (b,s,num_labels)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        # 批次大小大于1时,必须要有填充 id
        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
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
                logger.warning_once(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        loss = None
        if labels is not None:
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
@add_start_docstrings(
    """
    GPT2 Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    GPT2_START_DOCSTRING,
)
class GPT2ForTokenClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPT2Model(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    # fmt: off
    @add_code_sample_docstrings(
        checkpoint="brad1141/gpt2-finetuned-comp2",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_loss=0.25,
        expected_output=[
            "Lead",
            "Lead",
            "Lead",
            "Position",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
        ],
    )
    # fmt: on
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    The GPT-2 Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2ForQuestionAnswering(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_CHECKPOINT_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

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
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
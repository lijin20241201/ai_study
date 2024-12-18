import keras
from keras_cv.layers import DropPath
from keras import ops
from keras import layers
import tensorflow as tf  # only for dataloader
from skimage.data import chelsea
import numpy as np
# SE模块是一种用于提高卷积神经网络性能的架构，它通过重新标定特征图（feature maps）的通道来工作，给予每个通道不同的重要性。
# GlobalAvgPool2D 用于对每个特征图（feature map）进行全局平均池化，从而得到一个形状为 (batch_size, channels, 1, 1) 
# 的输出。这里的 batch_size 是批量大小，channels 是输入特征图的通道数。由于 keepdims=True，输出的特征图在高度和宽度上
# 都被压缩到了1，但保留了通道数不变。
# 这里的 inp * self.expansion 是为了创建一个压缩（squeeze）后的中间层。inp 是输入特征图的通道数，self.expansion 是一
# 个小于1的因子（如0.25），用于减少这个中间层的维度。这种压缩的目的是减少模型的复杂度和计算量，同时保留足够的信息用于后续的激
# 发（excitation）过程。通过乘以 self.expansion，我们得到一个新的维度，这个维度小于原始的通道数，但足以捕获重要信息。
# sigmoid 函数可以在任何维度的输出上使用，不仅仅限于一维。sigmoid 函数的作用是将任意实值压缩到 (0, 1) 区间内，这里它
# 被用于生成每个通道的权重（或称为重要性）。self.output_dim 表示输出特征图的通道数，每个通道都会得到一个对应的 sigmoid 
# 输出值，这些值将被用于重新标定原始输入特征图的通道。
# 在SE模块中，使用 use_bias=False 是因为SE模块的设计目标是重新标定特征图的通道，而不是添加额外的偏移量。通过仅使用权重（
# 即卷积核或全连接层的参数）来调整特征图，我们可以更直接地控制特征图的重新标定过程
# 通过全局平均池化,线性变换,sigmoid,得到各个通道特征图的权重,最后返回加权后的特征图
class SqueezeAndExcitation(layers.Layer):
   
    def __init__(self, output_dim=None, expansion=0.25, **kwargs):
        super().__init__(**kwargs)
        self.expansion = expansion
        self.output_dim = output_dim

    def build(self, input_shape):
        inp = input_shape[-1]
        # 有output_dim就是它,否则是输入的最后一维
        self.output_dim = self.output_dim or inp 
        # avg_pool
        self.avg_pool = layers.GlobalAvgPool2D(keepdims=True, name="avg_pool")
        self.fc = [
            layers.Dense(int(inp * self.expansion), use_bias=False, name="fc_0"),
            layers.Activation("gelu", name="fc_1"),
            layers.Dense(self.output_dim, use_bias=False, name="fc_2"),
            layers.Activation("sigmoid", name="fc_3"),
        ]
        super().build(input_shape)
    # x是通过SE模块处理后的特征图，它包含了每个通道的重要性权重（这些权重是通过sigmoid函数得到的，值在0到1之间）。
    # 然后，这些权重被用来重新标定原始输入特征图的每个通道，即通过将x（权重）与inputs（原始特征图）进行逐元素相
    # 乘来实现。最后返回的是加权特征图
    def call(self, inputs, **kwargs):
        x = self.avg_pool(inputs)
        for layer in self.fc:
            x = layer(x)
        return x * inputs
class ReduceSize(layers.Layer):
    def __init__(self, keepdims=False, **kwargs):
        super().__init__(**kwargs)
        self.keepdims = keepdims

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        dim_out = embed_dim if self.keepdims else 2 * embed_dim
        self.pad1 = layers.ZeroPadding2D(1, name="pad1")
        self.pad2 = layers.ZeroPadding2D(1, name="pad2")
        # 提取特征块
        self.conv = [
            # 深度卷积,提取空间信息
            layers.DepthwiseConv2D(
                kernel_size=3, strides=1, padding="valid", use_bias=False, name="conv_0"
            ),
            layers.Activation("gelu", name="conv_1"),
            SqueezeAndExcitation(name="conv_2"),
            # 逐点卷积,混合通道信息
            layers.Conv2D(
                embed_dim,
                kernel_size=1,
                strides=1,
                padding="valid",
                use_bias=False,
                name="conv_3",
            ),
        ]
        # 缩小特征图大小块
        self.reduction = layers.Conv2D(
            dim_out,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            name="reduction",
        )
        self.norm1 = layers.LayerNormalization(
            -1, 1e-05, name="norm1"
        )  # eps like PyTorch
        self.norm2 = layers.LayerNormalization(-1, 1e-05, name="norm2")

    def call(self, inputs, **kwargs):
        # 标准化数据
        x = self.norm1(inputs) 
        xr = self.pad1(x)
        for layer in self.conv:
            xr = layer(xr)
        x = x + xr # 残差连接
        x = self.pad2(x)
        x = self.reduction(x)
        x = self.norm2(x)  # 标准化数据
        return x

class MLP(layers.Layer): # 线性投影块
    def __init__(
        self,
        hidden_features=None,
        out_features=None,
        activation="gelu",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.activation = activation
        self.dropout = dropout

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.hidden_features = self.hidden_features or self.in_features
        self.out_features = self.out_features or self.in_features
        self.fc1 = layers.Dense(self.hidden_features, name="fc1")
        self.act = layers.Activation(self.activation, name="act")
        self.fc2 = layers.Dense(self.out_features, name="fc2")
        self.drop1 = layers.Dropout(self.dropout, name="drop1")
        self.drop2 = layers.Dropout(self.dropout, name="drop2")

    def call(self, inputs, **kwargs):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# 输入填充（Padding）
# 目的：在进行卷积操作之前，对输入图像进行填充是为了确保卷积操作后的特征图尺寸不会减小
# 得太快，或者在某些情况下，为了保持特征图尺寸不变。实现：通过添加额外的像素边界到
# 输入图像的边缘来实现。
# 卷积提取嵌入（Patches with Embeddings）
# 目的：使用卷积操作将输入图像分割成多个小块（patches），并将每个小块转换为一个嵌入向量（embedding）。
# 这样，图像就被转换为了一个序列的嵌入向量，便于后续处理。实现：通过卷积层实现，卷积核的大小和步长决定
# 了提取的patches的大小和重叠程度。
# 特征提取与下采样（Feature Extraction + Downsampling）
# 目的：在提取patches的同时，可能还需要对特征进行进一步的提取和初步的下采样，以减少特征图的空间维度，同
# 时增加通道维度（即嵌入向量的维度），为后续处理做准备。实现：虽然您提到这个模块在提取特征时既不减少也不
# 增加空间维度，但通常在GCViT这样的架构中，会有某种形式的下采样（如通过步长大于1的卷积）来减少空间维度。
# 不过，也可能存在特殊设计的层（如Fused-MBConv），它们在保持空间维度不变的同时，通过增加通道数来“隐含
# 地”进行下采样。
# 重叠Patches
# 特点：GCViT与其他一些模型（如ViT或SwinTransformer）的一个显著区别是它创建了重叠的patches。这意味着
# 相邻的patches之间会有共享的部分，这有助于模型捕获更多的局部上下文信息。
# 实现：通过调整卷积操作的步长和填充来实现重叠。例如，如果卷积核大小为k，步长为s（小于k），则会产生重叠的
# patches。
# 空间维度减少
# 描述：您提到patch_embed调用了一个ReduceSizeConv2D函数，该函数通过卷积减少了输入的空间维度。这是通过
# 卷积层的步长（strides=2）大于1来实现的。
# 参数：kernel_size=3表示卷积核的大小为3x3，strides=2表示卷积时的步长为2，这意味着输出特征图的空间维
# 度将是输入的一半（在忽略填充的情况下）。

class PatchEmbed(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
    def build(self, input_shape):
        self.pad = layers.ZeroPadding2D(1, name="pad")
        # 普通卷积,步长2,会减小尺寸,use_bias默认True
        self.proj = layers.Conv2D(self.embed_dim, 3, 2, name="proj")
        # 包括卷积提取特征和下采样
        self.conv_down = ReduceSize(keepdims=True, name="conv_down")

    def call(self, inputs, **kwargs):
        x = self.pad(inputs)
        x = self.proj(x)
        x = self.conv_down(x)
        return x

# Global Token Generation模块的主要目的是从输入的特征图中提取全局信息，并将这些信息编码为一系列
# 的全局标记（tokens）。这些全局标记随后可以用于增强模型对图像全局上下文的理解，从而提高模型的性能。
# 特征提取（Feature Extraction）：
# 这个层与前面提到的其他特征提取模块相似，但有几个关键区别：
# 使用MaxPooling来减少特征图的空间维度，而不是通过增加通道数（即特征维度）来“隐含地”进行下采样。
# 不使用Layer Normalization。
# 生成全局标记（Generate Global Tokens）：
# 经过多次特征提取和维度减少后，模块会生成一个或多个全局标记。这些标记是通过对整个特征图进行池化（可
# 能是全局平均池化或全局最大池化）来获得的，从而捕获了图像的全局信息。
# 重要的是，这些全局标记在整个图像中是共享的，即整个图像只使用一组全局标记来表示其全局上下文。这种设计
# 显著减少了计算量，因为不需要为每个局部窗口单独生成全局标记。
# 扩展全局标记：
# 为了使全局标记能够与每个局部窗口的标记一起使用（在全局-局部上下文注意力机制中），需要将全局标记复制并
# 扩展到与局部窗口数量相匹配的维度。
# 如果图像被划分为M个局部窗口，那么全局标记将被复制M次，以便与每个局部窗口的标记相对应。
# 通过为每个输出通道学习不同的权重，逐点卷积层能够允许网络将来自不同输入通道的信息以非线性的方式组合起来。
# 这种组合方式是非常灵活的，因为每个输出通道都可以是输入通道中所有或某些通道信息的加权和。因此，逐点卷积
# 层能够有效地“混合”通道信息，使得网络能够学习更复杂的特征表示。

class FeatureExtraction(layers.Layer):
    def __init__(self, keepdims=False, **kwargs):
        super().__init__(**kwargs)
        self.keepdims = keepdims
    
    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.pad1 = layers.ZeroPadding2D(1, name="pad1")
        self.pad2 = layers.ZeroPadding2D(1, name="pad2")
        self.conv = [
            # 深度卷积,提取通道内的空间信息
            layers.DepthwiseConv2D(3, 1, use_bias=False, name="conv_0"),
            layers.Activation("gelu", name="conv_1"),
            SqueezeAndExcitation(name="conv_2"), # 加权特征图
            layers.Conv2D(embed_dim, 1, 1, use_bias=False, name="conv_3"), # 逐点卷积,混合通道信息
        ]
        if not self.keepdims:
            self.pool = layers.MaxPool2D(3, 2, name="pool")
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        xr = self.pad1(x)
        for layer in self.conv:
            xr = layer(xr)
        x = x + xr # 残差
        if not self.keepdims:
            x = self.pool(self.pad2(x))
        return x

class GlobalQueryGenerator(layers.Layer):
    
    def __init__(self, keepdims=False, **kwargs):
        super().__init__(**kwargs)
        self.keepdims = keepdims

    def build(self, input_shape):
        self.to_q_global = [
            FeatureExtraction(keepdims, name=f"to_q_global_{i}")
            for i, keepdims in enumerate(self.keepdims)
        ]
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.to_q_global:
            x = layer(x)
        return x

class WindowAttention(layers.Layer):

    def __init__(
        self,
        window_size,
        num_heads,
        global_query,
        qkv_bias=True,
        qk_scale=None,
        attention_dropout=0.0,
        projection_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        self.global_query = global_query
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout

    def build(self, input_shape):
        embed_dim = input_shape[0][-1] # d
        head_dim = embed_dim // self.num_heads # d_k
        self.scale = self.qk_scale or head_dim**-0.5
        self.qkv_size = 3 - int(self.global_query)
        self.qkv = layers.Dense(
            embed_dim * self.qkv_size, use_bias=self.qkv_bias, name="qkv"
        )
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=[
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ],
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype,
        )
        self.attn_drop = layers.Dropout(self.attention_dropout, name="attn_drop")
        self.proj = layers.Dense(embed_dim, name="proj")
        self.proj_drop = layers.Dropout(self.projection_dropout, name="proj_drop")
        self.softmax = layers.Activation("softmax", name="softmax")
        super().build(input_shape)

    def get_relative_position_index(self):
        coords_h = ops.arange(self.window_size[0])
        coords_w = ops.arange(self.window_size[1])
        coords = ops.stack(ops.meshgrid(coords_h, coords_w, indexing="ij"), axis=0)
        # 2表示有两个坐标轴，而 height_size * width_size 表示网格中所有点的总数。coords_flatten
        # 的每一行现在包含了网格上所有点的 X 或 Y 坐标
        coords_flatten = ops.reshape(coords, [2, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = ops.transpose(relative_coords, axes=[1, 2, 0])
        relative_coords_xx = relative_coords[:, :, 0] + self.window_size[0] - 1
        relative_coords_yy = relative_coords[:, :, 1] + self.window_size[1] - 1
        relative_coords_xx = relative_coords_xx * (2 * self.window_size[1] - 1)
        relative_position_index = relative_coords_xx + relative_coords_yy
        return relative_position_index

    def call(self, inputs, **kwargs):
        if self.global_query:
            inputs, q_global = inputs
            B = ops.shape(q_global)[0]  # B, N, C
        else:
            inputs = inputs[0]
        B_, N, C = ops.shape(inputs)  # B*num_window, num_tokens, channels
        qkv = self.qkv(inputs)
        qkv = ops.reshape(
            qkv, [B_, N, self.qkv_size, self.num_heads, C // self.num_heads]
        )
        qkv = ops.transpose(qkv, [2, 0, 3, 1, 4])
        if self.global_query:
            k, v = ops.split(
                qkv, indices_or_sections=2, axis=0
            )  # for unknown shame num=None will throw error
            q_global = ops.repeat(
                q_global, repeats=B_ // B, axis=0
            )  # num_windows = B_//B => q_global same for all windows in a img
            q = ops.reshape(q_global, [B_, N, self.num_heads, C // self.num_heads])
            q = ops.transpose(q, axes=[0, 2, 1, 3])
        else:
            q, k, v = ops.split(qkv, indices_or_sections=3, axis=0)
            q = ops.squeeze(q, axis=0)

        k = ops.squeeze(k, axis=0)
        v = ops.squeeze(v, axis=0)

        q = q * self.scale
        attn = q @ ops.transpose(k, axes=[0, 1, 3, 2])
        relative_position_bias = ops.take(
            self.relative_position_bias_table,
            ops.reshape(self.get_relative_position_index(), [-1]),
        )
        relative_position_bias = ops.reshape(
            relative_position_bias,
            [
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ],
        )
        relative_position_bias = ops.transpose(relative_position_bias, axes=[2, 0, 1])
        attn = attn + relative_position_bias[None,]
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = ops.transpose((attn @ v), axes=[0, 2, 1, 3])
        x = ops.reshape(x, [B_, N, C])
        x = self.proj_drop(self.proj(x))
        return x

# 当你调用 ops.meshgrid(coords_h, coords_w, indexing="ij") 时，你实际上是在为二维空间中的每个点生成坐标
# 对。这里的 coords_h 和 coords_w 分别代表高度（或垂直方向）和宽度（或水平方向）的坐标值列表。参数 indexing
# ="ij" 指定了坐标的索引方式，其中 "ij" 索引遵循 MATLAB 风格，即第一个维度是行（高度），第二个维度是列（宽度）。

class Block(layers.Layer):
    
    def __init__(
        self,
        window_size,
        num_heads,
        global_query,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        dropout=0.0,
        attention_dropout=0.0,
        path_drop=0.0,
        activation="gelu",
        layer_scale=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.num_heads = num_heads
        self.global_query = global_query
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.path_drop = path_drop
        self.activation = activation
        self.layer_scale = layer_scale

    def build(self, input_shape):
        B, H, W, C = input_shape[0]
        self.norm1 = layers.LayerNormalization(-1, 1e-05, name="norm1")
        self.attn = WindowAttention(
            window_size=self.window_size,
            num_heads=self.num_heads,
            global_query=self.global_query,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attention_dropout=self.attention_dropout,
            projection_dropout=self.dropout,
            name="attn",
        )
        self.drop_path1 = DropPath(self.path_drop)
        self.drop_path2 = DropPath(self.path_drop)
        self.norm2 = layers.LayerNormalization(-1, 1e-05, name="norm2")
        self.mlp = MLP(
            hidden_features=int(C * self.mlp_ratio),
            dropout=self.dropout,
            activation=self.activation,
            name="mlp",
        )
        if self.layer_scale is not None:
            self.gamma1 = self.add_weight(
                name="gamma1",
                shape=[C],
                initializer=keras.initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype,
            )
            self.gamma2 = self.add_weight(
                name="gamma2",
                shape=[C],
                initializer=keras.initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        self.num_windows = int(H // self.window_size) * int(W // self.window_size)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.global_query:
            inputs, q_global = inputs
        else:
            inputs = inputs[0]
        B, H, W, C = ops.shape(inputs)
        x = self.norm1(inputs)
        # create windows and concat them in batch axis
        x = self.window_partition(x, self.window_size)  # (B_, win_h, win_w, C)
        # flatten patch
        x = ops.reshape(x, [-1, self.window_size * self.window_size, C])
        # attention
        if self.global_query:
            x = self.attn([x, q_global])
        else:
            x = self.attn([x])
        # reverse window partition
        x = self.window_reverse(x, self.window_size, H, W, C)
        # FFN
        x = inputs + self.drop_path1(x * self.gamma1)
        x = x + self.drop_path2(self.gamma2 * self.mlp(self.norm2(x)))
        return x

    def window_partition(self, x, window_size):
       
        B, H, W, C = ops.shape(x)
        x = ops.reshape(
            x,
            [
                -1,
                H // window_size,
                window_size,
                W // window_size,
                window_size,
                C,
            ],
        )
        x = ops.transpose(x, axes=[0, 1, 3, 2, 4, 5])
        windows = ops.reshape(x, [-1, window_size, window_size, C])
        return windows

    def window_reverse(self, windows, window_size, H, W, C):
        
        x = ops.reshape(
            windows,
            [
                -1,
                H // window_size,
                W // window_size,
                window_size,
                window_size,
                C,
            ],
        )
        x = ops.transpose(x, axes=[0, 1, 3, 2, 4, 5])
        x = ops.reshape(x, [-1, H, W, C])
        return x

class Level(layers.Layer):
    
    def __init__(
        self,
        depth,
        num_heads,
        window_size,
        keepdims,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        dropout=0.0,
        attention_dropout=0.0,
        path_drop=0.0,
        layer_scale=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.keepdims = keepdims
        self.downsample = downsample
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.path_drop = path_drop
        self.layer_scale = layer_scale

    def build(self, input_shape):
        path_drop = (
            [self.path_drop] * self.depth
            if not isinstance(self.path_drop, list)
            else self.path_drop
        )
        self.blocks = [
            Block(
                window_size=self.window_size,
                num_heads=self.num_heads,
                global_query=bool(i % 2),
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                path_drop=path_drop[i],
                layer_scale=self.layer_scale,
                name=f"blocks_{i}",
            )
            for i in range(self.depth)
        ]
        self.down = ReduceSize(keepdims=False, name="downsample")
        self.q_global_gen = GlobalQueryGenerator(self.keepdims, name="q_global_gen")
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        q_global = self.q_global_gen(x)  # shape: (B, win_size, win_size, C)
        for i, blk in enumerate(self.blocks):
            if i % 2:
                x = blk([x, q_global])  # shape: (B, H, W, C)
            else:
                x = blk([x])  # shape: (B, H, W, C)
        if self.downsample:
            x = self.down(x)  # shape: (B, H//2, W//2, 2*C)
        return x

class GCViT(keras.Model):
    def __init__(
        self,
        window_size,
        embed_dim,
        depths,
        num_heads,
        drop_rate=0.0,
        mlp_ratio=3.0,
        qkv_bias=True,
        qk_scale=None,
        attention_dropout=0.0,
        path_drop=0.1,
        layer_scale=None,
        num_classes=11,
        head_activation="softmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attention_dropout = attention_dropout
        self.path_drop = path_drop
        self.layer_scale = layer_scale
        self.num_classes = num_classes
        self.head_activation = head_activation

        self.patch_embed = PatchEmbed(embed_dim=embed_dim, name="patch_embed")
        self.pos_drop = layers.Dropout(drop_rate, name="pos_drop")
        path_drops = np.linspace(0.0, path_drop, sum(depths))
        keepdims = [(0, 0, 0), (0, 0), (1,), (1,)]
        self.levels = []
        for i in range(len(depths)):
            path_drop = path_drops[sum(depths[:i]) : sum(depths[: i + 1])].tolist()
            level = Level(
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                keepdims=keepdims[i],
                downsample=(i < len(depths) - 1),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                dropout=drop_rate,
                attention_dropout=attention_dropout,
                path_drop=path_drop,
                layer_scale=layer_scale,
                name=f"levels_{i}",
            )
            self.levels.append(level)
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-05, name="norm")
        self.pool = layers.GlobalAvgPool2D(name="pool")
        # 分类层
        self.head = layers.Dense(num_classes, name="head", activation=head_activation)

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x = self.patch_embed(inputs)  # shape: (B, H, W, C)
        x = self.pos_drop(x)
        # [<Level name=levels_0, built=False>, <Level name=levels_1, built=False>, 
        # <Level name=levels_2, built=False>, <Level name=levels_3, built=False>]
        # print(self.levels)
        for level in self.levels:
            x = level(x)  # shape: (B, H_, W_, C_)
        x = self.norm(x)
        x = self.pool(x)  # shape: (B, C__)
        x = self.head(x)
        return x

    def build_graph(self, input_shape=(224, 224, 3)):
       
        x = keras.Input(shape=input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x), name=self.name)

    def summary(self, input_shape=(224, 224, 3)):
        return self.build_graph(input_shape).summary()


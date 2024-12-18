def selective_kernel_feature_fusion(
    multi_scale_feature_1, multi_scale_feature_2, multi_scale_feature_3
):
    channels = list(multi_scale_feature_1.shape)[-1] # d
    # add 层用于将三个多尺度特征图（multi_scale_feature_1、multi_scale_feature_2、multi_scale_feature_3）
    # 相加，生成一个组合的特征图 combined_feature。实现了一种特征融合的效果，其中每个特征图的贡献都是相加的。
    combined_feature = layers.Add()( 
        [multi_scale_feature_1, multi_scale_feature_2, multi_scale_feature_3]
    )
    gap = layers.GlobalAveragePooling2D()(combined_feature) # 全局池化
    # 变形
    channel_wise_statistics = layers.Reshape((1, 1, channels))(gap)
    compact_feature_representation = layers.Conv2D( # 压缩
        filters=channels // 8, kernel_size=(1, 1), activation="relu"
    )(channel_wise_statistics)
    # 虽然这三个特征描述符结构相同,但是在训练时权重却会不同,因为他们分别和不同
    # 特征图加权,生成全局特征描述符,对通道求softmax会为各个通道分配概率分数,
    # 但是加和是1,主要是标记各个通道的重要性
    # 使用 softmax 激活函数生成通道描述符时，每个描述符中的通道值之和将为 1。这意味着每个描述符
    # 都表示了一个概率分布，其中每个通道的重要性是相对于其他通道而言的。这可能会引入一种竞争机制，
    # 其中一些通道的重要性可能会增加，而其他通道的重要性则会相应减少。如果您想要的是每个通道都有一
    # 个独立的“重要性”分数（而不是相对重要性），那么使用 sigmoid 激活函数可能更为合适
    feature_descriptor_1 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    feature_descriptor_2 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    feature_descriptor_3 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    # 将描述符与特征图相乘，以实现基于通道重要性的特征重标定,在训练中,上面的卷积
    # 会生成不同的权重
    # 权重在训练开始前是随机初始化的。这意味着对于每个 Conv2D 层（即使它们具有相同的配置，如相同的输入
    # 通道数、输出通道数、卷积核大小等），它们的权重都会独立地随机初始化。
    # 在训练过程中，这些权重会根据各自对应的损失函数的梯度进行更新。由于每个 Conv2D 层都连接到了不同的 
    # multi_scale_feature_x（其中 x 是 1, 2, 或 3），并且这些特征图在训练过程中会接收到不同的输入和
    # 梯度信号，因此它们的权重将独立地演变，并最终收敛到不同的值
    feature_1 = multi_scale_feature_1 * feature_descriptor_1
    feature_2 = multi_scale_feature_2 * feature_descriptor_2
    feature_3 = multi_scale_feature_3 * feature_descriptor_3
    # 聚合特征，将三个残差
    aggregated_feature = layers.Add()([feature_1, feature_2, feature_3])
    return aggregated_feature
# 虽然 SKFF 块在多分辨率分支之间融合信息，但我们还需要一个 在特征张量内共享信息的机制，包括空间和 channel 维度，
# 该维度由 DAU 块完成。DAU 抑制 Less 用处 功能，并且只允许信息量更大的信息进一步传递。此功能 通过使用 Channel 
# Attention 和 Spatial Attention 机制实现重新校准。
# Channel Attention 分支利用了 通过应用 squeeze 和 excitation 操作进行卷积特征映射。给定一个特征 map 中，
# squeeze 操作将跨空间维度的 Global Average Pooling 应用于 对全局上下文进行编码，从而生成特征描述符。激励运
# 算符将 这个特征描述符通过两个卷积层，然后是 sigmoid 门控 并生成激活。最后，Channel Attention 分支的输出由
# 使用输出激活重新缩放输入特征图。
# Spatial Attention 分支旨在利用 卷积特征。Spatial Attention 的目标是生成空间注意力 map 并使用它来重新校
# 准传入的要素。生成空间注意力 map 中，Spatial Attention 分支首先独立应用 Global Average Pooling，然后 
# 沿通道维度和连接对输入特征的最大池化操作 输出以形成合成特征图，然后通过卷积传递 和 sigmoid 激活以获得空间注意
# 力图。这个空间注意力图是 然后用于重新缩放输入特征图。
class ChannelPooling(layers.Layer):
    def __init__(self, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis
        self.concat = layers.Concatenate(axis=self.axis)

    def call(self, inputs):
        # 在特征轴聚合,之后因为聚合后少一维,就需要在后边加一维
        # 对特征聚合求均值,对通道池化
        average_pooling = tf.expand_dims(tf.reduce_mean(inputs, axis=-1), axis=-1)
        # 对特征聚合求最大值
        max_pooling = tf.expand_dims(tf.reduce_max(inputs, axis=-1), axis=-1)
        return self.concat([average_pooling, max_pooling])

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
# 空间注意力
def spatial_attention_block(input_tensor):
    # 将多维特征图压缩成二维的，从而可以在空间维度上应用注意力机制
    compressed_feature_map = ChannelPooling(axis=-1)(input_tensor) # 对特征的聚合
    # 使用一个Conv2D层将压缩后的特征图（现在是二维的）转换成一个单一的通道图，这个图表示了每
    # 个空间位置的重要性。这里使用了1x1的卷积核，因为不需要改变空间维度，只需要调整通道数。
    # 保持空间维度,跨通道信息融合
    feature_map = layers.Conv2D(1, kernel_size=(1, 1))(compressed_feature_map)
    # 将注意力图（feature_map）与原始输入特征图（input_tensor）相乘，以调整每个空间位置的重要性。
    # sigmoid会求出每个hxw位置的分数,这个分数反应了当前位置的重要性
    feature_map = keras.activations.sigmoid(feature_map) 
    return input_tensor * feature_map # 返回加权值
# 通道注意力,这种机制通常用于增强或抑制输入特征图中的特定通道，以提高网络对重要特征的关注度。
def channel_attention_block(input_tensor):
    channels = list(input_tensor.shape)[-1]
    # 使用全局平均池化来压缩空间维度，只保留通道信息
    average_pooling = layers.GlobalAveragePooling2D()(input_tensor)
    # 重塑池化后的输出，以匹配原始特征图的形状，但空间维度为1x1  
    feature_descriptor = layers.Reshape((1, 1, channels))(average_pooling)
    # 使用1x1卷积来压缩通道数，并应用ReLU激活函数,这一步是为了提取更紧凑的通道特征描述  
    feature_activations = layers.Conv2D(
        filters=channels // 8, kernel_size=(1, 1), activation="relu"
    )(feature_descriptor)
    # 再次使用1x1卷积将通道数恢复到原始大小，并应用sigmoid激活函数  
    # sigmoid输出值在0到1之间，作为每个通道重要性的权重  
    feature_activations = layers.Conv2D(
        filters=channels, kernel_size=(1, 1), activation="sigmoid"
    )(feature_activations)
    # 将学习到的通道权重与原始输入特征图相乘，实现通道的重新标定 
    return input_tensor * feature_activations
# 双注意力单元块
def dual_attention_unit_block(input_tensor):
    channels = list(input_tensor.shape)[-1]
    feature_map = layers.Conv2D(
        channels, kernel_size=(3, 3), padding="same", activation="relu"
    )(input_tensor)
    # feature_map = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(
    #     feature_map
    # )
    channel_attention = channel_attention_block(feature_map) # 通道注意力
    spatial_attention = spatial_attention_block(feature_map) # 空间注意力
    # 合并特征
    concatenation = layers.Concatenate(axis=-1)([channel_attention, spatial_attention])
    concatenation = layers.Conv2D(channels, kernel_size=(1, 1))(concatenation)
    return layers.Add()([input_tensor, concatenation])
# Multi-Scale Residual Block 能够通过以下方式生成空间精确的输出 保持高分辨率表示，同时接收丰富的上下文信息 
# 从低分辨率。MRB 由多个（本文中为 3 个）组成 并行连接的全卷积流。它允许跨 parallel streams 以便在 低分辨
# 率功能，反之亦然。MIRNet 采用递归残差设计 （使用跳过连接）来简化学习过程中的信息流。在 为了保持我们架构的残
# 余性质，残差大小调整模块是 用于执行 Multi-scale 中使用的缩减采样和上采样操作 残差块。
def down_sampling_module(input_tensor):
    channels = list(input_tensor.shape)[-1]
    main_branch = layers.Conv2D(channels, kernel_size=(1, 1), activation="relu")(
        input_tensor
    )
    # main_branch = layers.Conv2D(
    #     channels, kernel_size=(3, 3), padding="same", activation="relu"
    # )(main_branch)
    main_branch = layers.MaxPooling2D()(main_branch)
    main_branch = layers.Conv2D(channels * 2, kernel_size=(1, 1))(main_branch)
    skip_branch = layers.MaxPooling2D()(input_tensor)
    skip_branch = layers.Conv2D(channels * 2, kernel_size=(1, 1))(skip_branch)
    return layers.Add()([skip_branch, main_branch])
def up_sampling_module(input_tensor):
    channels = list(input_tensor.shape)[-1]
    main_branch = layers.Conv2D(channels, kernel_size=(1, 1), activation="relu")(
        input_tensor
    )
    # main_branch = layers.Conv2D(
    #     channels, kernel_size=(3, 3), padding="same", activation="relu"
    # )(main_branch)
    main_branch = layers.UpSampling2D()(main_branch)
    main_branch = layers.Conv2D(channels // 2, kernel_size=(1, 1))(main_branch)
    skip_branch = layers.UpSampling2D()(input_tensor)
    skip_branch = layers.Conv2D(channels // 2, kernel_size=(1, 1))(skip_branch)
    return layers.Add()([skip_branch, main_branch])
# MRB Block
def multi_scale_residual_block(input_tensor, channels):
    # 特征图
    level1 = input_tensor
    level2 = down_sampling_module(input_tensor)
    level3 = down_sampling_module(level2)
    # DAU,空间注意力和通道注意力
    level1_dau = dual_attention_unit_block(level1)
    level2_dau = dual_attention_unit_block(level2)
    level3_dau = dual_attention_unit_block(level3)
    # SKFF，把不同尺寸的特征图传人全局注意力模块
    # level1是比较当前,和小尺寸的
    # 对每个单独尺寸的特征图skff
    level1_skff = selective_kernel_feature_fusion(
        level1_dau,
        up_sampling_module(level2_dau),
        up_sampling_module(up_sampling_module(level3_dau)),
    )
    
    level2_skff = selective_kernel_feature_fusion(
        down_sampling_module(level1_dau),
        level2_dau,
        up_sampling_module(level3_dau),
    )
    level3_skff = selective_kernel_feature_fusion(
        down_sampling_module(down_sampling_module(level1_dau)),
        down_sampling_module(level2_dau),
        level3_dau,
    )
    # DAU 2
    level1_dau_2 = dual_attention_unit_block(level1_skff)
    level2_dau_2 = up_sampling_module((dual_attention_unit_block(level2_skff)))
    level3_dau_2 = up_sampling_module(
        up_sampling_module(dual_attention_unit_block(level3_skff))
    )
    # SKFF 2
    skff_ = selective_kernel_feature_fusion(level1_dau_2, level2_dau_2, level3_dau_2)
    conv = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(skff_)
    return layers.Add()([input_tensor, conv])
def recursive_residual_group(input_tensor, num_mrb, channels):
    conv1 = layers.SeparableConv2D(channels, kernel_size=(3, 3), padding="same")(input_tensor)
    for _ in range(num_mrb):
        conv1 = multi_scale_residual_block(conv1, channels)
    conv2 = layers.SeparableConv2D(channels, kernel_size=(3, 3), padding="same")(conv1)
    return layers.Add()([conv2, input_tensor])
def mirnet_model(num_rrg, num_mrb, channels):
    # input_tensor = keras.Input(shape=[None, None, 3]) 可以对不同大小图片推理
    input_tensor = keras.Input(shape=[IMAGE_SIZE,IMAGE_SIZE,3])
    x1 = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(input_tensor)
    for _ in range(num_rrg):
        x1 = recursive_residual_group(x1, num_mrb, channels)
    conv = layers.SeparableConv2D(3, kernel_size=(3, 3), padding="same")(x1)
    output_tensor = layers.Add()([input_tensor, conv])
    return keras.Model(input_tensor, output_tensor)
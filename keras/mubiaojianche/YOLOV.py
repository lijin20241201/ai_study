def get_tensor_input_name(tensor):
    if keras_3():
        return tensor._keras_history.operation.name
    else:
        return tensor.node.layer.name

class classproperty(property):
    def __get__(self, _, owner_cls):
        return self.fget(owner_cls)
def format_docstring(**replacements):
    def decorate(obj):
        doc = obj.__doc__
        # We use `str.format()` to replace variables in the docstring, but use
        # double brackets, e.g. {{var}}, to mark format strings. So we need to
        # to swap all double and single brackets in the source docstring.
        doc = "{".join(part.replace("{", "{{") for part in doc.split("{{"))
        doc = "}".join(part.replace("}", "}}") for part in doc.split("}}"))
        obj.__doc__ = doc.format(**replacements)
        return obj
    return decorate
def parse_model_inputs(input_shape, input_tensor, **kwargs):
    if input_tensor is None:
        return keras.layers.Input(shape=input_shape, **kwargs)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return keras.layers.Input(
                tensor=input_tensor, shape=input_shape, **kwargs
            )
        else:
            return input_tensor
# 下采样中的填充
def correct_pad_downsample(inputs, kernel_size):
    img_dim = 1
    input_size = inputs.shape[img_dim : (img_dim + 2)] # (h,w)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None: # 在设置可变输入模型时有用
        adjust = (1, 1)
    else:
        # 根据输入尺寸的奇偶性来调整一个很小的值（0 或 1）
        # 当输入尺寸是偶数时,adjust=(1,1),是奇数时,adjust=(0,0)
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    
    correct = (kernel_size[0] // 2, kernel_size[1] // 2) 
    # 这种填充用于下采样时,correct 变量计算了卷积核大小的一半（向下取整）
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )
# 卷积块(可以是普通卷积,也可以是逐点卷积)
def conv_bn(inputs,filters,kernel_size=1,strides=1,activation="swish",name="conv_bn",):
    if kernel_size > 1: # 核大小>1时,填充
        inputs = keras.layers.ZeroPadding2D(
            padding=kernel_size // 2, name=f"{name}_pad"
        )(inputs)
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides,
        use_bias=False,
        name=f"{name}_conv",
    )(inputs)
    x = keras.layers.BatchNormalization(
        momentum=0.97,
        epsilon=1e-3,
        name=f"{name}_bn",
    )(x)
    x = keras.layers.Activation(activation,name=name)(x)
    return x
# 空间金字塔池化（SPPF）,一般在特征图较小时做
# inputs:输入的特征图，预期是一个四维张量（b, h, w, c）
#  pool_size:池化窗口的大小，默认为5。
def spatial_pyramid_pooling_fast(
    inputs, pool_size=5, activation="swish",name="spp_fast"
):
    channel_axis = -1
    input_channels = inputs.shape[channel_axis] # c
    hidden_channels = int(input_channels // 2) # c/2 
    x = conv_bn(  # 点卷积切换到c/2通道大小（b, h, w, c/2）
        inputs,
        hidden_channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_pre",
    )
    # 这里池化操作不会下采样
    # 空间金字塔池化是一种用于卷积神经网络的池化技术，它可以在不同尺度上聚合特征，从而增强模型对尺度变化的鲁棒性。
    # 每次池化操作都从不同的尺度（或说不同的感受野）上提取了特征。第一次池化直接从原始特征图中提取，
    # 第二次和第三次池化则分别在前一次池化的结果上进行，因此它们捕捉到了更大尺度的特征。由于使用了最
    # 大值池化，这些操作倾向于保留最显著的特征，即那些在某个局部区域内具有最大值的特征。
    # 从池化1到3提取的特征会越来越明显,因为后者是在前者的5x5区域取特征
    # 尺度是通过在不同大小的池化窗口上操作来实现的。虽然这些池化窗口在物理上（即像素级别）是重叠的（
    # 因为步长是1），但它们捕获了输入特征图上不同“感受野”的局部信息。因此，每次池化
    # 操作都可以被视为在一个特定的尺度上提取特征。
    # 第一次是在原始特征图上5x5的最大值池化,第二次是在这个池化后的输出在继续最大值池化,
    # 第三次又是在第二次池化的基础上最大值池化,其实就相当于模型用了个更大的核在原始特征图
    # 上最大池化,所以他们虽然空间尺寸相同,但是数据区别很大,可以说线条特征一次比一次清晰
    pool_1 = keras.layers.MaxPooling2D( # （b, h, w, c/2）
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool1"
    )(x)
    pool_2 = keras.layers.MaxPooling2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool2"
    )(pool_1)
    pool_3 = keras.layers.MaxPooling2D( 
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool3"
    )(pool_2)
    # 原始特征图和三次池化后的特征图被沿着通道轴拼接起来。这样做的好处是，模型可以同时利用
    # 到不同尺度的特征信息，这对于提高模型的泛化能力和性能通常是有益的。（b, h, w, 2c）
    out = ops.concatenate([x, pool_1, pool_2, pool_3], axis=channel_axis)
    out = conv_bn( # 切换回原通道数（b, h, w,c）
        out,
        input_channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_output",
    )
    return out
# 残差连接,负责提取特征
def csp_block(
    inputs,
    channels=-1,
    depth=2,
    shortcut=True,
    expansion=0.5,# 扩张
    activation="swish",
    name="csp_block",
):
    channel_axis = -1
    channels = channels if channels > 0 else inputs.shape[channel_axis] # c
    hidden_channels = int(channels * expansion) # c/2
    # 点卷积块,用于切换通道
    pre = conv_bn( # (b,h,w,c)
        inputs,
        hidden_channels * 2,
        kernel_size=1,
        activation=activation,
        name=f"{name}_pre",
    )
    # short 和 deep 将是两个新的张量,它们分别是 pre 沿着最后一个维度被均匀分割成
    # 的两部分
    short, deep = ops.split(pre, 2, axis=channel_axis) # (b,h,w,c/2)
    out = [short, deep] # out中的out[-1]一直没变
    for id in range(depth):  # 遍历两次
        deep = conv_bn( # 卷积块1,改变的一直是deep指向的对象
            deep,
            hidden_channels,
            kernel_size=3,
            activation=activation,
            name=f"{name}_pre_{id}_1",
        )
        deep = conv_bn( # 卷积块2
            deep,
            hidden_channels,
            kernel_size=3,
            activation=activation,
            name=f"{name}_pre_{id}_2",
        )
        # 如果设置了使用残差,就残差,否则是deep
        deep = (out[-1] + deep) if shortcut else deep
        out.append(deep) # 把卷积后特征图加进out,加两次
    # depth要是2的话,这个形状变成(b,h,w,2c),在最后一个轴合并特征
    out = ops.concatenate(out, axis=channel_axis)
    out = conv_bn( # 用点卷积变回原通道数(b,h,w,c) 
        out,
        channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_output",
    )
    return out
def get_YOLOV8Backbone(stackwise_channels,stackwise_depth,input_shape=(None, None, 3),
                       input_tensor=None,include_rescaling=True,activation="swish"):
    # 模型输入(b,h,w,c)
    inputs = parse_model_inputs(input_shape, input_tensor)
    x = inputs
    if include_rescaling: 
        x = keras.layers.Rescaling(1 / 255.0)(x) # (归一化)
    stem_width = stackwise_channels[0] # 取列表中的第一个
    x = conv_bn( # 下采样普通卷积(b,h/2,w/2,c/2)
        x,
        stem_width // 2,
        kernel_size=3,
        strides=2,
        activation=activation,
        name="stem_1",
    )
    x = conv_bn( # 下采样普通卷积(b,h/4,w/4,c)
        x,
        stem_width,
        kernel_size=3,
        strides=2,
        activation=activation,
        name="stem_2",
    )
    # 用来存储金字塔特征图
    pyramid_level_inputs = {"P1": get_tensor_input_name(x)}
    # (channel, depth)
    for stack_id, (channel, depth) in enumerate(
        zip(stackwise_channels, stackwise_depth)
    ):
        # 除去第一个迭代,因为第一次的在前面已经处理过了
        stack_name = f"stack{stack_id + 1}"
        if stack_id >= 1:
            x = conv_bn( # 下采样普通卷积
                x,
                channel,
                kernel_size=3,
                strides=2,
                activation=activation,
                name=f"{stack_name}_downsample",
                
            )
        x = csp_block( # 精提特征
            x,
            depth=depth,
            expansion=0.5,
            activation=activation,
            name=f"{stack_name}_c2f",
        )
        # 在最后一次迭代时(SPPF:空间金字塔池化)
        if stack_id == len(stackwise_depth) - 1:
            x = spatial_pyramid_pooling_fast(
                x,
                pool_size=5,
                activation=activation,
                name=f"{stack_name}_spp_fast",
            )
        pyramid_level_inputs[f"P{stack_id + 2}"] = (
               get_tensor_input_name(x)
            )
    return keras.Model(inputs,x),pyramid_level_inputs
# 生成一系列锚框（anchor boxes），这些锚框在目标检测等任务中用于预测不同大小和宽高比的边界框。
def get_anchors(
    image_shape, # 输入特征图的形状，例如(height, width)
    strides=[8, 16, 32], # 锚框生成时在不同层级上使用的步长 
    base_anchors=[0.5, 0.5],
):
    base_anchors = ops.array(base_anchors, dtype="float32")
    all_anchors = [] # 初始化一个空列表，用于存储所有生成的锚点 
    all_strides = [] # 用于存储与每个锚点对应的步长
    for stride in strides: # 遍历所有指定的步长 
        # 生成高度和宽度上的网格点坐标
        hh_centers = ops.arange(0, image_shape[0], stride)
        ww_centers = ops.arange(0, image_shape[1], stride)
        # 使用meshgrid生成网格坐标点 
        ww_grid, hh_grid = ops.meshgrid(ww_centers, hh_centers)
        # 将x和y坐标堆叠并重塑为(b, 1, 2)的形状，其中b是网格点的总数 
         # 将网格坐标点转换为适合的形状,这里堆叠时是(y,x)的形式
        grid = ops.cast(
            ops.reshape(ops.stack([hh_grid, ww_grid], 2), [-1, 1, 2]),
            "float32",
        )
        # 获取当前特征图上的所有锚框中心点
        anchors = (
            ops.expand_dims(
                base_anchors * ops.array([stride, stride], "float32"), 0
            )
            + grid
        )
        anchors = ops.reshape(anchors, [-1, 2]) # (b,2)
        all_anchors.append(anchors)
        # 添加所有步长,每个锚点对应一个步长  
        all_strides.append(ops.repeat(stride, anchors.shape[0]))
    # 合并所有步长下的锚框坐标,以及对应的步长
    all_anchors = ops.cast(ops.concatenate(all_anchors, axis=0), "float32")
    all_strides = ops.cast(ops.concatenate(all_strides, axis=0), "float32")
    # 这样所有的锚点变成指定特征图下的坐标形式
    all_anchors = all_anchors / all_strides[:, None]
    # 交换锚框坐标的x和y坐标,1表示x,0表示y,None是增加一维,在最后一个轴合并,
    # 坐标变成(x,y)
    all_anchors = ops.concatenate(
        [all_anchors[:, 1, None], all_anchors[:, 0, None]], axis=-1
    )
    return all_anchors, all_strides
# apply_path_aggregation_fpn 方法名中的“path aggregation”可能指的是在 FPN 
# 架构中引入了一种新的特征聚合策略，这种策略可能不仅仅依赖于传统的自底向上和自顶向下
# 的路径，还可能包括横向连接、跨层连接或某种形式的特征增强路径。这种方法可能旨在进一
# 步提高特征金字塔中特征表示的质量，从而提升模型的性能。
# FPN 是一种用于目标检测、语义分割等任务的网络结构，它通过构建一个多尺度的特征金字塔来
# 有效地利用不同层次的特征信息。
# 通常用于目标检测任务中以融合不同尺度的特征图。在YOLO这样的检测框架中，FPN有助
# 于提升小物体检测的能力，因为它允许网络利用更浅层的特征（这些特征保留了更多细节
# 信息）来补充深层的语义信息。fpn有利于各个特征图间的信息互补
# P3 特征图的空间尺寸较大，保留了更多的局部细节信息。
def apply_path_aggregation_fpn(features, depth=3,name="fpn"):
    p3, p4, p5 = features
    # 上采样P5特征图,通过简单的复制轴上的数据两次来让尺寸变为两倍
    p5_upsampled = ops.repeat(ops.repeat(p5, 2, axis=1), 2, axis=2)
    # 合并特征
    p4p5 = ops.concatenate([p5_upsampled, p4], axis=-1) # (b,h,w,c_hb)
    p4p5 = csp_block( # 提取特征
        p4p5,
        channels=p4.shape[-1],
        depth=depth,
        shortcut=False, # 是否残差,这里False
        activation="swish",
        name=f"{name}_p4p5",
    )
    # 上采样P4P5,和P3合并特征,然后用CSPBlock提前特征.
    p4p5_upsampled = ops.repeat(ops.repeat(p4p5, 2, axis=1), 2, axis=2)
    p3p4p5 = ops.concatenate([p4p5_upsampled, p3], axis=-1) # 合并特征
    p3p4p5 = csp_block( # # 在p3级别的尺寸上提取特征
        p3p4p5,
        channels=p3.shape[-1],
        depth=depth,
        shortcut=False,
        activation="swish",
        name=f"{name}_p3p4p5",
    )
    p3p4p5_d1 = conv_bn( # 下采样
        p3p4p5,
        p3p4p5.shape[-1],
        kernel_size=3,
        strides=2,
        activation="swish",
        name=f"{name}_p3p4p5_downsample1",
    )
    # 和p4p5合并特征
    p3p4p5_d1 = ops.concatenate([p3p4p5_d1, p4p5], axis=-1)
    # 这个尺寸的特征图上提取特征
    p3p4p5_d1 = csp_block( 
        p3p4p5_d1,
        channels=p4p5.shape[-1],
        shortcut=False,
        activation="swish",
        name=f"{name}_p3p4p5_downsample1_block",
    )
    # 下采样
    p3p4p5_d2 =conv_bn(
        p3p4p5_d1,
        p3p4p5_d1.shape[-1],
        kernel_size=3,
        strides=2,
        activation="swish",
        name=f"{name}_p3p4p5_downsample2",
    )
    # 和p5尺寸的特征图特征合并
    p3p4p5_d2 = ops.concatenate([p3p4p5_d2, p5], axis=-1)
    # 在这个尺寸的特征图提取特征(默认depth=2)
    p3p4p5_d2 = csp_block(
        p3p4p5_d2,
        channels=p5.shape[-1],
        shortcut=False,
        activation="swish",
        name=f"{name}_p3p4p5_downsample2_block",
    )
    #  # p3p4p5是融合了较抽象特征图特征的较大特征图
    # p3p4p5_d1是下采样时融合,p3p4p5_d2是尺寸最小的特征图
    return [p3p4p5, p3p4p5_d1, p3p4p5_d2]
def apply_yolo_v8_head(
    inputs, # 不同尺寸的特征图
    num_classes,
    name="yolo_v8_head",
):
    box_channels = max(64, inputs[0].shape[-1] // 4) # max(64,c/4),64
    class_channels = max(num_classes, inputs[0].shape[-1]) # max(num_classes,c),64
    # print(box_channels,class_channels)
    outputs = [] # 对应不同尺寸特征图的输出
    # 这里的feature对应提取的特征图
    for id, feature in enumerate(inputs):
        cur_name = f"{name}_{id+1}"
        # 边界框预测
        box_predictions = conv_bn( # 卷积块1
            feature,
            box_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_box_1",
        )
        box_predictions = conv_bn(# 卷积块2
            box_predictions,
            box_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_box_2",
        )
        box_predictions = keras.layers.Conv2D( # (...,64),点卷积
            filters=64,
            kernel_size=1,
            name=f"{cur_name}_box_3_conv",
        )(box_predictions)
        # 类别预测
        class_predictions = conv_bn(# 卷积块1
            feature,
            class_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_class_1",
        )
        class_predictions = conv_bn(# 卷积块2
            class_predictions,
            class_channels,
            kernel_size=3,
            activation="swish",
            name=f"{cur_name}_class_2",
        )
        class_predictions = keras.layers.Conv2D( # 点卷积(...,num_classes)
            filters=num_classes,
            kernel_size=1,
            name=f"{cur_name}_class_3_conv",
        )(class_predictions)
        # (None, 28, 28, 64) (None, 28, 28, 8)
        # print(box_predictions.shape,class_predictions.shape)
        # 转换成对特征维度求sigmoid,
        class_predictions = keras.layers.Activation("sigmoid",
                           name=f"{cur_name}_classifier")(class_predictions)
        # print(class_predictions.shape) (None, 28, 28, 8) 
        # 在最后一个轴合并
        out = ops.concatenate([box_predictions, class_predictions], axis=-1)
        # print(out.shape) (None, 28, 28, 72)
        # 因为batch_size是None,所以在变形时是不会用到b的,(None, 784, 72),
        # (None, 196, 72),(None, 49, 72)
        # 将二维的网格坐标摊平成了一维的向量。这些点，在变形后的一维向量中，可以被理解为
        # 网格特征点,对应锚点
        out = keras.layers.Reshape( # (n,c)
            [-1, out.shape[-1]],name=f"{cur_name}_output_reshape"
        )(out)
        # print(out.shape) (None, 784, 72)
        outputs.append(out)
    # print(len(outputs)) 3
    # 在索引1的轴合并,就是在空间位置摊平的那个轴,这样得到的是每个单位点对应特征的集合
    # (None, 1029, 72),这些点的集合对应锚框中心点
    outputs = ops.concatenate(outputs, axis=1)
    # print(outputs.shape)
    # keras.layers.Activation("linear") 和在层定义中不显式指定激活函数（
    # 即使用 None 或完全不写激活函数）在功能上是相似的，但它们在表达意图和可
    # 读性方面有所不同。
    outputs = keras.layers.Activation( # (None, 1029, 72)
        "linear", dtype="float32",name="box_outputs"
    )(outputs)
    # print(outputs.shape)
    return {
        "boxes": outputs[:, :, :64],
        "classes": outputs[:, :, 64:],
    }
 # 边界框预测
def decode_regression_to_boxes(preds):
    preds_bbox = keras.layers.Reshape((-1, 4, 64 // 4))( # (b,4,16)
        preds
    )
    # 对特征轴归一化,之后对这16个位置加权
    preds_bbox = ops.nn.softmax(preds_bbox, axis=-1) * ops.arange(
        64 // 4, dtype="float32"
    )
    # 返回最后一维的聚合,形状变成(b,4)
    return ops.sum(preds_bbox, axis=-1)
def dist2bbox(distance, anchor_points):
    # 在最后一个维度均匀拆分偏移
    left_top, right_bottom = ops.split(distance, 2, axis=-1)
    x1y1 = anchor_points - left_top # 根据锚框中心点算的左上角点
    x2y2 = anchor_points + right_bottom # 右下角点
    return ops.concatenate((x1y1, x2y2), axis=-1)  # xyxy bbox
# 特征提取
def get_feature_extractor(model, layer_names, output_keys=None):
    if not output_keys: # 如果没有设置utput_keys,设置为层名称
        output_keys = layer_names
    items = zip(output_keys, layer_names)
    # 获取层名称到当前层输出特征图的映射字典
    outputs = {key: model.get_layer(name).output for key, name in items}
    return keras.Model(inputs=model.inputs, outputs=outputs)

# 判断锚框中心点是否在真实边界框内
def is_anchor_center_within_box(anchors, gt_bboxes):
    return ops.all(
        ops.logical_and(
            gt_bboxes[:, :, None, :2] < anchors,
            gt_bboxes[:, :, None, 2:] > anchors,
        ),
        axis=-1,
    )
# 计算边界框面积
def _compute_area(box):
    y_min, x_min, y_max, x_max = ops.split(box[..., :4], 4, axis=-1)
    return ops.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)
# 计算边界框交集大小
def _compute_intersection(boxes1, boxes2):
    y_min1, x_min1, y_max1, x_max1 = ops.split(boxes1[..., :4], 4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = ops.split(boxes2[..., :4], 4, axis=-1)
    boxes2_rank = len(boxes2.shape)
    perm = [1, 0] if boxes2_rank == 2 else [0, 2, 1]
    # [N, M] or [batch_size, N, M]
    # 计算两个框相交部位
    intersect_ymax = ops.minimum(y_max1, ops.transpose(y_max2, perm))
    intersect_ymin = ops.maximum(y_min1, ops.transpose(y_min2, perm))
    intersect_xmax = ops.minimum(x_max1, ops.transpose(x_max2, perm))
    intersect_xmin = ops.maximum(x_min1, ops.transpose(x_min2, perm))
    intersect_height = intersect_ymax - intersect_ymin # h
    intersect_width = intersect_xmax - intersect_xmin # w
    zeros_t = ops.cast(0, intersect_height.dtype)
    intersect_height = ops.maximum(zeros_t, intersect_height)
    intersect_width = ops.maximum(zeros_t, intersect_width)
    return intersect_height * intersect_width
def is_relative(bounding_box_format):
    if (
        bounding_box_format.lower()
        not in bounding_box.converters.TO_XYXY_CONVERTERS
    ):
        # 验证格式是否支持,如果 bounding_box_format 不是支持的格式之一，函数会抛出一个 ValueError 异常
        raise ValueError(
            "`is_relative()` received an unsupported format for the argument "
            f"`bounding_box_format`. `bounding_box_format` should be one of "
            f"{bounding_box.converters.TO_XYXY_CONVERTERS.keys()}. "
            f"Got bounding_box_format={bounding_box_format}"
        )
    # 判断是否为相对坐标,以rel开头的是,返回True,否则False
    return bounding_box_format.startswith("rel")
def as_relative(bounding_box_format): # 变成相对
    if not is_relative(bounding_box_format):
        return "rel_" + bounding_box_format
    return bounding_box_format
class Task(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backbone = None # 主干网
        self._functional_layer_ids = set(
            id(layer) for layer in self._flatten_layers()
        )
    # 这个方法用于控制当 Python 解释器尝试获取类实例的属性列表时的行为。__dir__ 方法通常用于提
    # 供一个类实例可访问属性的名字列表。在这个特定的实现中，__dir__ 方法过滤掉了某些属性，并排除
    # 了由 _functional_layer_ids 标记的属性。
    def __dir__(self):
        def filter_fn(attr):
            if attr in ["backbone", "_backbone"]:
                return False
            try:
                return id(getattr(self, attr)) not in self._functional_layer_ids
            except:
                return True

        return filter(filter_fn, super().__dir__())
    @property
    def backbone(self):
        return self._backbone
    @backbone.setter
    def backbone(self, value):
        self._backbone = value
    def get_config(self):
        return {
            "name": self.name,
            "trainable": self.trainable,
        }
    # 这个装饰器表明 from_config 是一个类方法，而不是实例方法。这意味着它可以直接通过类来调用，而不需要创建类的实例
    # cls 参数代表这个方法所属的类，这使得我们可以在方法内部引用类本身。config 参数是一个字典，它包含了
    # 创建类实例所需的所有配置信息。
    @classmethod
    def from_config(cls, config):
        # 如果backbone在配置的keys里面,并且对应的值是字典类型
        if "backbone" in config and isinstance(config["backbone"], dict):
            # 反序列化 'backbone' 字典，将其转换为实际的对象。
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        # 使用解包操作符 **config 将配置字典作为关键字参数传递给类构造函数，从而创建并返回一个新的类实例。
        return cls(**config)

    @classproperty
    def presets(cls):
        return {}

    @classproperty
    def presets_with_weights(cls):
        return {}

    @classproperty
    def presets_without_weights(cls):
        return {
            preset: cls.presets[preset]
            for preset in set(cls.presets) - set(cls.presets_with_weights)
        }

    @classproperty
    def backbone_presets(cls):
        return {}

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=None,
        input_shape=None,
        **kwargs,
    ):
        if preset in cls.presets:
            preset = cls.presets[preset]["kaggle_handle"]

        preset_cls = check_preset_class(preset, (cls, Backbone))

        # Backbone case.
        if issubclass(preset_cls, Backbone):
            backbone = load_from_preset(
                preset,
                load_weights=load_weights,
            )
            return cls(backbone=backbone, **kwargs)

        # Task case.
        return load_from_preset(
            preset,
            load_weights=load_weights,
            input_shape=input_shape,
            config_overrides=kwargs,
        )

    @property
    def layers(self):
        layers = super().layers
        if hasattr(self, "backbone") and self.backbone in layers:
            # We know that the backbone is not part of the graph if it has no
            # inbound nodes.
            if len(self.backbone._inbound_nodes) == 0:
                layers.remove(self.backbone)
        return layers

    def __setattr__(self, name, value):
        # Work around torch setattr for properties.
        if name in ["backbone"]:
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to set up a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have a distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        if not cls.presets:
            cls.from_preset.__func__.__doc__ = """Not implemented.

            No presets available for this class.
            """

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = Task.from_preset.__doc__
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets_with_weights), ""),
                preset_names='", "'.join(cls.presets),
                preset_with_weights_names='", "'.join(cls.presets_with_weights),
            )(cls.from_preset.__func__)

def _center_yxhw_to_xyxy(boxes, images=None, image_shape=None):
    y, x, height, width = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0],
        axis=-1,
    )
def _center_xywh_to_xyxy(boxes, images=None, image_shape=None):
    x, y, width, height = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0],
        axis=-1,
    )
def _xywh_to_xyxy(boxes, images=None, image_shape=None):
    x, y, width, height = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate([x, y, x + width, y + height], axis=-1)
def _xyxy_to_center_yxhw(boxes, images=None, image_shape=None):
    left, top, right, bottom = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [
            (top + bottom) / 2.0,
            (left + right) / 2.0,
            bottom - top,
            right - left,
        ],
        axis=-1,
    )
def _rel_xywh_to_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    x, y, width, height = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [
            image_width * x,
            image_height * y,
            image_width * (x + width),
            image_height * (y + height),
        ],
        axis=-1,
    )
def _xyxy_no_op(boxes, images=None, image_shape=None):
    return boxes
def _xyxy_to_xywh(boxes, images=None, image_shape=None):
    left, top, right, bottom = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [left, top, right - left, bottom - top],
        axis=-1,
    )
def _xyxy_to_rel_xywh(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom = ops.split(boxes, ALL_AXES, axis=-1)
    left, right = (
        left / image_width,
        right / image_width,
    )
    top, bottom = top / image_height, bottom / image_height
    return ops.concatenate(
        [left, top, right - left, bottom - top],
        axis=-1,
    )
def _xyxy_to_center_xywh(boxes, images=None, image_shape=None):
    left, top, right, bottom = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate(
        [
            (left + right) / 2.0,
            (top + bottom) / 2.0,
            right - left,
            bottom - top,
        ],
        axis=-1,
    )
# 相对位置
def _rel_xyxy_to_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom = ops.split(
        boxes,
        ALL_AXES,
        axis=-1,
    )
    left, right = left * image_width, right * image_width
    top, bottom = top * image_height, bottom * image_height
    return ops.concatenate(
        [left, top, right, bottom],
        axis=-1,
    )
def _xyxy_to_rel_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom = ops.split(
        boxes,
        ALL_AXES,
        axis=-1,
    )
    left, right = left / image_width, right / image_width
    top, bottom = top / image_height, bottom / image_height
    return ops.concatenate(
        [left, top, right, bottom],
        axis=-1,
    )
def _yxyx_to_xyxy(boxes, images=None, image_shape=None):
    y1, x1, y2, x2 = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate([x1, y1, x2, y2], axis=-1)
def _rel_yxyx_to_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    top, left, bottom, right = ops.split(
        boxes,
        ALL_AXES,
        axis=-1,
    )
    left, right = left * image_width, right * image_width
    top, bottom = top * image_height, bottom * image_height
    return ops.concatenate(
        [left, top, right, bottom],
        axis=-1,
    )
def _xyxy_to_yxyx(boxes, images=None, image_shape=None):
    x1, y1, x2, y2 = ops.split(boxes, ALL_AXES, axis=-1)
    return ops.concatenate([y1, x1, y2, x2], axis=-1)
# 相对,就是归一化后的坐标
def _xyxy_to_rel_yxyx(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom = ops.split(boxes, ALL_AXES, axis=-1)
    left, right = left / image_width, right / image_width
    top, bottom = top / image_height, bottom / image_height
    return ops.concatenate(
        [top, left, bottom, right],
        axis=-1,
    )
TO_XYXY_CONVERTERS = {
    "xywh": _xywh_to_xyxy,
    "center_xywh": _center_xywh_to_xyxy,
    "center_yxhw": _center_yxhw_to_xyxy,
    "rel_xywh": _rel_xywh_to_xyxy,
    "xyxy": _xyxy_no_op,
    "rel_xyxy": _rel_xyxy_to_xyxy,
    "yxyx": _yxyx_to_xyxy,
    "rel_yxyx": _rel_yxyx_to_xyxy,
}

FROM_XYXY_CONVERTERS = {
    "xywh": _xyxy_to_xywh,
    "center_xywh": _xyxy_to_center_xywh,
    "center_yxhw": _xyxy_to_center_yxhw,
    "rel_xywh": _xyxy_to_rel_xywh,
    "xyxy": _xyxy_no_op,
    "rel_xyxy": _xyxy_to_rel_xyxy,
    "yxyx": _xyxy_to_yxyx,
    "rel_yxyx": _xyxy_to_rel_yxyx,
}
def convert_format(
    boxes, source, target, images=None, image_shape=None, dtype="float32"
):
    if isinstance(boxes, dict):
        converted_boxes = boxes.copy()
        converted_boxes["boxes"] = convert_format(
            boxes["boxes"],
            source=source,
            target=target,
            images=images,
            image_shape=image_shape,
            dtype=dtype,
        )
        return converted_boxes

    if boxes.shape[-1] is not None and boxes.shape[-1] != 4:
        raise ValueError(
            "Expected `boxes` to be a Tensor with a final dimension of "
            f"`4`. Instead, got `boxes.shape={boxes.shape}`."
        )
    if images is not None and image_shape is not None:
        raise ValueError(
            "convert_format() expects either `images` or `image_shape`, but "
            f"not both. Received images={images} image_shape={image_shape}"
        )

    _validate_image_shape(image_shape)

    source = source.lower()
    target = target.lower()
    if source not in TO_XYXY_CONVERTERS:
        raise ValueError(
            "`convert_format()` received an unsupported format for the "
            "argument `source`. `source` should be one of "
            f"{TO_XYXY_CONVERTERS.keys()}. Got source={source}"
        )
    if target not in FROM_XYXY_CONVERTERS:
        raise ValueError(
            "`convert_format()` received an unsupported format for the "
            "argument `target`. `target` should be one of "
            f"{FROM_XYXY_CONVERTERS.keys()}. Got target={target}"
        )

    boxes = ops.cast(boxes, dtype)
    if source == target:
        return boxes

    # rel->rel conversions should not require images
    if source.startswith("rel") and target.startswith("rel"):
        source = source.replace("rel_", "", 1)
        target = target.replace("rel_", "", 1)

    boxes, images, squeeze = _format_inputs(boxes, images)
    to_xyxy_fn = TO_XYXY_CONVERTERS[source]
    from_xyxy_fn = FROM_XYXY_CONVERTERS[target]

    try:
        in_xyxy = to_xyxy_fn(boxes, images=images, image_shape=image_shape)
        result = from_xyxy_fn(in_xyxy, images=images, image_shape=image_shape)
    except RequiresImagesException:
        raise ValueError(
            "convert_format() must receive `images` or `image_shape` when "
            "transforming between relative and absolute formats."
            f"convert_format() received source=`{format}`, target=`{format}, "
            f"but images={images} and image_shape={image_shape}."
        )
    return _format_outputs(result, squeeze)
def compute_iou(
    boxes1,
    boxes2,
    bounding_box_format,
    use_masking=False,
    mask_val=-1,
    images=None,
    image_shape=None,
):
    boxes1_rank = len(boxes1.shape)
    boxes2_rank = len(boxes2.shape)
    if boxes1_rank not in [2, 3]:
        raise ValueError(
            "compute_iou() expects boxes1 to be batched, or to be unbatched. "
            f"Received len(boxes1.shape)={boxes1_rank}, "
            f"len(boxes2.shape)={boxes2_rank}. Expected either "
            "len(boxes1.shape)=2 AND or len(boxes1.shape)=3."
        )
    if boxes2_rank not in [2, 3]:
        raise ValueError(
            "compute_iou() expects boxes2 to be batched, or to be unbatched. "
            f"Received len(boxes1.shape)={boxes1_rank}, "
            f"len(boxes2.shape)={boxes2_rank}. Expected either "
            "len(boxes2.shape)=2 AND or len(boxes2.shape)=3."
        )

    target_format = "yxyx"
    if bounding_box.is_relative(bounding_box_format):
        target_format = bounding_box.as_relative(target_format)

    boxes1 = bounding_box.convert_format(
        boxes1,
        source=bounding_box_format,
        target=target_format,
        images=images,
        image_shape=image_shape,
    )

    boxes2 = bounding_box.convert_format(
        boxes2,
        source=bounding_box_format,
        target=target_format,
        images=images,
        image_shape=image_shape,
    )

    intersect_area = _compute_intersection(boxes1, boxes2) # 交集
    boxes1_area = _compute_area(boxes1) # 计算boxes1面积
    boxes2_area = _compute_area(boxes2)
    boxes2_area_rank = len(boxes2_area.shape)
    boxes2_axis = 1 if (boxes2_area_rank == 2) else 0
    boxes1_area = ops.expand_dims(boxes1_area, axis=-1)
    boxes2_area = ops.expand_dims(boxes2_area, axis=boxes2_axis)
    union_area = boxes1_area + boxes2_area - intersect_area
    # iou
    res = ops.divide(intersect_area, union_area + keras.backend.epsilon())

    if boxes1_rank == 2:
        perm = [1, 0]
    else:
        perm = [0, 2, 1]

    if not use_masking:
        return res
    mask_val_t = ops.cast(mask_val, res.dtype) * ops.ones_like(res)
    boxes1_mask = ops.less(ops.max(boxes1, axis=-1, keepdims=True), 0.0)
    boxes2_mask = ops.less(ops.max(boxes2, axis=-1, keepdims=True), 0.0)
    background_mask = ops.logical_or(
        boxes1_mask, ops.transpose(boxes2_mask, perm)
    )
    iou_lookup_table = ops.where(background_mask, mask_val_t, res)
    return iou_lookup_table
def compute_ciou(boxes1, boxes2, bounding_box_format):
    target_format = "xyxy"
    if bounding_box.is_relative(bounding_box_format):
        target_format = bounding_box.as_relative(target_format)

    boxes1 = bounding_box.convert_format(
        boxes1, source=bounding_box_format, target=target_format
    )

    boxes2 = bounding_box.convert_format(
        boxes2, source=bounding_box_format, target=target_format
    )

    x_min1, y_min1, x_max1, y_max1 = ops.split(boxes1[..., :4], 4, axis=-1)
    x_min2, y_min2, x_max2, y_max2 = ops.split(boxes2[..., :4], 4, axis=-1)

    width_1 = x_max1 - x_min1
    height_1 = y_max1 - y_min1 + keras.backend.epsilon()
    width_2 = x_max2 - x_min2
    height_2 = y_max2 - y_min2 + keras.backend.epsilon()

    intersection_area = ops.maximum(
        ops.minimum(x_max1, x_max2) - ops.maximum(x_min1, x_min2), 0
    ) * ops.maximum(
        ops.minimum(y_max1, y_max2) - ops.maximum(y_min1, y_min2), 0
    )
    union_area = (
        width_1 * height_1
        + width_2 * height_2
        - intersection_area
        + keras.backend.epsilon()
    )
    iou = ops.squeeze(
        ops.divide(intersection_area, union_area + keras.backend.epsilon()),
        axis=-1,
    )

    convex_width = ops.maximum(x_max1, x_max2) - ops.minimum(x_min1, x_min2)
    convex_height = ops.maximum(y_max1, y_max2) - ops.minimum(y_min1, y_min2)
    convex_diagonal_squared = ops.squeeze(
        convex_width**2 + convex_height**2 + keras.backend.epsilon(),
        axis=-1,
    )
    centers_distance_squared = ops.squeeze(
        ((x_min1 + x_max1) / 2 - (x_min2 + x_max2) / 2) ** 2
        + ((y_min1 + y_max1) / 2 - (y_min2 + y_max2) / 2) ** 2,
        axis=-1,
    )

    v = ops.squeeze(
        ops.power(
            (4 / math.pi**2)
            * (ops.arctan(width_2 / height_2) - ops.arctan(width_1 / height_1)),
            2,
        ),
        axis=-1,
    )
    alpha = v / (v - iou + (1 + keras.backend.epsilon()))

    return iou - (
        centers_distance_squared / convex_diagonal_squared + v * alpha
    )
class YOLOV8LabelEncoder(keras.layers.Layer):
    def __init__(
        self,
        num_classes,
        max_anchor_matches=10,
        alpha=0.5,
        beta=6.0,
        epsilon=1e-9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_anchor_matches = max_anchor_matches
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def assign(
        self, scores, decode_bboxes, anchors, gt_labels, gt_bboxes, gt_mask
    ):
        num_anchors = anchors.shape[0]
        # 通过[:, None, :]的切片操作，gt_labels被扩展成了一个三维张量。[N, 1, C]
        # ops.maximum(gt_labels[:, None, :], 0)确保gt_labels中的所有值都是非负的。这
        # 是因为有些框架中的标签可能包含负数（如-1用于表示忽略或未定义类别），但在这个上下文
        # 中我们只关心正类别。ops.cast(..., "int32")将上一步的结果转换为int32类型。这
        # 是为了确保后续操作（如索引操作）的数据类型一致性。
        # ops.take_along_axis函数用于沿着指定的轴（axis）使用索引张量来选取元素。在这里，
        # 它沿着最后一个轴（axis=-1）工作，即类别的维度。
        # 第一个参数scores是一个三维张量，包含了模型对每个样本属于每个类别的预测得分
        # 第二个参数是之前通过一系列操作得到的索引张量，它指定了从scores中选取哪些类别的得分。具体来说，
        # 对于每个样本，它都选取gt_labels中对应类别的得分（因为gt_labels中的值是类别索引，且已经被转
        # 换为非负整数）。
        # 这行代码的目的是从模型的预测得分中，对于每个样本，只选取其真实类别对应的得分。
        bbox_scores = ops.take_along_axis(
            scores,
            ops.cast(ops.maximum(gt_labels[:, None, :], 0), "int32"),
            axis=-1,
        )
        
        bbox_scores = ops.transpose(bbox_scores, (0, 2, 1)) 
        overlaps = compute_ciou(
            ops.expand_dims(gt_bboxes, axis=2),
            ops.expand_dims(decode_bboxes, axis=1),
            bounding_box_format="xyxy",
        )

        alignment_metrics = ops.power(bbox_scores, self.alpha) * ops.power(
            overlaps, self.beta
        )
        alignment_metrics = ops.where(gt_mask, alignment_metrics, 0)

        # Only anchors which are inside of relevant GT boxes are considered
        # for assignment.
        # This is a boolean tensor of shape (B, num_gt_boxes, num_anchors)
        matching_anchors_in_gt_boxes = is_anchor_center_within_box(
            anchors, gt_bboxes
        )
        alignment_metrics = ops.where(
            matching_anchors_in_gt_boxes, alignment_metrics, 0
        )

        # The top-k highest alignment metrics are used to select K candidate
        # anchors for each GT box.
        candidate_metrics, candidate_idxs = ops.top_k(
            alignment_metrics, self.max_anchor_matches
        )
        candidate_idxs = ops.where(candidate_metrics > 0, candidate_idxs, -1)

        # We now compute a dense grid of anchors and GT boxes. This is useful
        # for picking a GT box when an anchor matches to 2, as well as returning
        # to a dense format for a mask of which anchors have been matched.
        anchors_matched_gt_box = ops.zeros_like(overlaps)
        for k in range(self.max_anchor_matches):
            anchors_matched_gt_box += ops.one_hot(
                candidate_idxs[:, :, k], num_anchors
            )

        # We zero-out the overlap for anchor, GT box pairs which don't match.
        overlaps *= anchors_matched_gt_box
        # In cases where one anchor matches to 2 GT boxes, we pick the GT box
        # with the highest overlap as a max.
        gt_box_matches_per_anchor = ops.argmax(overlaps, axis=1)
        gt_box_matches_per_anchor_mask = ops.max(overlaps, axis=1) > 0
        
        gt_box_matches_per_anchor = ops.cast(gt_box_matches_per_anchor, "int32")

        # We select the GT boxes and labels that correspond to anchor matches.
        bbox_labels = ops.take_along_axis(
            gt_bboxes, gt_box_matches_per_anchor[:, :, None], axis=1
        )
        bbox_labels = ops.where(
            gt_box_matches_per_anchor_mask[:, :, None], bbox_labels, -1
        )
        class_labels = ops.take_along_axis(
            gt_labels, gt_box_matches_per_anchor, axis=1
        )
        class_labels = ops.where(
            gt_box_matches_per_anchor_mask, class_labels, -1
        )

        class_labels = ops.one_hot(
            ops.cast(class_labels, "int32"), self.num_classes
        )

        
        alignment_metrics *= anchors_matched_gt_box
        max_alignment_per_gt_box = ops.max(
            alignment_metrics, axis=-1, keepdims=True
        )
        max_overlap_per_gt_box = ops.max(overlaps, axis=-1, keepdims=True)

        normalized_alignment_metrics = ops.max(
            alignment_metrics
            * max_overlap_per_gt_box
            / (max_alignment_per_gt_box + self.epsilon),
            axis=-2,
        )
        class_labels *= normalized_alignment_metrics[:, :, None]

        # On TF backend, the final "4" becomes a dynamic shape so we include
        # this to force it to a static shape of 4. This does not actually
        # reshape the Tensor.
        bbox_labels = ops.reshape(bbox_labels, (-1, num_anchors, 4))
        return (
            ops.stop_gradient(bbox_labels),
            ops.stop_gradient(class_labels),
            ops.stop_gradient(
                ops.cast(gt_box_matches_per_anchor > -1, "float32")
            ),
        )

    def call(
        self, scores, decode_bboxes, anchors, gt_labels, gt_bboxes, gt_mask
    ):
       
        if isinstance(gt_bboxes, tf.RaggedTensor):
            dense_bounding_boxes = bounding_box.to_dense(
                {"boxes": gt_bboxes, "classes": gt_labels},
            )
            gt_bboxes = dense_bounding_boxes["boxes"]
            gt_labels = dense_bounding_boxes["classes"]

        if isinstance(gt_mask, tf.RaggedTensor):
            gt_mask = gt_mask.to_tensor()

        max_num_boxes = ops.shape(gt_bboxes)[1]

        # If there are no GT boxes in the batch, we short-circuit and return
        # empty targets to avoid NaNs.
        return ops.cond(
            ops.array(max_num_boxes > 0),
            lambda: self.assign(
                scores, decode_bboxes, anchors, gt_labels, gt_bboxes, gt_mask
            ),
            lambda: (
                ops.zeros_like(decode_bboxes),
                ops.zeros_like(scores),
                ops.zeros_like(scores[..., 0]),
            ),
        )

    def count_params(self):
        # The label encoder has no weights, so we short-circuit the weight
        # counting to avoid having to `build` this layer unnecessarily.
        return 0

    def get_config(self):
        config = {
            "max_anchor_matches": self.max_anchor_matches,
            "num_classes": self.num_classes,
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
class YOLOV8Detector(Task):
    def __init__(
        self,
        backbone,
        num_classes,
        bounding_box_format,
        fpn_depth=2,
        label_encoder=None,
        prediction_decoder=None,
        **kwargs,
    ):
        extractor_levels = ["P3", "P4", "P5"]
        extractor_layer_names = [
            pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )
        #(224, 224, 3)
        images = keras.layers.Input(feature_extractor.input_shape[1:])
        features = list(feature_extractor(images).values())
        # (None, 28, 28, 64),(None, 14, 14, 128),(None, 7, 7, 256)
        fpn_features = apply_path_aggregation_fpn(
            features, depth=fpn_depth, name="pa_fpn"
        )
        # 经过对不同尺寸的特征图提取特征得到包含信息的网络特征点
        outputs = apply_yolo_v8_head(
            fpn_features,
            num_classes,
        )
        # 为了使损失度量更加清晰易懂，我们使用了一个具有良好名称的无操作层。
        boxes = keras.layers.Concatenate(axis=1, name="box")([outputs["boxes"]])
        scores = keras.layers.Concatenate(axis=1, name="class")(
            [outputs["classes"]]
        )
        # 形状变成(b,total_anchs,64),total_anchs对应各个特征轴中的所有锚框中心点
        outputs = {"boxes": boxes, "classes": scores}
        super().__init__(inputs=images, outputs=outputs, **kwargs)

        self.bounding_box_format = bounding_box_format
        # 默认的非最大值抑制,置信度阈值是0.2,iou阈值是0.7
        self._prediction_decoder = (
            prediction_decoder
            or layers.NonMaxSuppression(
                bounding_box_format=bounding_box_format,
                from_logits=False,
                confidence_threshold=0.2,
                iou_threshold=0.7,
            )
        )
        self.backbone = backbone
        self.fpn_depth = fpn_depth
        self.num_classes = num_classes
        self.label_encoder = label_encoder or YOLOV8LabelEncoder(
            num_classes=num_classes
        )

    def compile(
        self,
        box_loss,
        classification_loss,
        box_loss_weight=7.5,
        classification_loss_weight=0.5,
        metrics=None,
        **kwargs,
    ):
        if metrics is not None:
            raise ValueError("User metrics not yet supported for YOLOV8")

        if isinstance(box_loss, str):
            if box_loss == "ciou":
                box_loss = CIoULoss(bounding_box_format="xyxy", reduction="sum")
            elif box_loss == "iou":
                warnings.warn(
                    "YOLOV8 recommends using CIoU loss, but was configured to "
                    "use standard IoU. Consider using `box_loss='ciou'` "
                    "instead."
                )
            else:
                raise ValueError(
                    f"Invalid box loss for YOLOV8Detector: {box_loss}. Box "
                    "loss should be a keras.Loss or the string 'ciou'."
                )
        if isinstance(classification_loss, str):
            if classification_loss == "binary_crossentropy":
                classification_loss = keras.losses.BinaryCrossentropy(
                    reduction="sum"
                )
            else:
                raise ValueError(
                    "Invalid classification loss for YOLOV8Detector: "
                    f"{classification_loss}. Classification loss should be a "
                    "keras.Loss or the string 'binary_crossentropy'."
                )

        self.box_loss = box_loss
        self.classification_loss = classification_loss
        self.box_loss_weight = box_loss_weight
        self.classification_loss_weight = classification_loss_weight

        losses = {
            "box": self.box_loss,
            "class": self.classification_loss,
        }

        super().compile(loss=losses, **kwargs)

    def train_step(self, *args):
        data = args[-1]
        args = args[:-1]
        x, y = unpack_input(data)
        return super().train_step(*args, (x, y))

    def test_step(self, *args):
        data = args[-1]
        args = args[:-1]
        x, y = unpack_input(data)
        return super().test_step(*args, (x, y))

    def compute_loss(self, x, y, y_pred, sample_weight=None, **kwargs):
        box_pred, cls_pred = y_pred["boxes"], y_pred["classes"]

        pred_boxes = decode_regression_to_boxes(box_pred)
        pred_scores = cls_pred

        anchor_points, stride_tensor = get_anchors(image_shape=x.shape[1:])
        stride_tensor = ops.expand_dims(stride_tensor, axis=-1)

        gt_labels = y["classes"]

        mask_gt = ops.all(y["boxes"] > -1.0, axis=-1, keepdims=True)
        gt_bboxes = bounding_box.convert_format(
            y["boxes"],
            source=self.bounding_box_format,
            target="xyxy",
            images=x,
        )

        pred_bboxes = dist2bbox(pred_boxes, anchor_points)

        target_bboxes, target_scores, fg_mask = self.label_encoder(
            pred_scores,
            ops.cast(pred_bboxes * stride_tensor, gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_bboxes /= stride_tensor
        target_scores_sum = ops.maximum(ops.sum(target_scores), 1)
        box_weight = ops.expand_dims(
            ops.sum(target_scores, axis=-1) * fg_mask,
            axis=-1,
        )

        y_true = {
            "box": target_bboxes * fg_mask[..., None],
            "class": target_scores,
        }
        y_pred = {
            "box": pred_bboxes * fg_mask[..., None],
            "class": pred_scores,
        }
        sample_weights = {
            "box": self.box_loss_weight * box_weight / target_scores_sum,
            "class": self.classification_loss_weight / target_scores_sum,
        }

        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights, **kwargs
        )

    def decode_predictions(
        self,
        pred,
        images,
    ):
        boxes = pred["boxes"]
        scores = pred["classes"]

        boxes = decode_regression_to_boxes(boxes)

        anchor_points, stride_tensor = get_anchors(image_shape=images.shape[1:])
        stride_tensor = ops.expand_dims(stride_tensor, axis=-1)

        box_preds = dist2bbox(boxes, anchor_points) * stride_tensor
        box_preds = bounding_box.convert_format(
            box_preds,
            source="xyxy",
            target=self.bounding_box_format,
            images=images,
        )

        return self.prediction_decoder(box_preds, scores)

    def predict_step(self, *args):
        outputs = super().predict_step(*args)
        if isinstance(outputs, tuple):
            return self.decode_predictions(outputs[0], args[-1]), outputs[1]
        else:
            return self.decode_predictions(outputs, args[-1])

    @property
    def prediction_decoder(self):
        return self._prediction_decoder

    @prediction_decoder.setter
    def prediction_decoder(self, prediction_decoder):
        if prediction_decoder.bounding_box_format != self.bounding_box_format:
            raise ValueError(
                "Expected `prediction_decoder` and YOLOV8Detector to "
                "use the same `bounding_box_format`, but got "
                "`prediction_decoder.bounding_box_format="
                f"{prediction_decoder.bounding_box_format}`, and "
                "`self.bounding_box_format="
                f"{self.bounding_box_format}`."
            )
        self._prediction_decoder = prediction_decoder
        self.make_predict_function(force=True)
        self.make_train_function(force=True)
        self.make_test_function(force=True)

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "bounding_box_format": self.bounding_box_format,
            "fpn_depth": self.fpn_depth,
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "label_encoder": keras.saving.serialize_keras_object(
                self.label_encoder
            ),
            "prediction_decoder": keras.saving.serialize_keras_object(
                self._prediction_decoder
            ),
        }

    @classmethod
    def from_config(cls, config):
        config["backbone"] = keras.saving.deserialize_keras_object(
            config["backbone"]
        )
        label_encoder = config.get("label_encoder")
        if label_encoder is not None and isinstance(label_encoder, dict):
            config["label_encoder"] = keras.saving.deserialize_keras_object(
                label_encoder
            )
        prediction_decoder = config.get("prediction_decoder")
        if prediction_decoder is not None and isinstance(
            prediction_decoder, dict
        ):
            config["prediction_decoder"] = (
                keras.saving.deserialize_keras_object(prediction_decoder)
            )
        return cls(**config)



    

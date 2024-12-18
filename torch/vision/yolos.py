# 用于计算Dice损失，这是一种类似于广义IOU的损失函数，特别适用于分割任务中的二分类问题。
def dice_loss(inputs, targets, num_boxes):
    # inputs：一个浮点张量，其形状任意。这个张量存储了每个样本的预测值。
    # targets：一个与 inputs 形状相同的浮点张量。存储了每个元素的二分类标签（0表示负类，1表示正类）。
    # num_boxes：一个整数，表示有效目标的数量，用于平均损失。
    inputs = inputs.sigmoid() # sigmoid激活：首先将 inputs 通过 sigmoid 函数，将其转换为概率分布。
    # 将 inputs 张量展平为二维张量，形状为 (batch_size, -1)。
    inputs = inputs.flatten(1)
    # 计算分子部分：计算 inputs 和 targets 相乘后的和，即交集部分的两倍。
    numerator = 2 * (inputs * targets).sum(1)
    # 计算分母部分：计算 inputs 和 targets 的和。
    denominator = inputs.sum(-1) + targets.sum(-1)
    # 计算Dice系数：根据Dice系数的定义计算损失。
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 求平均损失：最后将所有样本的损失求和，并除以 num_boxes，得到平均损失。
    return loss.sum() / num_boxes
# 用于计算Focal Loss，这是一种专门用于解决类别不平衡问题的损失函数。Focal Loss最初在
# RetinaNet中提出，用于密集检测任务。
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    # inputs：一个形状任意的 torch.FloatTensor。存储了每个样本的预测值。
    # targets：一个与 inputs 形状相同的 torch.FloatTensor。存储了每个元素的二分类
    # 标签（0表示负类，1表示正类）。
    # num_boxes：一个整数，表示有效目标的数量，用于平均损失
    # alpha：一个可选的浮点数，默认值为 0.25，用于平衡正负样本的权重。
    # gamma：一个可选的浮点数，默认值为 2，用于调节容易样本和困难样本的权重。
    # 计算预测概率：使用 sigmoid 函数将 inputs 转换为概率
    prob = inputs.sigmoid()
    # 计算交叉熵损失：使用 binary_cross_entropy_with_logits 计算二分类交叉熵损失。
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 添加调制因子：根据 Focal Loss 的定义，添加调制因子 (1−pt)γ(1−pt​)γ，以平衡容易样本和困难样本。
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    # 应用权重因子：如果 alpha 不小于0，则应用权重因子 αtαt​，以进一步平衡正负样本。
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    # 计算平均损失：最后将所有样本的损失求平均，并除以 num_boxes，得到最终的平均损失。
    return loss.mean(1).sum() / num_boxes
# 这段代码定义了一个 _upcast 函数，它的目的是保护数值乘法操作免受溢出的影响，通过将张量类型提升
# 到更高的精度类型来实现这一点。具体来说，这个函数会检查传入的张量 t 的数据类型，并将其转换为适
# 当的更高精度的数据类型。
def _upcast(t: Tensor) -> Tensor:
    # 如果 t 是浮点数类型（即 t.is_floating_point() 为 True），那么函数会检查 t 是否已经是 
    # torch.float32 或 torch.float64 类型。如果不是，则将 t 转换为 torch.float32 类型。
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        # 如果 t 不是浮点数类型（即 t 是整数类型），那么函数会检查 t 是否已经是 torch.int32 
        # 或 torch.int64 类型。如果不是，则将 t 转换为 torch.int64 类型。
        return t if t.dtype in (torch.int32, torch.int64) else t.int()
def box_area(boxes: Tensor) -> Tensor:
    # 计算边界框的面积
    boxes = _upcast(boxes) # 转换数据类型
    # (x2-x1)*(y2-y1)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

# 计算两个边界框集合之间的交并比（Intersection over Union, IoU）的功能。IoU 是一种常用的
# 评估边界框重叠程度的指标，常用于目标检测任务中的评价。
# boxes1：形状为 [N, 4] 的张量，表示 N 个边界框，每个边界框由 (x1, y1, x2, y2) 组成
# ，其中 (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标。
# boxes2：形状为 [M,4] 的张量，表示 M 个边界框
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1) # 边界框1的面积
    area2 = box_area(boxes2) # 边界框2的面积
    # left_top 计算两个边界框集合之间的最大左上角坐标。
    # boxes1[:, None, :2]：在 boxes1 的第 1 轴增加了一维，使其形状变为 [N, 1, 4]。
    # boxes2[:, :2]：保留 boxes2 的左上角坐标，形状为 [M, 2]。
    # torch.max(boxes1[:, None, :2], boxes2[:, :2])：这一步骤会将 [N, 1, 2] 形状的 
    # boxes1 和 [M, 2] 形状的 boxes2 进行逐元素的最大值计算。
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # right_bottom 计算两个边界框集合之间的最小右下角坐标。
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    # width_height 计算交集区域的宽度和高度，并使用 .clamp(min=0) 确保宽度和高度都是非负数。
    # clamp 是 PyTorch 中的一个函数，它可以将张量中的每一个元素限制在一个指定的范围内。对于
    # clamp(min=0)，它确保张量中的每一个元素都不小于0。如果某个元素小于0，则会被设置为0。
    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    # inter 计算交集区域的面积。boxes1中的每个边界框和boxes2中的所有边界框的交集面积
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]
    # union 计算并集区域的面积，通过将 boxes1 和 boxes2 的面积相加再减去交集面积得到
    # 为什么需要增加一维？形状匹配：
    # area1 的形状是 [N]，表示有 N 个边界框的面积。
    # area2 的形状是 [M]，表示有 M 个边界框的面积。
    # inter 的形状是 [N, M]，表示 N 个边界框与 M 个边界框之间的交集面积。
    # 为了能够将 area1 与 area2 相加，并且与 [N, M] 形状的 inter 相减，我们需要
    # 让 area1 的形状变为 [N, 1]，这样可以通过广播机制与 [M] 形状的 area2 相加，
    # 并且与 [N, M] 形状的 inter 相减。
    # area1[:, None]：在 area1 的第 1 轴增加了一维，使其形状变为 [N, 1]
    # area2：保持形状 [M]。inter：形状为 [N, M]。
    union = area1[:, None] + area2 - inter
    iou = inter / union # iou 计算 IoU，即交集面积除以并集面积。
    return iou, union
# 用于计算广义交并比（Generalized Intersection over Union, GIoU），这是一种扩展了传统 IoU 的概念，旨在更好
# 地衡量两个边界框之间的重叠情况，并且能够处理边界框完全包含的情况。
def generalized_box_iou(boxes1, boxes2):
    # boxes1：形状为 [N, 4] 的张量，表示 N 个边界框，每个边界框由 (x1, y1, x2, y2) 组成，其中
    # (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标。
    # boxes2：形状为 [M, 4] 的张量，表示 M 个边界框，格式同 boxes1
    # 验证边界框格式：检查 boxes1 和 boxes2 是否为 [x0, y0, x1, y1] 格式，确保每个边界框的右
    # 下角坐标大于等于左上角坐标。如果不符合格式，则抛出 ValueError。
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    # 计算 IoU 和并集面积：调用 box_iou 函数计算两个边界框集合之间的 IoU 和并集面积。
    iou, union = box_iou(boxes1, boxes2)
    # 计算包围框的左上角和右下角坐标：
    # top_left 计算两个边界框集合之间的最小左上角坐标。
    # bottom_right 计算两个边界框集合之间的最大右下角坐标。
    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    # 计算包围框的宽度和高度：
    # width_height 计算包围框的宽度和高度，并使用 .clamp(min=0) 确保宽度和高度都是非负数。
    # area 计算包围框的面积。
    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]
    # 计算 GIoU
    return iou - (area - union) / area
# 此函数接收一个包含多个张量的列表 tensor_list，并将这些张量合并成一个统一大小的张量，
# 并创建一个 NestedTensor 对象。
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # 获取最大尺寸：
    if tensor_list[0].ndim == 3:
        #获取最大尺寸：使用 _max_by_axis 函数找到 tensor_list 中所有张量的最大尺寸。
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        # 创建一个新的张量 tensor，其形状为 [batch_size, num_channels, height, width]，
        # 其中 height 和 width 是最大尺寸。
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        # 初始化掩码：创建一个全 True 的布尔张量 mask，形状为 [batch_size, height, width]。
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # 填充张量和更新掩码：遍历 tensor_list 中的每个张量 img，将其复制到 tensor 的相应位置，
        # 并更新 mask 以标记有效区域。
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    #返回 NestedTensor 对象创建并返回 NestedTensor 对象，其中包含 tensor 和 mask。
    return NestedTensor(tensor, mask)

# 配置YOLOS模型的基础参数
# YolosConfig 类继承自 PretrainedConfig，用于定义YOLOS模型的配置信息。
class YolosConfig(PretrainedConfig):
    model_type = "yolos"
    def __init__(
        self,
        hidden_size=512, # 隐藏层的维度大小。
        num_hidden_layers=8, # 隐藏层的数量。
        num_attention_heads=8, # 多头注意力机制的头数
        intermediate_size=2048, # 前馈神经网络（FFN）的中间层大小。
        hidden_act="gelu", # 隐藏层激活函数。
        hidden_dropout_prob=0.0,# 隐藏层的Dropout概率。
        attention_probs_dropout_prob=0.0, # 注意力概率的Dropout概率。
        initializer_range=0.02,# 决定了模型权重初始化的范围，通常使用截断正态分布进行初始化。
        layer_norm_eps=1e-12,# 层规范化中的epsilon值。
        image_size=[512, 864],# 输入图像的大小。
        patch_size=16, # 图像分割的补丁大小。
        num_channels=3, # 图像的通道数
        qkv_bias=True,# 查询、键、值（Query-Key-Value）是否使用偏置。
        num_detection_tokens=100,# 检测目标的标记数量。
        # 是否使用中间位置嵌入。
        # 在某些变体中，可能会在编码器和解码器之间加入位置嵌入，以增强模型的空间感知能力。
        use_mid_position_embeddings=True,
        # 是否启用辅助损失,在训练过程中，除了主损失外，还可以启用辅助损失来改善训练效果。
        auxiliary_loss=False,
        # 匈牙利匹配器相关参数：
        # 这些参数用于匈牙利算法（Hungarian matcher），该算法用于匹配预测框和真实框，以计算损失。
        class_cost=1,# 类别匹配成本。
        bbox_cost=5,# 边界框匹配成本。
        giou_cost=2,# 广义交并比（GIoU）匹配成本。
        # 作用：这些系数用于平衡不同类型的损失贡献，以优化总体损失函数。
        # 损失系数：bbox_loss_coefficient：边界框损失系数。
        # giou_loss_coefficient：广义交并比损失系数。
        # eos_coefficient：结束符号（EOS）损失系数。
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.num_detection_tokens = num_detection_tokens
        self.use_mid_position_embeddings = use_mid_position_embeddings
        self.auxiliary_loss = auxiliary_loss
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient

# 用于配置YOLOS模型导出为ONNX格式的相关信息。
class YolosOnnxConfig(OnnxConfig):
    # 支持ONNX导出的PyTorch最低版本。确保使用的PyTorch版本支持ONNX导出所需的特性。
    torch_onnx_minimum_version = version.parse("1.11")
    # 模型输入的形状描述。定义了模型输入的形状，包括批量大小、通道数、高度和宽度。
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )
    # 验证ONNX模型时的绝对容差。用于验证ONNX模型输出与原生模型输出之间的差异。
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
    # 默认的ONNX操作集版本。指定ONNX模型导出时使用的基本操作集版本。
    @property
    def default_onnx_opset(self) -> int:
        return 12
# 这个类使用了Python的 dataclass 装饰器来简化类的定义，自动实现了构造函数、字符串表示等方法。这个类用
# 于封装YOLOS模型的输出数据，包括但不限于：
@dataclass
# 用于封装YOLOS模型的输出
class YolosObjectDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None # 可选的总损失
    loss_dict: Optional[Dict] = None # 可选的损失字典，可能包含不同损失项的详细信息。
    logits: torch.FloatTensor = None # 模型的分类得分。
    pred_boxes: torch.FloatTensor = None # 预测的边界框坐标。
    auxiliary_outputs: Optional[List[Dict]] = None # 辅助输出，可能包含中间层的结果。
    last_hidden_state: Optional[torch.FloatTensor] = None # 最后一个隐藏状态。
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None # 所有隐藏层的状态。
    attentions: Optional[Tuple[torch.FloatTensor]] = None # 所有注意力层的注意力权重。
# 这个类继承自 nn.Module，定义了YOLOS模型的嵌入层，包括CLSToken、检测Token、位置嵌入和补丁嵌入。
# 嵌入层的主要作用是将输入的图像转换为适合模型处理的形式，并添加位置信息。
class YolosEmbeddings(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # cls_token：分类标记（CLS Token），用于表示整个输入序列的表示。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # detection_tokens：检测tokens，用于捕获图像中的潜在目标对象。
        self.detection_tokens = nn.Parameter(torch.zeros(1, config.num_detection_tokens, config.hidden_size))
        # patch_embeddings：补丁嵌入，将图像分割成补丁并嵌入到隐藏空间。
        self.patch_embeddings = YolosPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches # np,图片被分成的块数
        # pe：位置嵌入，用于表示图像块和检测tokens和cls的位置信息。1 是批处理维度，表示位置嵌入对于每个样本都是一样的
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + config.num_detection_tokens + 1, config.hidden_size)
        )
        # dropout：丢弃层，用于防止过拟合。
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # interpolation：插值模块，用于处理输入图像尺寸的变化。
        self.interpolation = InterpolateInitialPositionEmbeddings(config)
        self.config = config
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 处理输入：获取输入图像的尺寸信息。b,c,h,w
        batch_size, num_channels, height, width = pixel_values.shape
        # 生成图像块嵌入：将输入图像转换为图像块嵌入。
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.size() # b,s,d
        # 添加CLSToken和检测Tokens：将CLSToken和检测Tokens扩展到当前批次大小，并与图像块嵌入拼接。
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (b,1,d)
        detection_tokens = self.detection_tokens.expand(batch_size, -1, -1) #(b,n_det,d)
        # 依次将cls token的嵌入和图像块token的嵌入和要检测的token的嵌入在Time轴拼接
        embeddings = torch.cat((cls_tokens, embeddings, detection_tokens), dim=1)
        # add positional encoding to each token
        # this might require interpolation of the existing position embeddings
        # 位置嵌入插值：根据输入图像的尺寸对位置嵌入进行插值处理。
        position_embeddings = self.interpolation(self.position_embeddings, (height, width))
        # 添加位置嵌入：将位置嵌入加到补丁嵌入上。
        embeddings = embeddings + position_embeddings
        # 应用Dropout：最后应用Dropout层，减少过拟合的风险。
        embeddings = self.dropout(embeddings)
        return embeddings
# nn.Module 子类，其目的是处理位置嵌入（position embeddings）的尺寸调整。位置嵌入是用于提供序列中
# 每个元素位置信息的重要组件，尤其在视觉Transformer模型中，位置嵌入帮助模型理解图像中不同补丁（patches）
# 的相对位置。当输入图像的尺寸与预训练模型中使用的图像尺寸不同时，就需要对位置嵌入进行调整，使其适应新的输入尺寸。
class InterpolateInitialPositionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # 初始化方法接收一个 config 对象，该对象包含了模型的配置信息，如补丁大小、图像尺寸等。
        self.config = config
    # 前向传播方法 forward
    # pos_embed：位置嵌入张量，形状为 (1,seq_len,hidden_size)。
    # img_size：当前输入图像的尺寸，是一个二元组 (height, width)。
    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        # 分离CLSToken的位置嵌入,这里把cls放前面,作为序列的开始
        cls_pos_embed = pos_embed[:, 0, :] # (1,d),因为中间是一个索引,维度会紧凑
        cls_pos_embed = cls_pos_embed[:, None] # (1,1,d)
        # 从位置嵌入张量的序列长度轴中抽取最后num_detection_tokens个元素（即检测Token的位置嵌入）。
        det_pos_embed = pos_embed[:, -self.config.num_detection_tokens :, :] # (1,num_det,d)
        # 从位置嵌入张量的序列长度的轴中抽取从第二个元素开始到最后 num_detection_tokens 之前的元素
        # （即图像块Token的位置嵌入），并将维度进行转置
        patch_pos_embed = pos_embed[:, 1 : -self.config.num_detection_tokens, :] # (1,num_patch,d)
        patch_pos_embed = patch_pos_embed.transpose(1, 2) # (1,d,num_patch)
        batch_size, hidden_size, seq_len = patch_pos_embed.shape  # 1,d,s
        # 获取默认的配置得到的patch_height, patch_width
        patch_height, patch_width = (
            self.config.image_size[0] // self.config.patch_size,
            self.config.image_size[1] // self.config.patch_size,
        )
        # 将图像块位置嵌入的形状从 (b,d,s) 转换为 (b,d, patch_height, patch_width)。
        patch_pos_embed = patch_pos_embed.view(batch_size, hidden_size, patch_height, patch_width)
        height, width = img_size # 获取输入图片的h,w
        # 根据输入图片的h,w计算新的new_patch_heigth,new_patch_width 
        new_patch_heigth, new_patch_width = height // self.config.patch_size, width // self.config.patch_size
        # 使用双三次插值（bicubic interpolation）调整补丁位置嵌入的大小，以适应当前输入图像的尺寸。
        # 当 align_corners=False 时，默认情况下，输入和输出的角点不再是对应关系。取而代之的是，插值操作会根据输入和输
        # 出的像素索引进行线性插值。这意味着插值操作会根据输入和输出的像素索引进行均匀分布的插值。这种方式通常更加灵活，并
        # 且避免了边缘像素的重复问题。
        # 设置 align_corners=False 可以避免在插值过程中出现的一些不自然的现象，尤其是在处理非均匀缩放或者非线性变换
        # 的情况下。使用 False 通常会产生更自然的插值结果，并且在大多数情况下是推荐的做法
        patch_pos_embed = nn.functional.interpolate( # (1,d,new_patch_heigth, new_patch_width)
            patch_pos_embed, size=(new_patch_heigth, new_patch_width), mode="bicubic", align_corners=False
        )
        # 重塑并拼接最终的位置嵌入：将调整后的补丁位置嵌入展平并转置，再与CLSToken和检测Token的位
        # 置嵌入拼接起来，形成最终的位置嵌入张量。
        # 2表示从索引2的轴开始展平,之后换轴(1,d,s)-->(1,s,d)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        # 之后在序列长度维度合并
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        return scale_pos_embed
class InterpolateMidPositionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config 
    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        # pos_embed:(5,1,65,256),分别表示层的深度,批次大小,序列长度,嵌入维度
        cls_pos_embed = pos_embed[:, :, 0, :] # (5,1,256)
        cls_pos_embed = cls_pos_embed[:, None] # (5,1,1,256)
        det_pos_embed = pos_embed[:, :, -self.config.num_detection_tokens :, :] #(5,1,num_det,256)
        patch_pos_embed = pos_embed[:, :, 1 : -self.config.num_detection_tokens, :] # (5,1,n_pat,256)
        patch_pos_embed = patch_pos_embed.transpose(2, 3) # (5,1,256,n_pat)
        depth, batch_size, hidden_size, seq_len = patch_pos_embed.shape # 自动拆包
        # 获取配置的patch_height, patch_width(就是高和宽上的分成的块数)
        patch_height, patch_width = (
            self.config.image_size[0] // self.config.patch_size,
            self.config.image_size[1] // self.config.patch_size,
        )
        # (5*1,256,patch_height, patch_width)
        patch_pos_embed = patch_pos_embed.view(depth * batch_size, hidden_size, patch_height, patch_width)
        height, width = img_size # 传入的输入图片的高和宽
        # 根据输入图片的高和宽计算new_patch_height, new_patch_width
        new_patch_height, new_patch_width = height // self.config.patch_size, width // self.config.patch_size
        # (depth*1,new_patch_heigth, new_patch_width)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_patch_height, new_patch_width), mode="bicubic", align_corners=False
        )
        # (depth*1,d,new_patch_heigth, new_patch_width)-->(depth*1,d,n_s)-->(depth*1,n_s,d)
        # 之后设置成内存连续,-->(depth,1,n_s,d)
        patch_pos_embed = (
            patch_pos_embed.flatten(2)
            .transpose(1, 2)
            .contiguous()
            .view(depth, batch_size, new_patch_height * new_patch_width, hidden_size)
        )
        # 在序列长度所在的轴合并
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=2)
        return scale_pos_embed
# 用于将输入图像转换为一系列补丁（patches）并嵌入到隐藏空间的模块。这个类首先将输入图像分割成补丁，然
# 后通过一个卷积层将这些补丁映射到隐藏空间中，从而得到补丁嵌入
class YolosPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 输入图像的尺寸，可以是单个整数（正方形图像）或一个二元组（非正方形图像）
        # patch_size：分割图像时的补丁尺寸，可以是单个整数或一个二元组。
        image_size, patch_size = config.image_size, config.patch_size # size,patch_size
        # 输入图像的通道数，通常是3（RGB图像）。隐藏空间的维度。
        num_channels, hidden_size = config.num_channels, config.hidden_size # c,d
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 根据输入图像尺寸和补丁尺寸计算得出的补丁总数。
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0]) # n_patches
        self.image_size = image_size 
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        # 一个卷积层，用于将补丁映射到隐藏空间。这样图像块都被一个嵌入向量表示
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        # 输入验证：确保输入图像的通道数与配置文件中设定的一致。
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 补丁嵌入生成,卷积操作：通过 projection 卷积层将输入图像分割成补丁，并将每个补丁映射到隐藏空间。
        # 展平操作：将卷积后的张量展平为 (batch_size, hidden_size, num_patches) 的形状。
        # 转置操作：将张量的形状从 (batch_size, hidden_size, num_patches) 转置为
        # (batch_size, num_patches, hidden_size)，这样可以更好地适应后续的 Transformer 层。
        # (b,d,h,w)-->(b,d,s)-->(b,s,d)
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
class YolosSelfAttention(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 嵌入维度必须能被头数整除
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.num_attention_heads = config.num_attention_heads # h
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # dk
        self.all_head_size = self.num_attention_heads * self.attention_head_size # d
        # q,k,v线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        # dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # (b,s,h,dk)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape) 
        return x.permute(0, 2, 1, 3) #(b,h,s,dk)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states) # (b,s,d)
        key_layer = self.transpose_for_scores(self.key(hidden_states)) # (b,h,s,dk)
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer) # (b,h,s,dk)
        
        # (b,h,q_len,dk)@(b,h,dk,k_len)-->(b,h,q_len,k_len)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 在k_len上进行归一化,值的大小表示和q_len上指定token的相似度,值越大越相似,能捕捉到
        # 长距离的关系
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # dropout
        attention_probs = self.dropout(attention_probs)
        # 如果有掩码,就对应元素相乘,这样对应掩码中0的位置会变成0,但是这样其他token的概率和
        # 将不会是1,这个和传统的方式不一样,传统方式是先掩码,之后归一化,填充之外token的概率
        # 和会是1
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # (b,h,q_len,k_len)@(b,h,v_len,dk)-->(b,h,q_len,dk)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (b,h,q_len,dk)-->(b,q_len,h,dk),之后设置成内存连续
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # (b,q_len,h,dk)-->（b,q_len,d）
        context_layer = context_layer.view(new_context_layer_shape)
        # 如果输出注意力权重就加上,否则只输出多头自注意力的输出
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
# 继承自YolosSelfAttention
class YolosSdpaSelfAttention(YolosSelfAttention):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__(config)
        # 设置dropout
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states)) # (b,h,s,dk)
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )
        #(b,h,q_len,dk)-->(b,q_len,h,dk)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) 
        context_layer = context_layer.view(new_context_layer_shape) # (b,s,d)
        return context_layer, None
class YolosSelfOutput(nn.Module): # 最后一个线性层
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
class YolosAttention(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        self.attention = YolosSelfAttention(config)
        self.output = YolosSelfOutput(config)
        # 存储已经修剪的头集合
        self.pruned_heads = set()
    def prune_heads(self, heads: Set[int]) -> None:
        #如果没有要修剪的头,直接返回
        if len(heads) == 0:
            return
        # 当前修剪过的头索引列表,未修剪的头索引对应的嵌入位置
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )
        # 修剪线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # 去除修剪的头,就是当前模型具有的头数
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        # 更新q,k,v的嵌入维度
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        # 合并集合,去重
        self.pruned_heads = self.pruned_heads.union(heads)
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        # 经过最后一个线性层转换,因为有修剪头,转换刚好又变成hidden_size
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
class YolosSdpaAttention(YolosAttention):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__(config)
        self.attention = YolosSdpaSelfAttention(config)
class YolosIntermediate(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (h,s,d)-->(h,s,h_d)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
class YolosOutput(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # (h,s,h_d)-->(h,s,d)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor # 残差
        return hidden_states
class YolosLayer(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 是否在进入前馈层前拆分样本的嵌入维度
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1 # 序列长度所在轴
        # 获取配置的自注意力
        self.attention = YOLOS_ATTENTION_CLASSES[config._attn_implementation](config)
        #前馈层前部分
        self.intermediate = YolosIntermediate(config)
        # 前馈层后部分
        self.output = YolosOutput(config)
        # 层标准化
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),# 在自注意力之前用标准化
            head_mask, # 传入的掩码
            # 是否输出注意力权重
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0] # 注意力输出
        outputs = self_attention_outputs[1:]  # 权重
        # 自注意力前后残差
        hidden_states = attention_output + hidden_states
        # 前馈层前的标准化
        layer_output = self.layernorm_after(hidden_states)
        # 前馈层(分成了两部分)
        layer_output = self.intermediate(layer_output)
        # 后部分,里面残差
        layer_output = self.output(layer_output, hidden_states)
        # 输出包括:编码器的输出+权重
        outputs = (layer_output,) + outputs
        return outputs
class YolosEncoder(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        self.config = config
        # 编码器层列表
        self.layer = nn.ModuleList([YolosLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False # 是否梯度检查
        # cls和patches和detection_tokens合起来的序列长度
        seq_length = (
            1 + (config.image_size[0] * config.image_size[1] // config.patch_size**2) + config.num_detection_tokens
        )
        # 位置嵌入:形状(depth,1,s,d)
        self.mid_position_embeddings = (
            nn.Parameter(
                torch.zeros(
                    config.num_hidden_layers - 1,
                    1,
                    seq_length,
                    config.hidden_size,
                )
            )
            if config.use_mid_position_embeddings
            else None
        )
        self.interpolation = InterpolateMidPositionEmbeddings(config) if config.use_mid_position_embeddings else None
    def forward(
        self,
        hidden_states: torch.Tensor,
        height,
        width,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 各个encoder层的输出
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None # 各个层的注意力概率权重
        # 传入参数形状(depth,1,s,d),输出:(depth,1,n_s,d)
        if self.config.use_mid_position_embeddings:
            interpolated_mid_position_embeddings = self.interpolation(self.mid_position_embeddings, (height, width))
        # 遍历编码器层列表的各个编码器层
        for i, layer_module in enumerate(self.layer):
            # 如果要输出隐藏状态,就加上,第一次的是嵌入的,之后是各个编码器層的输出
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 获取每个层对应的掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果设置了梯度检查,并且是训练状态
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数调用编码器层的forward获取输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else: # 否则如果设置不用梯度检查或者不是训练模式,就走常规的编码器层调用
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
            # 这个是每一层的编码器输出
            hidden_states = layer_outputs[0]
            # 如果设置要用中间位置嵌入
            if self.config.use_mid_position_embeddings:
                # 如果不是最后一层,那就让编码器层的输出加上下一层的位置嵌入,这个位置嵌入可学习
                if i < (self.config.num_hidden_layers - 1):
                    hidden_states = hidden_states + interpolated_mid_position_embeddings[i]
            # 如果要输出注意力权重,就加上
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        # 经过了各个层的编码处理,如果设置了输出隐藏状态,那就加上最后的输出
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # 如果不返回字典形式,就返回元组,分别是最后的编码器输出,各个编码器层的输出,各个编码器层的注意力权重
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回字典形式的输出
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class YolosPooler(nn.Module):
    def __init__(self, config: YolosConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    # 取出 [CLS] 标记的表示后，再经过一个 dense 层和激活函数的原因有几点：
    # 非线性变换：通过一个线性层加上非线性激活函数，可以引入非线性变换，这有助于模型捕捉更复杂的特征关系。
    # 线性层可以对 [CLS] 标记的表示进行重新组合，使其更适合后续的任务。
    # 特征重构：线性层可以对 [CLS] 标记的表示进行特征重构，使其更加符合特定任务的需求。例如，在分类任务中，可
    # 能需要将 [CLS] 标记的表示映射到一个新的空间，使其更具有区分性。
    # 模型容量：通过增加一层线性变换，可以增加模型的容量，使其能够学习更复杂的模式。尽管 [CLS] 标记已经携带了
    # 很多信息，但经过线性层后，模型可以进一步优化这些信息。
    # 正则化作用：在某些情况下，通过线性层和激活函数可以起到一定的正则化作用，帮助防止过拟合。
    # 为什么选择 Tanh 作为激活函数？
    # 范围限制：Tanh 函数的输出范围是 [-1, 1]，这有助于将 [CLS] 标记的表示标准化到一个固定范围内。
    # 这种范围内的输出有助于后续层的学习，特别是当输出用于分类或其他任务时。
    # 非线性特性：Tanh 具有很强的非线性特性，可以引入非线性变换，使得模型能够学习更复杂的特征关系。
    # 平滑性：Tanh 函数是连续可微的，这意味着它在整个定义域内都是平滑的，这有助于优化过程中的梯度流。
    # 历史原因：在早期的神经网络研究中，Tanh 被广泛使用，并且在很多情况下表现良好。虽然现代的一些激活函数如 
    # ReLU 更受欢迎，但在某些任务或模型结构中，Tanh 仍然表现出色。
    # 在神经网络中，“线性”通常指的是一个数学运算，其中输出是输入的线性组合。具体来说，在神经网络的上下文中：
    # 线性层（Dense Layer）：也称为全连接层（Fully Connected Layer），它对输入进行线性变换，即输出是输
    # 入与权重矩阵的乘积加上偏置项。形式上，可以表示为 y=Wx+by=Wx+b，其中 WW 是权重矩阵，xx 是输入向量
    # ，bb 是偏置项，yy 是输出向量。线性层的作用是将输入向量映射到另一个空间中。
    # 卷积层（Convolutional Layer）：卷积层对输入应用卷积操作，这本质上也是一个线性操作，因为它计算的是输
    # 入与卷积核（滤波器）之间的内积。形式上，卷积层的输出可以视为输入在局部区域与多个滤波器的线性组合。
    # 尽管这两个操作本质上都是线性的，但它们的作用和应用场景有所不同。线性层通常用于连接输入和输出之间的全局
    # 关系，而卷积层则专注于捕捉输入中的局部特征。
    # 线性层 和 卷积层 都是线性变换的例子，它们各自以不同的方式处理输入数据。
    # 非线性激活函数 如 Tanh 用于打破线性变换的限制，使模型能够学习到更复杂的特征和模式。
    def forward(self, hidden_states): # (b,s,d)
        # 取CLS的编码器输出作为池化输出,0指取出序列长度维度的第一个token,之后形状变成(b,d)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor) # 简单的线性转换
        pooled_output = self.activation(pooled_output) # 激活函数(Tanh)
        return pooled_output
class YolosPreTrainedModel(PreTrainedModel):
    config_class = YolosConfig
    base_model_prefix = "vit" # 模型前缀vit
    main_input_name = "pixel_values" # 输入名称
    supports_gradient_checkpointing = True # 是否支持梯度检查
    _no_split_modules = []
    _supports_sdpa = True # 支持sdpa
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        # 初始化模型各组件的权重
        # 如果是线性层,卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对w用正太分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有bias,bias用0向量初始化
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是标准化层
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
# 使用 Python 的 dataclass 装饰器定义的数据类，用于封装神经网络模型的输出。这类通常用于深度学习
# 框架中，比如 PyTorch，用来存储模型在前向传播过程中产生的各种张量
@dataclass
class BaseModelOutputWithPoolingAndProjection(ModelOutput):
    # 这是模型最后一层的隐藏状态，通常用于下游任务的输入。
    last_hidden_state: torch.FloatTensor = None
    # 这是对序列的第一个隐藏状态（通常是 [CLS] 令牌）经过一个线性层（池化层）后的输出
    pooler_output: torch.FloatTensor = None
    # 这是一个元组，包含了每一层的隐藏状态。这对于一些需要访问所有层隐藏状态的任务非常有用，
    # 比如做可视化、特征提取或其他研究目的。Optional 表示这是一个可选项，意味着在某些情况
    # 下（如为了节省内存），模型可能不会保存所有的隐藏状态。
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 这是一个元组，包含了每一层的注意力权重矩阵。在使用多头注意力机制的模型中，这些权重可
    # 以用来分析模型是如何关注输入的不同部分的。
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 这是一个元组，包含了投影后的状态。这个属性不是常见的输出属性，可能是特定于某些模型的定制属性。
    # 投影状态可能是在某些特定任务中需要对模型的隐藏状态进行某种变换后得到的结果
    projection_state: Optional[Tuple[torch.FloatTensor]] = None
# @add_start_docstrings(
#     "The bare YOLOS Model transformer outputting raw hidden-states without any specific head on top.",
#     YOLOS_START_DOCSTRING,
# )
class YolosModel(YolosPreTrainedModel):
    def __init__(self, config: YolosConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config
        self.embeddings = YolosEmbeddings(config) # token and pos嵌入
        self.encoder = YolosEncoder(config) # 编码器
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) #标准化层
        self.pooler = YolosPooler(config) if add_pooling_layer else None # 池化层
        # Initialize weights and apply final processing
        self.post_init() # 初始化权重
    # 获取图像块嵌入
    def get_input_embeddings(self) -> YolosPatchEmbeddings:
        return self.embeddings.patch_embeddings
    # 修剪头,layer:层索引,调用指定层的多头注意力的修剪头方法
    # heads_to_prune: 这是一个字典，其键是整数（表示层的索引），值是一个整数列表（
    # 表示要修剪的注意力头的索引）。
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    # @add_start_docstrings_to_model_forward(YOLOS_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPooling,
    #     config_class=_CONFIG_FOR_DOC,
    #     modality="vision",
    #     expected_output=_EXPECTED_OUTPUT_SHAPE,
    # )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 先看传参,再看配置,根据这个设置是否输出权重,各个层的输出,是否返回字典形式
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 输入检验
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
       # 对注意力头的掩码
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # token和位置嵌入
        embedding_output = self.embeddings(pixel_values)
        # 经过编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            height=pixel_values.shape[-2],
            width=pixel_values.shape[-1],
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 编码器输出
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output) # 层标准化
        # 池花输出(b,d)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        # 如果不返回字典,就返回元组
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        # 返回字典形式的输出
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class YolosMLPPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers # 层数
        h = [hidden_dim] * (num_layers - 1) # (num_layers - 1)个元素的列表
        # 线性层列表
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x): 
        for i, layer in enumerate(self.layers):
            # 线性转换(input_dim)-->hidden_dim-->...-->(out_dim)
            # 在每个线性层之后用非线性激活函数,最后一层除外
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
# YolosHungarianMatcher 类是一个用于计算目标检测中预测结果与真实标签之间最佳匹配的类。这个类利用了匈牙利算法（
# Hungarian algorithm）来找到预测框（predictions）与真实框（ground-truth boxes）之间的最优配对，从而帮
# 助计算损失函数。
class YolosHungarianMatcher(nn.Module):
    # 初始化方法接收三个损失权重参数：class_cost、bbox_cost 和 giou_cost。这三个权重分别表示分类权重、
    # 边界框回归权重和 GIoU（Generalized Intersection over Union）权重。它们决定了不同权重在总损失中的
    # 比重。如果所有成本都为零，则会抛出一个 ValueError。
    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])
        self.class_cost = class_cost 
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # 不能全是0
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")
    # 该方法计算预测框与真实框之间的匹配，并返回匹配索引。使用 @torch.no_grad() 装饰器是因为这个
    # 方法不涉及梯度计算，主要用于评估和匹配，因此关闭自动梯度计算可以节省内存和加快计算速度。
    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size, num_queries = outputs["logits"].shape[:2] # b,num_det
        # We flatten to compute the cost matrices in a batch
        # 形状调整：将输出的类别概率和边界框预测展平成二维数组。在类别维度归一化
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # 对边界框的预测
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        # 合并真实标签和边界框：将真实标签和边界框在批次维度上合并。
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # 计算分类损失：计算分类的概率分布与真实标签之间的负对数似然近似值。
        class_cost = -out_prob[:, target_ids]
        # 计算边界框回归成本：计算预测边界框与真实边界框之间的 L1 距离。
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
        # 计算 GIou 成本：计算预测边界框与真实边界框之间的负 GIou。
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))
        # 组合成本矩阵：根据指定的成本系数组合所有成本。
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        # 分割并求解匹配：将成本矩阵分割成每个样本的成本矩阵，并使用匈牙利算法找到每个样本的最佳匹配。
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu() 
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        # 返回匹配索引：返回每个样本的匹配索引对。
        # YolosHungarianMatcher 类通过计算不同类型的成本（分类、边界框回归和 GIou），然后利用匈牙利算法来寻找预测框与真实框
        # 之间的最佳匹配。这个类主要用于目标检测任务中的损失计算，特别是在训练过程中，帮助优化器调整模型参数以减少预测误差。
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# YolosLoss 类定义了一个用于 YOLOS（YOLO Segmentation）或类似目标检测任务的损失计算模块。
# 这个类负责计算各种损失，包括分类损失、边界框回归损失、GIoU 损失、以及可选的掩码
# 损失等，并且它还考虑了匹配问题，通过匈牙利匹配器来找到预测框与真实框的最佳匹配。
# YolosLoss 类通过多种损失计算方式综合评估模型的预测质量，并且通过匈牙利匹配器来解决多预测框与多真
# 实框之间的匹配问题。这有助于模型在训练过程中不断调整，以最小化各类损失，提高目标检测的准确性和鲁棒性。
class YolosLoss(nn.Module):
    # 这个类计算了 YolosForObjectDetection/YolosForSegmentation 的损失。该过程分为两个步骤：
    # 1）我们计算真实框（ground truth boxes）与模型输出之间的匈牙利匹配；2）我们监督每一对匹配的
    # 真实框/预测框（监督类别和框）。
    # 关于 num_classes 参数的一个注释（摘自原始仓库中的 detr.py 文件）：“num_classes 参数的命
    # 名有些误导性。实际上，它对应于 max_obj_id + 1，其中 max_obj_id 是你的数据集中类的最大 ID。
    # 例如，在 COCO 数据集中，max_obj_id 是 90，所以我们传递的 num_classes 是 91。再举一个例子
    # ，对于只有一个类（ID 为 1）的数据集，你应该传递的 num_classes 是 2（max_obj_id + 1）。有
    # 关此问题的更多细节
    def __init__(self, matcher, num_classes, eos_coef, losses):
        super().__init__()
        self.matcher = matcher #  匈牙利匹配器实例，用于找到预测框与真实框的最佳匹配。
        self.num_classes = num_classes # 目标类别数量（不包括背景类别）。
        self.eos_coef = eos_coef # 无物体（背景）类别的权重系数。
        self.losses = losses # 一个字符串列表，定义了要计算的损失类型。
        # 构造函数还创建了一个 empty_weight 张量，用于分类损失计算时的权重分配。
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef # 最后一个是无类别的权重
        # 注册到内存,会随模型保存而保存
        self.register_buffer("empty_weight", empty_weight)
    # removed logging parameter, which was part of the original implementation
    # 损失计算方法
    # 此方法计算分类损失（cross entropy loss），针对每个预测框的类别标签进行监督。
    def loss_labels(self, outputs, targets, indices, num_boxes):
        # logits用于分类预测
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        # 模型对类别预测的候选
        source_logits = outputs["logits"]
        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    # 此方法计算预测框数量与实际框数量之间的差异，作为统计信息而非真正的损失项。
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses
    # 此方法计算边界框的回归损失（L1 loss）和 GIoU 损失，监督预测框的位置准确性。
    def loss_boxes(self, outputs, targets, indices, num_boxes):
       
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses
    # 此方法计算掩码损失（focal loss 和 dice loss），用于分割任务中的掩码预测监督。
    def loss_masks(self, outputs, targets, indices, num_boxes):
       
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # upsample predictions to the target size
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses
    # 此方法根据匹配索引重新排列预测框，使其与真实框相对应。
    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx
    # 此方法根据匹配索引重新排列真实框，使其与预测框相对应。
    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)
    # 此方法是整个损失计算的核心，它首先移除辅助输出（auxiliary outputs），然后计算主输出的损失，
    # 最后如果存在辅助输出，则计算辅助输出的损失。
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                num_boxes = reduce(num_boxes)
                world_size = PartialState().num_processes
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
# @add_start_docstrings(
#     """
#     YOLOS Model (consisting of a ViT encoder) with object detection heads on top, for tasks such as COCO detection.
#     """,
#     YOLOS_START_DOCSTRING,
# )
# 实现了对象检测所需的组件，包括编码器、分类头和边界框预测头等。此外，它还处理了损失计算以及模型输出的组织。
class YolosForObjectDetection(YolosPreTrainedModel):
    def __init__(self, config: YolosConfig):
        super().__init__(config)
        # self.vit：YOLOS 基于 ViT 的编码器模型。
        self.vit = YolosModel(config, add_pooling_layer=False)
        # self.class_labels_classifier：分类头，用于预测每个检测框所属的类别,额外添加了一个无检测目标的类别
        self.class_labels_classifier = YolosMLPPredictionHead(
            input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=config.num_labels + 1, num_layers=3
        )
        # self.bbox_predictor：边界框预测头，用于预测每个检测框的位置信息。(x1,y1,x2,y2)
        self.bbox_predictor = YolosMLPPredictionHead(
            input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=4, num_layers=3
        )
        # Initialize weights and apply final processing
        self.post_init()
    # 辅助方法 _set_aux_loss
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # 此方法用于处理中间层的辅助损失输出，以便在使用 TorchScript 导出模型时避免类型错误。
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    # @add_start_docstrings_to_model_forward(YOLOS_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=YolosObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward( # 前向传播方法 forward
        self,
        pixel_values: torch.FloatTensor,# 输入的像素值，即图像数据。
        labels: Optional[List[Dict]] = None, # 可选的标签列表，用于训练时提供 ground truth。
        output_attentions: Optional[bool] = None,# 是否输出注意力权重
        output_hidden_states: Optional[bool] = None, # 是否输出隐藏状态。
        return_dict: Optional[bool] = None, # 是否以字典的形式返回输出。
    ) -> Union[Tuple, YolosObjectDetectionOutput]:
        # 初始化返回字典标志,根据传入的 return_dict 参数或者配置文件中的 use_return_dict 
        # 属性来确定是否以字典形式返回。
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 编码器输出：输入图像数据通过 YOLOS 编码器模型 self.vit，获取隐藏状态。
        outputs = self.vit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] # 编码器输出(b,s,d)
        # 获取检测token的输出 (b,num_det,d)
        sequence_output = sequence_output[:, -self.config.num_detection_tokens :, :]
        # Class logits + predicted bounding boxes
        # 分类和边界框预测：使用分类头 self.class_labels_classifier 和边界框预测头
        # self.bbox_predictor 进行预测。
        logits = self.class_labels_classifier(sequence_output) # (b,num_det,num_labels + 1)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid() # (b,num_det,4)
        loss, loss_dict, auxiliary_outputs = None, None, None
        # 计算损失：如果提供了标签 labels，则计算损失。
        if labels is not None:
            # 创建匹配器 YolosHungarianMatcher 
            matcher = YolosHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # 损失包括类别损失,边界框损失,
            losses = ["labels", "boxes", "cardinality"]
            # 损失计算标准 YolosLoss。
            criterion = YolosLoss( # 损失函数
                matcher=matcher,
                num_classes=self.config.num_labels, # 要预测的类别数
                eos_coef=self.config.eos_coefficient,  # eos 系数
                losses=losses, 
            )
            criterion.to(self.device) # 交给设备
            outputs_loss = {}
            outputs_loss["logits"] = logits # 类别预测
            outputs_loss["pred_boxes"] = pred_boxes # 边界框角点预测
            # 辅助损失：如果配置中允许辅助损失 auxiliary_loss，则计算中间层的辅助损失。
            if self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs
            
            loss_dict = criterion(outputs_loss, labels)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output
        # 根据 return_dict 参数决定返回格式。
        return YolosObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
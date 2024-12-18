# 用于生成特征图上的锚框的模块。在目标检测任务中，锚框是预设的边界框，它们会在训练过程中与真实的目标框
# 进行匹配来预测目标的位置和类别.这个类的主要作用是在给定的特征图上初始化一系列预设的边界框，这些框将在后续
# 的目标检测网络中用于定位潜在的目标对象.对于不同尺寸的特征图，可以有不同的锚框大小和宽高比,这有助于检测不
# 同尺度的对象。
class AnchorGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }
    # 这个是初始化不同特征图上的锚框基准
    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),), # 高宽比
    ):
        super().__init__()
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio) for size, aspect_ratio in zip(sizes, aspect_ratios)
        ]
    # 这个方法根据给定的一组尺寸和宽高比(可以多个)生成零中心化(意思就是中心位置是原点)的锚框。尺寸决定
    # 了锚框的大小，而宽高比决定了锚框的形状。对于每一个尺寸和宽高比的组合，都会生成一个锚框。
    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        # 尺度和高宽比转换成Tensor类型
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        # 每个尺寸都对应3个不同形状的锚框(x1,y1,x2,y2)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round() # 四舍五入
    # 此方法用于设置锚框的数据类型和设备。它会遍历所有的 cell_anchors 并将它们移到指定的设备上，
    # 并转换成指定的数据类型。
    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device) for cell_anchor in self.cell_anchors]
    # 返回每个位置处的锚框数量。这是通过计算每个 sizes 和 aspect_ratios 组合的数量得出的。
    # [15]
    def num_anchors_per_location(self) -> List[int]:
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]
    # 根据网格大小和步幅 在每个特征图的位置上生成锚框。它使用了网格中的每个点作为锚框的中心
    # 并应用了之前生成的 cell_anchors 来创建一个网格中的所有可能锚框。
    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        anchors = [] # 用来保存对应大小图片的5个不同尺寸的特征图下的锚框
        cell_anchors = self.cell_anchors
        torch._assert(cell_anchors is not None, "cell_anchors should not be None")
        torch._assert(
            len(grid_sizes) == len(strides) == len(cell_anchors),
            "Anchors should be Tuple[Tuple[int]] because each feature "
            "map could potentially have different sizes and aspect ratios. "
            "There needs to be a match between the number of "
            "feature maps passed and the number of sizes / aspect ratios specified.",
        )
        # 遍历指定尺寸特征图下的步长,网格图(特征图)大小,对应的基准锚框
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device
            # 获取相对于原始图大小的网格锚框坐标
            shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            # torch.Size([4096, 1, 4]) torch.Size([1, 3, 4]) 为了广播
            # 4096是指有这么多锚框中心点,3是一个锚框中心点对应3个锚框
            # [12288, 4],对于第一个(64,64)尺寸的特征图有这么多的锚框
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors
    # 这是模型的前向传播方法。它接收一个图像列表 (image_list) 和一系列特征图 (feature_maps)。
    # 首先计算每个特征图的网格大小 (grid_sizes)，然后获取输入图像的尺寸。接着为每个特征图计算步
    # 幅 (strides)，并将 cell_anchors 设置到正确的数据类型和设备上。之后调用 grid_anchors
    # 方法来生成所有特征图上的锚框，并且为图像列表中的每个图像创建一个包含所有锚框的列表。
    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        # 网格图大小就是指定的特征图宽高
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps] 
        image_size = image_list.tensors.shape[-2:] # 图片大小
        # 获取特征图的数据类型和设备
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        # 步长:指定尺寸的特征图想对于原图的下采样步幅
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
            ]
            for g in grid_sizes
        ]
        self.set_cell_anchors(dtype, device) # 设置基准锚框的设备等
        # 获取所有尺寸特征图的所有锚框
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        for _ in range(len(image_list.image_sizes)):
            # 某个图片对应的不同尺寸特征图下的锚框
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        # anchors对应批次内不同图片的锚框,每个列表元素是图片对应的不同特征图下的所有锚框
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors

# 实现多尺度的 ROIAlign 池化。__annotations__ 字段用于类内部的类型注解。
class MultiScaleRoIAlign(nn.Module):
    __annotations__ = {"scales": Optional[List[float]], "map_levels": Optional[LevelMapper]}

    def __init__(
        self,
        featmap_names: List[str], # 指定将用于池化的特征图的名称。
        # 输出大小，可以是一个整数或一个元组，表示池化后的特征图大小。
        output_size: Union[int, Tuple[int], List[int]], 
        sampling_ratio: int, # 用于 RoIAlign 的采样比率。
        *,
        canonical_scale: int = 224, # 标准尺度，默认为 224。
        canonical_level: int = 4, # 标准层级，默认为 4。
    ):
        # 内部变量初始化
        super().__init__()
        _log_api_usage_once(self)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names # 存储传入的特征图名称。
        self.sampling_ratio = sampling_ratio # 存储传入的采样比率。
        self.output_size = tuple(output_size) # 存储传入的输出大小。
        self.scales = None # 初始化为 None，将在 forward 方法中计算。
        self.map_levels = None
        self.canonical_scale = canonical_scale # 存储传入的标准尺度和层级。
        self.canonical_level = canonical_level
    # x：一个有序字典，键为特征图名称，值为特征图张量。boxes：一个列表，包含每个图
    # 像的边界框张量。image_shapes：一个列表，包含每个图像的原始大小。
    def forward(
        self,
        x: Dict[str, Tensor],
        boxes: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tensor:       
        # 筛选特征图：使用 _filter_input 函数筛选出指定名称的特征图。
        x_filtered = _filter_input(x, self.featmap_names)
        # 计算 scales 和 map_levels：如果 scales 或 map_levels 为 None，则使
        # 用 _setup_scales 函数计算这些值。
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = _setup_scales(
                x_filtered, image_shapes, self.canonical_scale, self.canonical_level
            )
        # 执行多尺度 ROIAlign 池化：使用 _multiscale_roi_align 函数执行多尺度 
        # ROIAlign 池化操作，并返回结果。
        return _multiscale_roi_align(
            x_filtered,
            boxes,
            self.output_size,
            self.sampling_ratio,
            self.scales,
            self.map_levels,
        )
    # 这个方法返回一个字符串，描述了当前实例的状态，方便调试和打印。
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(featmap_names={self.featmap_names}, "
            f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})"
        )

# RPNHead 的主要任务是从给定的特征图中预测物体位置的边界框（bbox）和分类概率
class RPNHead(nn.Module):
    _version = 2 # 这个版本号用于向后兼容，在加载旧的模型状态字典时可能会用到
    # in_channels: 输入特征图的通道数,num_anchors: 在特征图的每个位置上使用的锚框数量。
    # conv_depth: 卷积层的深度
    def __init__(self, in_channels: int, num_anchors: int, conv_depth=1) -> None:
        super().__init__()
        # 构造了一个包含多个卷积层的序列 self.conv，用于特征提取
        convs = []
        for _ in range(conv_depth):
            convs.append(Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None))
        self.conv = nn.Sequential(*convs)
        # 点卷积
        # cls_logits: 用于预测每个锚框是否包含目标物体的概率。box_pred:用于预测每个锚框的边界框偏移量。
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        # 对所有卷积层的权重进行了初始化，遵循标准正态分布（标准差为0.01），并对偏置项进行了零初始化。
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # 这个方法用于在加载模型时处理版本兼容性问题。如果状态字典的版本低于当前版本，则需要将旧的键名映射到
        # 新的键名上。在这个例子中，如果是旧版本的状态字典，则将 conv.weight 和 conv.bias 映射到 
        # conv.0.0.weight 和 conv.0.0.bias
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            for type in ["weight", "bias"]:
                old_key = f"{prefix}conv.{type}"
                new_key = f"{prefix}conv.0.0.{type}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    # 前向传播方法接收一个包含多个特征图的列表 x，对于每个特征图，先通过卷积序列 self.conv 进行特征提取，
    # 然后分别通过 cls_logits 和 bbox_pred 卷积层得到分类分数和边界框预测。最后，返回两个列表，分别包含
    # 每个特征图的分类分数和边界框预测结果。
    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
# reference_boxes 通常指的是真实框（ground truth boxes），即标注数据中提供的物体实
# 际位置。而 proposals 则是指由模型生成的一组候选区域（region proposals），这些候选区域
# 可能是由诸如 RPN（Region Proposal Network）这样的组件生成的，用于指示可能包含目标对象的区域。
# 在目标检测算法中，“proposals” 和“anchors”（锚框）这两个概念有时会被混淆，但实际上它们在不同的
# 阶段扮演着不同的角色。让我来澄清一下它们之间的区别：
# 锚框是在特征图的每个位置预先定义的一组矩形框，它们有不同的尺度和宽高比。这些框是为了捕捉图像中不同大
# 小和形状的对象而设计的。例如，在 Faster R-CNN 或 SSD 中，会在多个尺度的特征图上设置多个默认的锚框
# 。锚框通常是静态的，即在训练和推理过程中保持不变。
# 区域提议（proposals）是由模型的一个子模块（如 RPN 在 Faster R-CNN 中）生成的，它是基于锚框经过初步
# 筛选和调整后得到的结果。RPN 使用卷积层来预测每个锚框的分类概率（是否包含物体）以及边界框的调整量。然后，
# RPN 使用非极大值抑制（NMS）等方法从中选择出高质量的提议框，这些提议框更有可能包含目标对象。
# “proposals” 是经过一定程度处理后的锚框，它们比原始的锚框更接近真实的物体边界框。
@torch.jit._script_if_tracing
def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    # perform some unpacking to make it JIT-fusion friendly
    # 这段英文注释的意思是：“进行一些解包操作，使得它对 JIT 融合友好。”
    # 这里的“JIT”指的是 Just-In-Time 编译技术，而“fusion”指的是操作融合。PyTorch 的 JIT 编译器旨在优化
    # 代码执行效率，减少不必要的内存开销，并提高性能。在 PyTorch 中，JIT 编译器可以将一系列的操作合并成一个
    # 单一的操作，从而减少调用开销和提高计算效率。
    # 在这段代码中，weights 是一个包含四个元素的张量，每个元素代表一个权重。将这些权重单独赋值给变量 wx, wy,
    # ww,wh，可以使得后续的计算更加简洁，并且有利于 JIT 编译器的优化。这种做法减少了在运行时的动态操作,使得编译器
    # 更容易识别和优化这些固定的计算模式。
    # 具体来说，这样做有几个好处：
    # 简化计算表达式：在后续的计算中可以直接使用这些变量名，而不是每次都需要访问张量的索引。
    # 性能优化：JIT 编译器可以更好地理解这些操作，并将它们融合为更高效的内核
    # 这段注释的意思是通过提前解包权重张量，使得接下来的计算逻辑更清晰，同时也更容易被 JIT 编译器优化。
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]
    # 提议框,unsqueeze又把它变成了二维张量
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)
    # 真实框
    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)
    # implementation starts here
    # 具体的边界框编码计算从这一行开始。计算提议框的宽度和高度.计算提议框的中心点坐标
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights
    # 计算真实框的宽度和高度,计算真实框的中心点坐标
    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights
    # targets_dx 计算了真实框中心点 x 坐标与提议框中心点 x 坐标的差值，并除以提议框的宽度，然后乘以权重 wx。
    # targets_dy 计算了真实框中心点 y 坐标与提议框中心点 y 坐标的差值，并除以提议框的高度，然后乘以权重 wy
    # targets_dw 计算了真实框宽度与提议框宽度的比例的对数值，并乘以权重 ww
    # targets_dh 计算了真实框高度与提议框高度的比例的对数值，并乘以权重 wh。
    # 这些权重（wx, wy, ww, wh）通常是为了平衡不同维度的影响。例如，宽度和高度的变化可能需要不同的缩放因子来
    # 确保回归目标在训练过程中具有良好的数值稳定性和收敛性。
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)
    # 最终，所有计算出来的偏移量 targets_dx, targets_dy, targets_dw, targets_dh 都被拼接到一起，形成
    # 一个完整的偏移量向量 targets。
    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets

# 假设我们有两个图像，每个图像有以下的真实框和提议框：
# 图像 1：eference_boxes_1：形状为 [3, 4] （3 个真实框）,proposals_1：形状为 [5, 4] （5 个提议框）
# 图像 2：reference_boxes_2：形状为 [2, 4] （2 个真实框）,roposals_2：形状为 [4, 4] （4 个提议框）
# 那么 reference_boxes 和 proposals 分别是：
# reference_boxes：形状为 [5, 4] （3 + 2 个真实框）
# proposals：形状为 [9, 4] （5 + 4 个提议框）
# boxes_per_image 为 [3, 2]，表示第一个图像有 3 个真实框，第二个图像有 2 个真实框。
# 最终，targets 会被分割成两个张量：第一个张量：形状为 [3, 4] （对应第一个图像的 3 个编码偏移量）
# 第二个张量：形状为 [2, 4] （对应第二个图像的 2 个编码偏移量）
# 这样就实现了将每个图像的编码偏移量恢复到原始的图像分割结构中。这种方法使得我们可以批量处理多个图像的数据，
# 同时还能保留每个图像的独立性。
class BoxCoder:
    # weights：一个包含四个浮点数的元组，用于缩放每个回归目标的权重。bbox_xform_clip：一个用于限制对数变换最
    # 大值的浮点数，默认值为 math.log(1000.0 / 16)。
    def __init__(
        self, weights: Tuple[float, float, float, float], bbox_xform_clip: float = math.log(1000.0 / 16)
    ) -> None:
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip
    # 编码方法,reference_boxes：一个包含真实框（ground truth boxes）的列表。
    # proposals：一个包含提议框（proposals）的列表。输出：一个包含编码后偏移量的列表。
    # 用于将一组真实框（reference_boxes）和一组提议框（proposals）编码成偏移量，以便用于训练回归器。
    def encode(self, reference_boxes: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
        # 统计每个图像的真实框数量
        # 这一行代码计算每个图像中真实框的数量，并存储在一个列表 boxes_per_image 中。假设 reference_boxes 
        # 是一个列表，其中每个元素是一个张量，表示一个图像中的真实框坐标。
        boxes_per_image = [len(b) for b in reference_boxes]
        # 将所有真实框连接成一个大张量：这一行代码将 reference_boxes 中的所有张量沿着第 0 维（即行方向）连接成
        # 一个新的张量。这样可以方便地一次性处理所有图像的真实框。
        reference_boxes = torch.cat(reference_boxes, dim=0)
        # 将所有提议框连接成一个大张量：同样，这一行代码将 proposals 中的所有张量沿着第 0 维连接成一个新的
        # 张量。这样可以方便地一次性处理所有图像的提议框。
        proposals = torch.cat(proposals, dim=0)
        # 调用 encode_single 方法进行编码：这一行代码调用 BoxCoder 类的 encode_single 方法，将连接后的 
        # reference_boxes 和 proposals 编码成偏移量。encode_single 方法返回一个张量，表示编码后的偏移量。
        targets = self.encode_single(reference_boxes, proposals)
        # 按原图像分割编码后的偏移量：最后，这一行代码将编码后的偏移量按照 boxes_per_image 列表中的数量分割成多个
        # 张量，每个张量对应一个图像的编码偏移量。split 方法的第一个参数是分割的长度列表，第二个参数是沿着哪个维度
        # 进行分割（在这里是第 0 维）。
        return targets.split(boxes_per_image, 0)
    # reference_boxes：一个包含真实框的张量。proposals：一个包含提议框的张量。
    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)
        return targets
    # rel_codes：一个包含回归目标（即偏移量）的张量。boxes：一个包含原始边界框的列表。
    # 输出：一个包含解码后边界框坐标的张量。
    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        torch._assert(
            isinstance(boxes, (list, tuple)),
            "This function expects boxes of type list or tuple.",
        )
        torch._assert(
            isinstance(rel_codes, torch.Tensor),
            "This function expects rel_codes of type torch.Tensor.",
        )
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes
    # rel_codes：一个包含回归目标（即偏移量）的张量。boxes：一个包含原始边界框的张量。
    # 输出：一个包含单个解码后边界框坐标的张量。
    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        boxes = boxes.to(rel_codes.dtype)
        # 计算提议框的宽度和高度，以及中心点坐标。
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        # 将编码后的偏移量除以相应的权重。
        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh
        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        # 对宽度和高度的对数变换结果进行裁剪，以防止过大值进入 torch.exp()。
        # 计算预测框的中心点坐标和宽度、高度。
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # 根据中心点坐标和宽度、高度计算预测框的四个顶点坐标。
        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_x + c_to_c_w
        pred_boxes4 = pred_ctr_y + c_to_c_h
        # 将计算出的四个顶点坐标堆叠并展平，形成最终的预测框坐标。
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes
# 这个 Matcher 类是用于将预测的边界框（proposals）与真实框（ground-truth boxes）进行匹配的工具。
# 匹配的过程基于一个质量矩阵（match_quality_matrix），该矩阵描述了每一对（真实框，预测框）的质量度
# 量（例如 IoU 交并比）。
class Matcher:
    
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }
    # high_threshold：高质量阈值，大于等于这个值的匹配被认为是高质量匹配。low_threshold：低质量阈值，
    # 小于这个值的匹配被认为是低质量匹配。allow_low_quality_matches：布尔值，如果为 True，则允许低质量匹配。
    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False) -> None:
       
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        torch._assert(low_threshold <= high_threshold, "low_threshold should be <= high_threshold")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches
    # match_quality_matrix：一个 M 行 N 列的张量，其中 M 是真实框的数量，N 是预测框的数量。每
    # 个元素表示一个真实框和一个预测框之间的质量度量。
    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No ground-truth boxes available for one of the images during training")
            else:
                raise ValueError("No proposal boxes available for one of the images during training")
        
        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        # 计算最高质量匹配,matched_vals：每个预测框的最大质量度量值。matches：每个预测框的最佳
        # 匹配真实框的索引。
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None  # type: ignore[assignment]
        # 处理低质量匹配,below_low_threshold：质量度量低于低阈值的预测框
        # between_thresholds：质量度量介于低阈值和高阈值之间的预测框。
        # matches：根据阈值将预测框标记为 BELOW_LOW_THRESHOLD 或 BETWEEN_THRESHOLDS
        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS
        # 允许低质量匹配（如果允许）
        if self.allow_low_quality_matches:
            if all_matches is None:
                torch._assert(False, "all_matches should not be None")
            else:
                self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)
        return matches

    def set_low_quality_matches_(self, matches: Tensor, all_matches: Tensor, match_quality_matrix: Tensor) -> None:
        # 找到最高质量匹配的真实框：
        # For each gt, find the prediction with which it has the highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties
        # 找到最高质量匹配的预测框：
        gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])
        # Example gt_pred_pairs_of_highest_quality:
        # (tensor([0, 1, 1, 2, 2, 3, 3, 4, 5, 5]),
        #  tensor([39796, 32055, 32070, 39190, 40255, 40390, 41455, 45470, 45325, 46390]))
        # Each element in the first tensor is a gt index, and each element in second tensor is a prediction index
        # Note how gt items 1, 2, 3, and 5 each have two ties
        # 更新预测框的匹配索引：
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

# 它的主要功能是在每张图片的候选区域（proposals）中选取固定比例的正样本（positives）和负样本（negatives）
# ，以便在后续的训练过程中达到样本平衡的目的。这在目标检测算法（如 Faster R-CNN）中是非常常见的做法，尤其是
# 在处理类别不平衡的问题时。
class BalancedPositiveNegativeSampler:
    # batch_size_per_image: 每张图片选取的样本总数。positive_fraction: 正样本在总样本中的比例。
    def __init__(self, batch_size_per_image: int, positive_fraction: float) -> None:
        
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
    # 这行定义了类的一个可调用方法 __call__，它接受一个列表 matched_idxs，其中每个元素都是一个 Tensor，
    # 每个列表元素表示每个图片中候选区域（proposals）与真实框（ground truth boxes）的匹配结果。
    # 方法返回一个元组，包含两个列表，一个是正样本的索引掩码列表，另一个是负样本的索引掩码列表。
    def __call__(self, matched_idxs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        # 初始化两个空列表 pos_idx 和 neg_idx，分别用于存储每张图片中正样本和负样本的索引掩码。
        pos_idx = []
        neg_idx = []
        # 这是一个循环，遍历输入列表 matched_idxs 中的每个元素（每个图片的匹配索引张量）。
        for matched_idxs_per_image in matched_idxs:
            # torch.where 函数用于筛选出满足条件的索引。
            # positive 变量存储了匹配索引大于等于1的索引，这些索引对应于正样本。
            # negative 变量存储了匹配索引等于0的索引，这些索引对应于负样本。
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]
            # 计算期望的正样本数量。self.batch_size_per_image 是每张图片的最大样本数
            # self.positive_fraction 是正样本的比例。
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            # 确保正样本的数量不超过可用的正样本数量。positive.numel() 返回正样本索引的数量。
            num_pos = min(positive.numel(), num_pos)
            # 计算期望的负样本数量，这是总样本数减去正样本数。
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            # 确保负样本的数量不超过可用的负样本数量。negative.numel() 返回负样本索引的数量。
            num_neg = min(negative.numel(), num_neg)
            # randomly select positive and negative examples
            # 使用 torch.randperm 随机排列正样本和负样本的索引，并从中选择前 num_pos 个正样本索引
            # 和前 num_neg 个负样本索引。
            # device=positive.device 确保随机排列发生在与原始索引相同的设备（CPU或GPU）上。
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            # 使用随机排列后的索引来选择实际的正样本和负样本索引。
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]
            # create binary mask from indices
            # 创建与 matched_idxs_per_image 同形状的二进制掩码张量，初始值全部为0。
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            # 将选定的正样本和负样本索引在掩码中标记为1。
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            # 将每张图片的正样本和负样本掩码追加到对应的列表中。
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        # 返回包含所有图片的正样本和负样本掩码的列表。
        return pos_idx, neg_idx

# 用于实现 Faster R-CNN 中的区域提案网络（RPN）。RPN 的主要任务是在给定的特征图上生成候选区域（
# region proposals），这些候选区域将被传递给后续的检测网络进行进一步处理。
class RegionProposalNetwork(torch.nn.Module):
    # __annotations__：这个字典用于类的类型注解，它指定了类的一些属性的类型。这些注解主要用
    # 于提高代码的可读性和维护性，以及方便静态类型检查工具理解和优化。
    # box_coder：用于编码和解码边界框的位置信息。
    # proposal_matcher：用于匹配候选区域（proposals）与真实框（ground truth boxes）。
    # fg_bg_sampler：用于从候选区域中选择正样本和负样本。
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }
    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        # Faster-RCNN Training
        fg_iou_thresh: float,# 候选区域与真实框的 IoU 超过此阈值时，被认为是正样本。
        bg_iou_thresh: float, # 候选区域与真实框的 IoU 低于此阈值时，被认为是负样本。
        batch_size_per_image: int, # 控制每张图片中用于训练的正负样本总数。
        positive_fraction: float, # 每批数据中正样本的比例。
        # Faster-RCNN Inference
        # 在进行 NMS 之前，选择前 n 个得分最高的候选区域。
        pre_nms_top_n: Dict[str, int], # NMS（非极大值抑制）之前的候选区域数量。
        post_nms_top_n: Dict[str, int], # 在 NMS 过程之后，保留前 n 个得分最高的候选区域。
        # 在 NMS 过程中，如果两个候选区域的 IoU 超过此阈值，则保留得分较高的一个。
        nms_thresh: float,
        # 用于过滤低得分候选区域的阈值，默认为 0.0。在推理阶段，去除得分低于此阈值的候选区域。
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        # 在特征图的每个位置生成一组默认的边界框，这些边界框将在后续步骤中用于生成候选区域。
        self.anchor_generator = anchor_generator
        # 对锚框进行分类和回归预测，生成初步的候选区域。
        self.head = head
        # 将边界框的位置信息转换为相对于锚框的位置偏移，或将位置偏移转换回边界框坐标。
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        # used during training
        # 计算两个边界框之间的 IoU，用于匹配候选区域与真实框。
        self.box_similarity = box_ops.box_iou
        # 根据 IoU 阈值将候选区域标记为正样本、负样本或忽略。
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )
        # 用于从候选区域中选择正样本和负样本。确保每批数据中正负样本的比例一致。
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        # used during testing
        # 在进行 NMS 之前，选择前 n 个得分最高的候选区域。
        self._pre_nms_top_n = pre_nms_top_n
        # 在 NMS 过程之后，保留前 n 个得分最高的候选区域。
        self._post_nms_top_n = post_nms_top_n
        # 在 NMS 过程中，如果两个候选区域的 IoU 超过此阈值，则保留得分较高的一个。
        self.nms_thresh = nms_thresh
        # 在推理阶段，去除得分低于此阈值的候选区域。
        self.score_thresh = score_thresh
        self.min_size = 1e-3 # 过滤掉太小的边界框。
    # 可以根据对象的当前状态（如是否处于训练模式）来动态地决定返回哪个值。
    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]
    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]
    # 这个函数的主要作用是将真实框（ground truth boxes）的目标分配给锚框（anchors），用于生成训练样本。
    # anchors：一个列表，每个元素是一个张量（Tensor），表示一个图像中的所有锚框。
    # targets：一个列表，每个元素是一个字典，包含每个图像的真实框信息，其中每个字典包含至少一个键 "boxes"，
    # 指向该图像中的真实框。
    # 函数返回两个列表：
    # labels：每个锚框的标签列表，表示它是正样本（前景）、负样本（背景）还是忽略。
    # matched_gt_boxes：每个锚框对应的匹配真实框的列表
    def assign_targets_to_anchors(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        #初始化两个空列表 labels 和 matched_gt_boxes，分别用于存储每个图像的标签和匹配的真实框。
        labels = []
        matched_gt_boxes = []
        # 使用 zip 函数遍历 anchors 和 targets 列表中的每个元素（即每张图像的锚框和真实框信息）。
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # 从 targets_per_image 字典中提取当前图像的真实框 gt_boxes。
            gt_boxes = targets_per_image["boxes"]
            # 检查当前图像是否有真实框：
            # 如果 gt_boxes 的元素数量为 0（即图像中没有任何真实框），则认为该图像是一个背景图像（负样本）。
            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                # 初始化 matched_gt_boxes_per_image 为一个全零张量，形状与 anchors_per_image 相同。
                # 初始化 labels_per_image 为一个全零张量，形状为 (anchors_per_image.shape[0],)，表示所有锚框都标记为背景（负样本）。
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # 如果图像中存在真实框，计算每个真实框与所有锚框之间的相似度矩阵（通常是 IoU）。
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                # 使用 proposal_matcher 对象来匹配每个锚框与真实框，并返回匹配索引 matched_idxs。
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                # 根据匹配索引 matched_idxs 提取每个锚框对应的匹配真实框，并处理索引越界的问题。
                # 将 matched_idxs 中大于等于 0 的索引标记为正样本，并转换为 float32 类型。
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)
                # Background (negative examples)
                # 标记那些 IoU 低于低阈值的锚框为背景（负样本）。
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0
                # 标记那些 IoU 在两个阈值之间的锚框为忽略，通常用 -1.0 表示。
                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0
            # 将当前图像的标签和匹配的真实框追加到各自的列表中。
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        # 最后返回两个列表 labels 和 matched_gt_boxes。
        return labels, matched_gt_boxes
    # 其目的是在执行非极大值抑制（Non-Maximum Suppression, NMS）之前选取一定数量的候选区域（proposals）。
    # 这些候选区域是基于一个称为“objectness”的分数来选择的，这个分数反映了每个锚框（anchor）作为目标物体的可能性。
    # objectness: 一个张量，表示每个锚框的“objectness”得分。
    # num_anchors_per_level: 一个列表，包含了不同层级（level）的特征图中锚框的数量。
    # 函数返回一个张量，表示在每个层级上选择的前N个最高得分的锚框的索引。
    # 这段代码的主要作用是：分割：按照不同的层级分割 objectness 张量。选择：对于每个层级，选择具有最高 
    # “objectness” 得分的前N个锚框。索引调整：调整索引以反映在整个 objectness 张量中的位置。合并：将所
    # 有层级的选择结果合并成一个单一的张量。
    # 这个函数通常会用在目标检测模型中，特别是在生成候选区域之后，但执行 NMS 之前。这样做的目的是减少后续处理
    # 的候选区域数量，提高效率。
    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        # 首先初始化一个空列表 r 用来存储每个层级上的选定索引，以及一个偏移量 offset 用于调整索引。
        r = []
        offset = 0
        # 通过 split 方法按层级分割 objectness 张量，这样可以单独处理每个层级上的锚框。
        for ob in objectness.split(num_anchors_per_level, 1):
            # 获取当前层级的锚框数量 num_anchors。然后计算需要选择的锚框数量 pre_nms_top_n，这是通过一个名为 
            # det_utils._topk_min 的函数来确定的，它考虑了预先设定的最大候选数 self.pre_nms_top_n()。
            num_anchors = ob.shape[1]
            pre_nms_top_n = det_utils._topk_min(ob, self.pre_nms_top_n(), 1)
            # 使用 topk 方法来选择具有最高 “objectness” 得分的前N个锚框，并获得它们的索引 top_n_idx。
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            # 将得到的索引添加到结果列表 r 中，并且加上当前的偏移量 offset。这是因为 topk 返回的索引是在分割
            # 后的子张量中的相对索引，而我们想要的是在整个 objectness 张量中的绝对索引。
            # 更新偏移量 offset，以便于下一个层级的索引计算。
            offset += num_anchors
        # 最后，使用 torch.cat 方法沿着维度1（列方向）合并所有层级的结果，并返回这个合并后的张量。
        return torch.cat(r, dim=1)
    # 这个函数的作用是从一组候选区域（proposals）中筛选出最有可能包含目标物体的区域，并对其进行非极大值抑制
    # （NMS）处理，最终返回经过筛选的候选区域和它们的得分。
    # 函数返回两个列表：final_boxes: 经过筛选后的候选区域列表。final_scores: 对应的得分列表。
    def filter_proposals(
        self,
        proposals: Tensor, # 一个形状为 (num_images, num_anchors, 4) 的张量，表示每张图片的所有候选区域。
        objectness: Tensor, # 一个形状为 (num_images, num_anchors) 的张量，表示每个候选区域的“objectness”得分。
        image_shapes: List[Tuple[int, int]], # 一个列表，包含每张图片的尺寸（高度和宽度）。
        num_anchors_per_level: List[int],# 一个列表，表示不同层级的特征图中锚框的数量。
    ) -> Tuple[List[Tensor], List[Tensor]]:
        # 获取图片的数量和设备（CPU或GPU）
        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        # 分离 objectness 张量，使其不再参与反向传播。然后将 objectness 张量重塑为 
        # (num_images, num_anchors) 的形式。
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)
        # 为每个层级的锚框创建一个标识符张量 levels，表示每个锚框所属的层级。然后将这些标识符扩展为与 
        # objectness 相同的形状。
        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)
        # 调用 _get_top_n_idx 方法，选择每个层级上的前N个得分最高的候选区域，并获取它们的索引。
        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        # 创建一个范围张量 image_range，表示每张图片的索引，并将其扩展为 (num_images, 1) 的形状。
        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]
        # 根据索引 top_n_idx 从 objectness、levels 和 proposals 中选择得分最高的候选区域。
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        # 计算 objectness 的 Sigmoid 得分，作为候选区域的概率得分。
        objectness_prob = torch.sigmoid(objectness)
        # 初始化两个空列表，用于存储最终的候选区域和得分。
        final_boxes = []
        final_scores = []
        # 遍历每张图片的候选区域、得分、层级和图片尺寸。
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 将候选区域裁剪到图片的有效范围内。
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            # 移除太小的候选区域（尺寸小于 self.min_size）。
            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # 移除得分低于 self.score_thresh 的候选区域。
            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # 对每个层级上的候选区域执行非极大值抑制（NMS），保留得分最高的候选区域。
            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # keep only topk scoring predictions
            # 保留 NMS 后得分最高的前 self.post_nms_top_n() 个候选区域。
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            # 将处理后的候选区域和得分追加到最终列表中。
            final_boxes.append(boxes)
            final_scores.append(scores)
        # 返回最终的候选区域列表 final_boxes 和得分列表 final_scores。
        return final_boxes, final_scores
    # 这个函数的目的是计算区域提案网络（RPN）的损失，包括两部分：一是“objectness”损失，即候选区域是否包含目标的二分类损失
    # ；二是回归损失，即候选区域与真实框之间的位置回归损失。
    # objectness: 一个形状为 (num_images, num_anchors) 的张量，表示每个锚框的“objectness”得分。
    # pred_bbox_deltas: 一个形状为 (num_images, num_anchors, 4) 的张量，表示每个锚框的边界框回归预测。
    # labels: 一个列表，每个元素是一个张量，表示每个锚框的标签（正样本、负样本或忽略）。
    # regression_targets: 一个列表，每个元素是一个张量，表示每个锚框与匹配的真实框之间的偏移量。
    # 函数返回两个张量：objectness_loss: “objectness”得分的二分类损失。box_loss: 边界框回归的平滑 L1 损失。
    def compute_loss(
        self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        # 使用 fg_bg_sampler 从标签中选择正样本和负样本的索引。
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 将正样本和负样本的索引拼接成一个一维张量，并使用 torch.where 提取出非零索引，得到实际的索引值。
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        # 将正样本和负样本的索引合并成一个张量。
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        # 将 objectness 张量展平成一维张量，便于后续计算。
        objectness = objectness.flatten()
        # 将标签和回归目标拼接成一个一维张量。
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        # 计算正样本的边界框回归损失。这里使用了平滑 L1 损失（Smooth L1 Loss），这是一种混合了 L1 和 L2 
        # 损失的损失函数，可以在小误差时使用 L2 损失，在大误差时使用 L1 损失，从而兼顾了鲁棒性和准确性。
        # beta=1/9：平滑 L1 损失中的平滑参数，决定了何时从 L2 损失切换到 L1 损失。
        # reduction="sum"：指定损失的缩减方式为求和。/ (sampled_inds.numel())：对损失
        # 进行归一化，即除以选中的样本数量。
        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds.numel())
        # 计算“objectness”得分的二分类损失。这里使用了带 logits 的二元交叉熵损失（Binary Cross Entropy with Logits），
        # 适用于二分类问题，可以直接接受未经过 sigmoid 函数的原始输出。
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])
        # 返回计算得到的“objectness”损失和边界框回归损失。
        return objectness_loss, box_loss
    # 这个函数是区域提案网络（RPN）的前向传播方法，它负责生成候选区域（proposals）并计算损失（如果在训练模式下）。
    # 函数返回一个元组，包含两个元素：boxes: 一个列表，每个元素是一个张量，表示每个图像的候选区域。
    # losses: 一个字典，包含训练模式下的损失项。
    def forward(
        self,
        images: ImageList, # ImageList 类型，包含一批图像的信息。
        features: Dict[str, Tensor],# 一个字典，包含从图像中提取的不同层级的特征图。
        # 可选参数，如果在训练模式下，它是一个列表，每个元素是一个字典，包含每个图像的真实框信息（例如边界框
        # 坐标和类别标签）。
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        # 获取特征图,将输入的特征图字典转换为列表，以便可以逐层处理特征图。
        # RPN uses all feature maps that are available
        features = list(features.values())
        # 生成 objectness 和边界框偏移量,使用 RPN 的头部网络（self.head）生成每个锚框的“objectness”
        # 得分和边界框回归偏移量。
        objectness, pred_bbox_deltas = self.head(features)
        # 使用锚框生成器（self.anchor_generator）生成每个图像上的锚框。
        anchors = self.anchor_generator(images, features)
        # 获取图像数量和每层锚框数量,获取图像数量和每层特征图上的锚框数量。
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        # 将不同层级的 objectness 和 pred_bbox_deltas 拼接成一个张量。
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        # 解码边界框偏移量,使用边界框编码器（self.box_coder）将边界框偏移量应用到锚框上，得到解码后的候选区域
        # （proposals）。注意，这里使用 .detach() 将偏移量与计算图分离，因为在 Faster R-CNN 中不希望候选区
        # 域参与到反向传播中。
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        # 筛选候选区域:调用 filter_proposals 方法，根据 objectness 得分筛选出候选区域，并返回筛选后的候选区域和它们的得分。
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        # 计算损失（训练模式）:
        # 如果模型处于训练模式，计算 RPN 的损失：
        # 验证 targets 是否为空：如果 targets 是 None，抛出异常。
        # 分配目标到锚框：调用 assign_targets_to_anchors 方法，为每个锚框分配标签（正样本、负样本或忽略）和匹配的真实框。
        # 编码回归目标：使用边界框编码器将匹配的真实框编码为回归目标。
        # 计算损失：调用 compute_loss 方法计算“objectness”损失和边界框回归损失，并存储在 losses 字典中。
        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        # 返回筛选后的候选区域和损失字典（如果在训练模式下）。
        return boxes, losses
# 实现 Faster R-CNN 模型中的分类和边界框回归预测器。
class FastRCNNPredictor(nn.Module):
    # in_channels：输入特征图的通道数。num_classes：需要分类的目标类别数量（包括背景类别）。
    def __init__(self, in_channels, num_classes):
        # super().__init__()：调用父类的构造函数，完成基本的初始化。
        super().__init__()
        # self.cls_score：创建一个线性层，用于从输入特征中预测每个类别的得分。输入通道数为 in_channels，
        # 输出通道数为 num_classes。
        self.cls_score = nn.Linear(in_channels, num_classes)
        # self.bbox_pred：创建一个线性层，用于从输入特征中预测每个类别的边界框偏移量。输出通道数为
        # num_classes * 4，因为每个类别需要预测四个偏移量（即边界框的左上角和右下角坐标的变化量）。
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    # 用于接受输入特征，并产生分类得分和边界框偏移量的预测。
    # x：输入特征图，形状为 (batch_size, in_channels, height, width)。
    # 输出:scores：分类得分，形状为 (batch_size, num_classes)。
    # bbox_deltas：边界框偏移量，形状为 (batch_size, num_classes * 4)。
    def forward(self, x):
        # 输入维度检查：检查输入 x 的维度是否为 4（即形状为 (batch_size, in_channels, height, width)）。
        # 断言输入特征图的最后两个维度必须为 [1, 1]，这意味着输入特征已经被压缩到了单个像素点的大小。
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        # 将输入特征图展平为二维张量，形状变为 (batch_size, in_channels * height * width)。因为最后两维是 [1, 1]，
        # 所以展平后形状为 (batch_size, in_channels)。
        x = x.flatten(start_dim=1)
        # 分类得分预测：对展平后的特征进行分类得分预测，输出形状为 (batch_size, num_classes)。
        # 边界框偏移量预测：对展平后的特征进行边界框偏移量预测，输出形状为 (batch_size, num_classes * 4)。
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        # 返回分类得分 scores 和边界框偏移量 bbox_deltas。
        return scores, bbox_deltas

# 这个类继承自 GeneralizedRCNN，这意味着它复用了父类的一些通用功能，并在此基础上添加了
# 特定于 Faster R-CNN 的组件和参数。
class FasterRCNN(GeneralizedRCNN):
    def __init__(
        self,
        # 基础参数
        # 这是指模型中的骨干网络，通常是预先训练好的卷积神经网络（如 ResNet），用于从输入图像中提取特征。
        backbone,
        # 表示模型需要分类的目标类别数量，包括背景类。例如，对于 COCO 数据集来说，num_classes 应该设置
        # 为 81（80 个对象类别 + 1 个背景类别）。
        num_classes=None,
        # transform parameters
        # 这两个参数控制着输入到网络的图像尺寸。min_size 表示图像的最短边将会被缩放到的像素数；max_size 
        # 表示最长边的最大像素数，防止图像过大。
        min_size=800,
        max_size=1333,
        # 这些值用于标准化输入图像的像素值。image_mean 是每个颜色通道的平均值，image_std 是每个颜色
        # 通道的标准差。
        image_mean=None,
        image_std=None,
        # RPN parameters
        # 用于生成不同尺度和宽高比的锚框（anchors）。如果未提供，则使用默认的锚框生成器。
        rpn_anchor_generator=None,
        # 用于预测锚框是否包含目标对象的部分。如果没有提供，会根据骨干网络的输出通道数创建一个默认的 RPN 头。
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        # 在非极大值抑制（NMS）之前选择的前 N 个锚框数量，在训练和测试阶段可能不同。
        rpn_pre_nms_top_n_test=1000,
        # 在执行 NMS 后保留的前 N 个锚框数量。
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        # 执行 NMS 时的 IoU（交并比）阈值，IoU 大于此阈值的框会被抑制。
        rpn_nms_thresh=0.7,
        # 分别指定了前景和背景的 IoU 阈值，用于确定一个锚框是正样本还是负样本。
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        # 前者是指每个图像中采样的锚框数量，后者是指这些锚框中正样本所占的比例。
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        # 用于过滤掉那些得分低于此阈值的锚框。
        rpn_score_thresh=0.0,
        # Box parameters
        # 用于从特征图中裁剪出感兴趣的区域（ROIs），然后调整到固定的大小。如果未提供，则使用默认的 ROI 池化层。
        box_roi_pool=None,
        # 用于从 ROI 中提取更高级的特征。如果没有提供，则会创建一个默认的头部。
        box_head=None,
        # 用于从高层次特征中预测边界框的位置和类别。如果没有提供，则会创建一个默认的预测器。
        box_predictor=None,
        # 分别是在最终输出中过滤低得分框的阈值和执行 NMS 的 IoU 阈值。
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        # 每张图像最终输出的最大检测框数量。
        # box_detections_per_img 参数是用来限制每张图像最终输出的检测框数量，这个限制是在 NMS 之后应用的。
        box_detections_per_img=100,
        # 用于确定一个 ROI 是正样本还是负样本的 IoU 阈值。
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        # 分别是指每个图像中用于训练的 ROI 数量，以及其中正样本所占的比例。
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        # 边界框回归时使用的权重，通常用于平衡不同维度（宽度、高度、中心点坐标）的损失。
        bbox_reg_weights=None,
        **kwargs, # 允许传递其他未明确列出的关键字参数给父类或其他组件。
    ):
        # 这里检查了 backbone 是否具有 out_channels 属性，以确保可以获取到骨干网络的输出通道数。
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        # 这里检查了 rpn_anchor_generator 是否为 AnchorGenerator 类型或 None
        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )
        # 这里检查了 box_roi_pool 是否为 MultiScaleRoIAlign 类型或 None。
        if not isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None))): # 多尺度RoI对齐
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )
        # 这里确保了 num_classes 和 box_predictor 不能同时指定或同时为空。
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")
        # 这里初始化了 RPN 的各个组件，包括锚框生成器、RPN 头、以及 RPN 网络。
        out_channels = backbone.out_channels
        if rpn_anchor_generator is None:
            # 用于生成锚框
            rpn_anchor_generator = _default_anchorgen() 
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
        # 设置默认的均值和标准差
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)
        super().__init__(backbone, rpn, roi_heads, transform)

# 装饰器和参数处理
# 这个装饰器用于注册模型，使得模型可以在框架内部被方便地引用和管理
@register_model()
# 这个装饰器用于处理旧接口的参数映射，使旧版本的代码仍然可以工作。它将旧的参数名映射到新的参数名，并提供默认值。
@handle_legacy_interface(
    weights=("pretrained", FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
# 这个函数用于构建并返回一个使用 ResNet50 作为骨干网络的 Faster R-CNN 模型。它接收多个参数：
def fasterrcnn_resnet50_fpn(
    *,
    weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None, # 指定模型的预训练权重。
    progress: bool = True, # 在下载预训练权重时显示进度条。
    num_classes: Optional[int] = None, # 指定模型需要分类的类别数量。
    # 指定骨干网络的预训练权重。
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    # 指定骨干网络中可训练的层数。
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,# 允许传递其他未明确列出的关键字参数给模型。
) -> FasterRCNN:
    # 参数验证和处理
    # 这两行代码用于验证 weights 和 weights_backbone 的有效性，并将它们转换为正确的类型。
    weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)
    # 这段代码根据提供的 weights 参数来设置 num_classes 的值：
    # 如果提供了 weights，则 weights_backbone 应该为 None。
    if weights is not None:
        weights_backbone = None
        # 如果提供了 weights，则 num_classes 的值应该覆盖为权重文件中包含的类别数量。
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    # 如果没有提供 weights 且 num_classes 为 None，则默认设置为 91（COCO 数据集的类别数量）。
    elif num_classes is None:
        num_classes = 91
    # 这段代码判断模型是否是预训练的，并根据这个判断来设置 trainable_backbone_layers 的值。
    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    # 初始化骨干网络,根据 is_trained 的值选择不同的规范化层：
    # 如果模型是预训练的，则使用 FrozenBatchNorm2d。否则，使用普通的 BatchNorm2d。
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d
    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    # 然后使用指定的参数初始化 ResNet50 骨干网络。这段代码将 ResNet50 转换成带有特征金字塔网络（FPN）
    # 的骨干网络。
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    # 构建 Faster R-CNN 模型,使用初始化好的骨干网络和类别数量创建 FasterRCNN 模型，并传递其他关键字参数。
    model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)
    # 加载预训练权重,如果提供了 weights，则加载预训练权重，并在特定情况下（如使用 COCO 数据集的预训练
    # 权重）修改某些参数。
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)
    return model # 最后返回构建好的 FasterRCNN 模型。

class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")
        if self.has_mask():
            if not all(["masks" in t for t in targets]):
                raise ValueError("Every element of targets should have a masks key")

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)
        return result, losses

__all__ = [
    "FasterRCNN",
    "FasterRCNN_ResNet50_FPN_Weights",
    "FasterRCNN_ResNet50_FPN_V2_Weights",
    "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
    "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
]

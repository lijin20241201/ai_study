# 它接受一个 PyTorch 张量 masks 作为输入，并返回每个掩码对应的边界框的坐标。这个函数的目
# 的是从给定的一组掩码中提取出边界框的位置信息
def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    # 检查是否在脚本模式或跟踪模式下运行。如果不是这两种模式，则记录该 API 被使用了一次
    # （这通常是为了收集内部使用统计）。
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(masks_to_boxes)
    # 如果输入张量 masks 是空张量（没有元素），则返回一个形状为 (0, 4) 的零张量，表示没有边界框。
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)
    n = masks.shape[0]
    # 初始化一个形状为 (n, 4) 的张量 bounding_boxes，其中 n 是 masks 中掩码的数量。每个
    # 边界框由四个值组成：左上角和右下角的坐标（x_min, y_min, x_max, y_max）。
    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)
    for index, mask in enumerate(masks):
        # 对于 masks 中的每一个掩码，找出掩码中非零元素的位置（这些位置代表了对象的部分）。
        # 会返回两个张量，一个是行索引，另一个是列索引,先列出行号（垂直方向，y坐标），然后
        # 是列号（水平方向，x坐标）。这种顺序有助于保持一致性和避免混淆。
        y, x = torch.where(mask != 0)
        # 使用 torch.min 和 torch.max 计算每个掩码的最小和最大 x 和 y 坐标，从而确定边界框的位置。
        bounding_boxes[index, 0] = torch.min(x) # 左上x
        bounding_boxes[index, 1] = torch.min(y) # 左上y
        bounding_boxes[index, 2] = torch.max(x) # 右下x
        bounding_boxes[index, 3] = torch.max(y) # 右下y
    return bounding_boxes
# 定义一个名为 TVTensor 的类，它继承自 torch.Tensor。
class TVTensor(torch.Tensor):
    # 定义一个静态方法 _to_tensor，该方法接收任意类型的数据 data 并
    # 将其转换为 torch.Tensor。接受可选参数 dtype（数据类型）、device
    # （设备） 和 requires_grad（是否需要梯度）。
    # 静态方法是与类关联的方法，但它不接收任何关于类或实例的特殊参数。它不知道类的
    # 状态，也不需要访问类的实例变量
    # 不依赖于类的状态：静态方法并不需要访问类的状态（类变量或实例变量），因此它不需要类的实例即可使用。
    # 没有默认的第一个参数：静态方法不默认接收 self 或 cls 参数。
    @staticmethod
    def _to_tensor(
        data: Any,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> torch.Tensor:
        # 如果 requires_grad 参数为 None，则检查 data 是否已经是 torch.Tensor，如果是
        # ，则使用其 requires_grad 属性；否则，默认为 False。
        if requires_grad is None:
            requires_grad = data.requires_grad if isinstance(data, torch.Tensor) else False
        # 返回一个具有指定 dtype 和 device 的 torch.Tensor，并设置 requires_grad 属性。
        return torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(requires_grad)
    # 定义一个类方法 _wrap_output，用于包装输出结果，确保结果是 TVTensor 类型而不是普通的 torch.Tensor。它接
    # 收 output（输出），以及可选的 args 和 kwargs（参数和关键字参数）。
    # 类方法类似于静态方法，但它们接收一个名为 cls 的参数，该参数指向当前类。类方法可以修改类状态，
    # 即它可以直接操作类变量。
    # 接收 cls 参数：类方法默认接收一个 cls 参数，这个参数指向当前类。
    # 可以访问和修改类的状态：类方法可以访问和修改类变量，也可以创建类的新实例。
    @classmethod
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        # 注释表示该段逻辑与 torch._tensor._convert 相同。如果 output 是 torch.Tensor 
        # 类型但不是 TVTensor 类型，则将其转换为 TVTensor 类型。
        if isinstance(output, torch.Tensor) and not isinstance(output, cls):
            output = output.as_subclass(cls)
        # 如果 output 是元组或列表，则递归地处理其中的每个元素，确保它们都是 TVTensor 类型。
        if isinstance(output, (tuple, list)):
            # Also handles things like namedtuples
            output = type(output)(cls._wrap_output(part, args, kwargs) for part in output)
        return output
    # 定义一个类方法 __torch_function__，该方法是 PyTorch 的子类化协议的一部分，用于确保当进行数学运算时返回正确类型
    # 的对象。它接收一个函数 func，该函数的类型信息 types，以及可选的参数 args 和 kwargs。
    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., torch.Tensor],
        types: Tuple[Type[torch.Tensor], ...],
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        # 如果 TVTensor 不是所有输入类型 types 的子类，则返回 NotImplemented 表示无法处理。
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented
        # 注释解释了为何使用 DisableTorchFunctionSubclass 上下文管理器。在此上下文中执行
        # 函数 func 并获取输出 output。
        with DisableTorchFunctionSubclass():
            output = func(*args, **kwargs or dict())
        # 检查是否必须返回子类实例，或者如果函数在强制返回子类列表中且第一个参数是 TVTensor 类型。
        must_return_subclass = _must_return_subclass()
        if must_return_subclass or (func in _FORCE_TORCHFUNCTION_SUBCLASS and isinstance(args[0], cls)):
            # If you're wondering why we need the `isinstance(args[0], cls)` check, remove it and see what fails
            # in test_to_tv_tensor_reference().
            # The __torch_function__ protocol will invoke the __torch_function__ method on *all* types involved in
            # the computation by walking the MRO upwards. For example,
            # `out = a_pure_tensor.to(an_image)` will invoke `Image.__torch_function__` with
            # `args = (a_pure_tensor, an_image)` first. Without this guard, `out` would
            # be wrapped into an `Image`.
            # 注释解释了为何需要 isinstance(args[0], cls) 这个检查。根据条件，返回包装后的输出。
            return cls._wrap_output(output, args, kwargs)
        # 如果不需返回子类实例且输出已经是 TVTensor 类型，则手动解除包装，返回普通 torch.Tensor 类型。
        if not must_return_subclass and isinstance(output, cls):
            # DisableTorchFunctionSubclass is ignored by inplace ops like `.add_(...)`,
            # so for those, the output is still a TVTensor. Thus, we need to manually unwrap.
            return output.as_subclass(torch.Tensor)
        return output
    # 定义一个方法 _make_repr，该方法用于生成包含额外信息的字符串表示形式。
    def _make_repr(self, **kwargs: Any) -> str:
        # This is a poor man's implementation of the proposal in https://github.com/pytorch/pytorch/issues/76532.
        # If that ever gets implemented, remove this in favor of the solution on the `torch.Tensor` class.
        extra_repr = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        return f"{super().__repr__()[:-1]}, {extra_repr})"
    # 注释提到这是一个简单实现，参考了一个提议的问题。根据传入的关键字参数生成额外的字符串表示，并添加
    # 到基础的字符串表示之后。
    # Add properties for common attributes like shape, dtype, device, ndim etc
    # this way we return the result without passing into __torch_function__
    @property
    def shape(self) -> _size:  # type: ignore[override]
        # 定义一个属性 shape，以直接返回形状信息而不触发 __torch_function__ 协议。
        with DisableTorchFunctionSubclass():
            return super().shape
    # 使用 DisableTorchFunctionSubclass 上下文管理器来避免触发 __torch_function__ 并返回父类的形状。
    # 定义一个属性 ndim，以返回维度数。
    @property
    def ndim(self) -> int:  # type: ignore[override]
        # 使用 DisableTorchFunctionSubclass 上下文管理器来避免触发 __torch_function__ 并返回父类的维度数。
        with DisableTorchFunctionSubclass():
            return super().ndim
    # 定义一个属性 device，以返回所在设备。
    @property
    def device(self, *args: Any, **kwargs: Any) -> _device:  # type: ignore[override]
        with DisableTorchFunctionSubclass():
            return super().device
    # 使用 DisableTorchFunctionSubclass 上下文管理器来避免触发 __torch_function__ 并返回父类的设备信息。
    @property
    def dtype(self) -> _dtype:  # type: ignore[override]
        # 定义一个属性 dtype，以返回数据类型。
        with DisableTorchFunctionSubclass():
            return super().dtype
    # 使用 DisableTorchFunctionSubclass 上下文管理器来避免触发 __torch_function__ 并返回父类的数据类型。
    # 定义一个方法 __deepcopy__，用于创建一个深度拷贝的对象。
    def __deepcopy__(self: D, memo: Dict[int, Any]) -> D:
        # We need to detach first, since a plain `Tensor.clone` will be part of the computation graph, which does
        # *not* happen for `deepcopy(Tensor)`. A side-effect from detaching is that the `Tensor.requires_grad`
        # attribute is cleared, so we need to refill it before we return.
        # Note: We don't explicitly handle deep-copying of the metadata here. The only metadata we currently have is
        # `BoundingBoxes.format` and `BoundingBoxes.canvas_size`, which are immutable and thus implicitly deep-copied by
        # `BoundingBoxes.clone()`.
        # 注释解释了为何需要先执行 .detach() 方法来断开计算图连接，然后克隆对象，并在返回前重新设置 requires_grad 属性。此外，还提到
        # 这里没有显式处理元数据的深拷贝，因为现有的元数据是不可变的。
        return self.detach().clone().requires_grad_(self.requires_grad)  # type: ignore[return-value]
# 这段代码定义了一个名为 Image 的类，它是 TVTensor 的子类，专门用来处理图像数据。
class Image(TVTensor):
    # 定义一个特殊方法 __new__，该方法控制对象的创建过程。这个方法接收任意类型的数据 data 以及可选的 dtype（数据类型）、
    # device（设备） 和 requires_grad（是否需要梯度）。这个方法的返回值是 Image 类型的对象。
    # __new__ 是一个静态方法，它负责创建一个新的实例。__new__ 必须总是返回一个类的新实例。通常情况下，这个方法由Python
    # 的解释器自动调用，当你使用 MyClass() 创建一个新实例时。
    # __new__ 方法主要在以下场景中使用：
    # 控制对象的创建过程；在创建实例之前做某些事情（例如，改变要返回的类）；返回一个已存在的实例，
    # 而不是每次都创建新的；如果你想改变实例创建的方式，比如改变内存分配方式或者想要返回一个替代类的实例。
    # __init__ 是一个实例方法，它在对象创建后立即被调用，用于初始化新创建的对象。__init__ 
    # 不需要返回任何东西，它的主要任务是设置对象的状态。
    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> Image:
        # 如果 data 是 PIL.Image.Image 类型，即Python Imaging Library（PIL）中的图像对象
        # ，那么导入 torchvision.transforms.v2.functional模块，并使用 pil_to_tensor 方
        # 法将图像转换成 torch.Tensor 对象。
        if isinstance(data, PIL.Image.Image):
            from torchvision.transforms.v2 import functional as F
            data = F.pil_to_tensor(data)
        # 使用 TVTensor 类中的 _to_tensor 方法将 data 转换为 TVTensor 对象，同时设置 dtype
        # 、device 和 requires_grad 属性。
        # 如果 cls 是一个继承自另一个类的子类，并且 _to_tensor 方法在这个子类中没有定义，那么这段
        # 代码实际上是在尝试调用父类中的 _to_tensor 方法。
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        # 如果转换后的张量维度小于 2（即不是一个至少二维的数组，通常图像至少有宽度和高度两个维度），
        # 则抛出 ValueError 异常。
        if tensor.ndim < 2:
            raise ValueError
        # 如果张量维度正好为 2（可能是一个灰度图像，只有宽度和高度），则增加一个新的维度（通常是通道维度）
        # ，使张量成为三维，例如 (1, height, width)。
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        # 最后返回一个 Image 类型的对象，这确保了返回的对象是一个 Image 实例而不是普通的 TVTensor。
        # as_subclass 方法通常在一个类需要确保其所有实例都具有某些特定行为时使用。这意味着即使 tensor 是通过一个
        # 通用的构造函数创建的基础张量，通过 as_subclass 调用，可以将其转换为 cls 类的一个实例，这样这个实例就
        # 会获得 cls 类的所有属性和方法。
        # 假设 cls 是一个名为 Image 的子类，并且这个子类继承自一个基础的张量类 TVTensor，那么即使 tensor 是一个普
        # 通的 TVTensor 对象，tensor.as_subclass(Image) 将确保返回的对象是一个 Image 实例
        return tensor.as_subclass(cls)
    # 定义一个方法 __repr__，该方法返回一个字符串，用来表示 Image 对象的信息。在这个方法中，直接调
    # 用 Image 类中的 _make_repr方法来生成字符串表示。_make_repr 方法负责构造一个包含额外信息的
    # 字符串表示形式。
    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr()
class BoundingBoxFormat(Enum):
    # 定义了一个枚举类 BoundingBoxFormat，用于表示不同的边界框格式：
    # XYXY：表示边界框的左上角坐标 (x1, y1) 和右下角坐标 (x2, y2)。
    # XYWH：表示边界框的左上角坐标 (x, y) 和宽度 w、高度 h。
    # CXCYWH：表示边界框的中心坐标 (cx, cy) 和宽度 w、高度 h。
    XYXY = "XYXY"
    XYWH = "XYWH"
    CXCYWH = "CXCYWH"
# 这段代码定义了一个 BoundingBoxes 类，该类扩展了 TVTensor 类，用于表示边界框数据。
# 它包含了处理边界框格式和画布大小的方法，并确保在创建和操作边界框时保持正确的元数据。
class BoundingBoxes(TVTensor):
    format: BoundingBoxFormat
    canvas_size: Tuple[int, int]
    # 这个类方法用于包装一个 torch.Tensor 对象，确保它是一个 BoundingBoxes 类型
    # 的对象，并且设置了边界框的格式和画布大小。
    @classmethod
    def _wrap(cls, tensor: torch.Tensor, *, format: Union[BoundingBoxFormat, str], canvas_size: Tuple[int, int], check_dims: bool = True) -> BoundingBoxes:  # type: ignore[override]
        # 如果设置了 check_dims，则检查张量的维度。如果张量是一维的，将其扩展为二维；如果不是一维
        # 也不是二维，则抛出 ValueError
        if check_dims:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 2:
                raise ValueError(f"Expected a 1D or 2D tensor, got {tensor.ndim}D")
        # 如果 format 是字符串，则将其转换为相应的 BoundingBoxFormat 枚举值。
        if isinstance(format, str):
            format = BoundingBoxFormat[format.upper()]
        # 使用 tensor.as_subclass(cls) 将张量转换为 BoundingBoxes 类型，并设置边界框
        # 的格式和画布大小。
        bounding_boxes = tensor.as_subclass(cls)
        bounding_boxes.format = format
        bounding_boxes.canvas_size = canvas_size
        return bounding_boxes
    # 这个方法用于创建一个 BoundingBoxes 对象。它首先将输入数据转换为 torch.Tensor，
    # 然后使用 _wrap 方法来包装这个张量。
    def __new__(
        cls,
        data: Any,
        *,
        format: Union[BoundingBoxFormat, str],
        canvas_size: Tuple[int, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> BoundingBoxes:
        # 使用 cls._to_tensor 方法将输入数据转换为 torch.Tensor。
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        # 使用 _wrap 方法包装转换后的张量，并设置边界框的格式和画布大小。
        return cls._wrap(tensor, format=format, canvas_size=canvas_size)
    # 这个类方法用于处理边界框输出的结果，确保输出是一个 BoundingBoxes 类型的对象，
    # 并恢复丢失的元数据。
    @classmethod
    def _wrap_output(
        cls,
        output: torch.Tensor,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> BoundingBoxes:
        # If there are BoundingBoxes instances in the output, their metadata got lost when we called
        # super().__torch_function__. We need to restore the metadata somehow, so we choose to take
        # the metadata from the first bbox in the parameters.
        # This should be what we want in most cases. When it's not, it's probably a mis-use anyway, e.g.
        # something like some_xyxy_bbox + some_xywh_bbox; we don't guard against those cases.
        # 使用 tree_flatten 将输入参数展平，然后找到第一个 BoundingBoxes 类型的对象，并从中提取格式和画布大小
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))  # type: ignore[operator]
        first_bbox_from_args = next(x for x in flat_params if isinstance(x, BoundingBoxes))
        format, canvas_size = first_bbox_from_args.format, first_bbox_from_args.canvas_size
        # 如果输出是一个 torch.Tensor 但不是一个 BoundingBoxes 对象，则使用 _wrap 方法包装它。如果输出
        # 是一个元组或列表，则递归地包装其中的每个元素。
        if isinstance(output, torch.Tensor) and not isinstance(output, BoundingBoxes):
            output = BoundingBoxes._wrap(output, format=format, canvas_size=canvas_size, check_dims=False)
        elif isinstance(output, (tuple, list)):
            output = type(output)(
                BoundingBoxes._wrap(part, format=format, canvas_size=canvas_size, check_dims=False) for part in output
            )
        return output
    # 这个方法用于生成一个包含边界框元数据的字符串表示形式。它调用了 _make_repr 方法，并传递了格式和画布大小。
    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(format=self.format, canvas_size=self.canvas_size)

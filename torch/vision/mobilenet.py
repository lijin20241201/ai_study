import torch
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union
from torch import nn, Tensor
from torchvision.utils import _log_api_usage_once, _make_ntuple
from torchvision.models._api import register_model, Weights, WeightsEnum
# 卷积块
class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self, 
        in_channels: int, # 输入通道
        out_channels: int, # 输出通道
        kernel_size: Union[int, Tuple[int, ...]] = 3, # 核大小
        stride: Union[int, Tuple[int, ...]] = 1, # 步长
        padding: Optional[Union[int, Tuple[int, ...], str]] = None, # 填充
        groups: int = 1, # 分组卷积
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d, # 批次标准化
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU, # 激活层
        dilation: Union[int, Tuple[int, ...]] = 1, # 膨胀率
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        # conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d 这种写法结合了类型注解
        # （Type Hints）和默认值的概念
        # 将torch.nn.Conv2d这个类本身赋值给了conv_layer，同时给出了一个类型注解说明conv_layer应该是一个
        # 可调用的（Callable）对象，其返回类型应该是torch.nn.Module或其子类的一个实例。
        # 省略号的使用更接近于表示“可调用对象接受任意数量的参数，并且这些参数的具体类型在这里不详细指定，但重要
        # 的是它的返回类型应该是torch.nn.Module或其子类的一个实例”
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:
        # 如果填充没指定
        if padding is None:
            # 如果核大小是整数类型并且膨胀率为整数类型
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        # 如果bias未设定,看norm_layer是不是None
        if bias is None:
            bias = norm_layer is None
        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        # 如果设定了标准化层,加进列表
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        # 如果有激活层,就添加激活层
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers) # 调用父类的初始化方法
        _log_api_usage_once(self) # 跟踪API的使用频率或收集统计信息
        self.out_channels = out_channels
        # 这个类一般作为基类
        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )
# 继承自ConvNormActivation
class Conv2dNormActivation(ConvNormActivation):
    def __init__(
        self, # 当前实例对象
        in_channels: int, # 输入通道
        out_channels: int, # 输出通道
        kernel_size: Union[int, Tuple[int, int]] = 3, 
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True, # 是否就地操作
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__( # 传入参数调用父类
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )
# 倒残差块(继承自nn.Module)
class InvertedResidual(nn.Module):
    # stride: 卷积的步长，通常为 1 或 2，用于控制特征图的空间尺寸。
    # expand_ratio: 扩展比例，用于控制第一个 1x1 卷积后的隐藏层通道数。如果为 1，则不进行扩展。
    # 初始化方法参数:inp:输入通道数,oup:输出通道数,stride:步长, expand_ratio:扩展比例
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, 
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__() # 调用父类初始化方法
        self.stride = stride # 保存为实例属性
        # 如果步长大小不是1或2,抛出错误
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")
        # norm_layer: 归一化层的类型，默认为 nn.BatchNorm2d。
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 计算隐藏层维度 hidden_dim，这是输入通道数乘以扩展比例后取整的结果。
        hidden_dim = int(round(inp * expand_ratio))
        # 使用残差连接的条件是步长为1且输入输出通道数相等时
        self.use_res_connect = self.stride == 1 and inp == oup
        # 构建模块序列,Python 中的泛型列表类型,列表中的元素类型应该为 nn.Module 或其子类
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # 如果 expand_ratio 不为 1，则首先添加一个 1x1 卷积层（点卷积），用于扩展特征图的通道数
            layers.append(
                Conv2dNormActivation(
                    inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # 深度卷积,指定groups=hidden_dim,每个输入通道都会使用它自己的卷积核进行卷积
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # 普通点卷积
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),  # 标准化层
            ]
        )
        # * 用于拆包可迭代对象作为位置参数,** 用于拆包字典作为关键字参数。
        # 将 layers 列表中的所有元素作为独立的参数传递给 nn.Sequential 的构造函数，从而构造出一个顺序模型。
        self.conv = nn.Sequential(*layers) 
        self.out_channels = oup # 保持为实例属性
        self._is_cn = stride > 1
    # x:(b,c,h,w)
    def forward(self, x: Tensor) -> Tensor:
        # 如果满足残差条件,返回残差连接
        if self.use_res_connect:
            return x + self.conv(x) # 序列化栈前后残差
        else:
            return self.conv(x)
# _make_divisible 的目的是将一个浮点数 v 转换为一个整数，同时确保这个整数是某个除数 divisor 的倍数，且满足一
# 些额外的条件。v: 需要被转换的浮点数。divisor: 转换后的整数需要是这个数的倍数。min_value: 转换后的整数的最小值，
# 默认为 divisor。如果设置为 None，则自动使用 divisor 作为最小值。
# 这个函数的一个典型应用场景是在调整神经网络层的通道数时，确保这些通道数是某个特定值（如8或16）的倍数，这有助于优化内
# 存访问和计算效率。同时，通过限制 new_v 相对于 v 的下降幅度，可以避免层的容量（即参数数量）急剧减少
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    # 首先，如果 min_value 没有被指定，那么它将被设置为 divisor。
    if min_value is None:
        min_value = divisor
    # print(min_value,divisor,v) 8 8 32.0
    # 接下来，计算 new_v，这是通过先将 v 加上 divisor / 2（这是为了四舍五入到最近的 divisor 的倍数）
    # ，然后除以 divisor 并向下取整（// 操作符），最后再乘以 divisor 来实现的。这样做的目的是尽量接近
    # 原始值 v，同时确保结果是 divisor 的倍数。但是，如果直接这样计算，可能会得到比 v 小很多的结果（特
    # 别是当 v 接近但略小于某个 divisor 的倍数时）
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    # 为了防止 new_v 相对于 v 下降太多（超过10%），函数检查 new_v 是否小于 v 的90%。如果是，那么
    # 将 new_v 增加 divisor，以确保它不会下降太多。
    if new_v < 0.9 * v:
        new_v += divisor
    # 最后，函数返回计算得到的 new_v。   
    return new_v
# MobileNetV2
class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000, # 多分类
        width_mult: float = 1.0, # 宽度因子
        # 残差设置,可选参数,List[List[int],默认None
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        # 可选参数,它是可调用对象,...表示它可以接受任意数量的参数,但是它的返回值是nn.Model类型(或者其子类),默认是None
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__() # 调用父类的初始化方法
        _log_api_usage_once(self)
        # 如果没有设定block
        if block is None:
            block = InvertedResidual # 设定block为倒置残差
        # 如果没指定标准化层
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # 设定为批次标准化
        # 设定输入通道
        input_channel = 32
        last_channel = 1280 # 设定最后一层的输出通道数
        # 如果没指定残差设置
        if inverted_residual_setting is None:
            # 每个元组[t, c, n, s]代表了一个特定层或一系列层的配置
            # t（expansion factor，扩展因子）：这个参数决定了在点卷积（Pointwise Convolution，即1x1卷积）之前，
            # 输入通道数将被扩展多少倍。扩展后的通道数等于输入通道数乘以t。如果t=1，则不进行扩展。
            # c（output channels，输出通道数）：这是经过Inverted Residual Block后输出的通道数。
            # n（number of blocks，块的数量）：这指定了具有相同t、c和s配置的Inverted Residual Block应该重复
            # 多少次。
            # s（stride，步长）：这个参数决定了块中第一个深度卷积（Depthwise Convolution）的步长。它用于控制特征
            # 图的空间尺寸。s=1表示特征图的空间尺寸保持不变，而s=2表示特征图的高度和宽度都将减半（即进行下采样）。
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2], # (24,112,112)
                [6, 32, 3, 2], # (32,56,56)
                [6, 64, 4, 2], # (64,28,28)
                [6, 96, 3, 1],
                [6, 160, 3, 2], # (160,14,14)
                [6, 320, 1, 1],
            ]
        # 如果用户指定了设置,但是没指定正确,抛出错误
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )
        # 确保第一个输出通道是8的倍数,确保最后的输出通道是8的倍数
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        # 构建序列化堆叠栈,里面的元素是nn.Module类型,第一个元素是自定义conv2d,步长2,下采样
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        # 构建特征提取块
        # 遍历设置中的每个元素(是个列表),t:因子,c:输出通道数,n:相同配置的残差块的重复次数,s:步长
        for t, c, n, s in inverted_residual_setting:
            # 首先把output_channel设定成8的倍数,相同的配置下输出通道数是一样的
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n): # 遍历相同配置的深度
                # 第一次的话也许要下采样,之后步长设定为1
                stride = s if i == 0 else 1
                # 把残差块添加进特征提取列表
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                # 这时候设定下个残差块的输入通道为output_channel
                input_channel = output_channel
        # 添加最后的自定义卷积块(点卷积)
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, 
                activation_layer=nn.ReLU6
            )
        )
        # encoder(自动拆包),特征提取模块
        self.features = nn.Sequential(*features)
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )
        # 权重初始化
        # .modules()方法是一个非常有用的递归迭代器，它遍历模块本身及其所有子模块，以生成一个模块迭代
        # 器。这允许你轻松地对模型中的所有层或模块应用某种操作，比如权重初始化。
        for m in self.modules():
            #对于nn.Conv2d，使用kaiming_normal_来初始化权重，这是一种根据输入和输出的数量自
            # 适应调整方差的初始化方法，常用于ReLU及其变体作为激活函数的网络。如果层有偏置项，则将
            # 其初始化为0。
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # 对于nn.BatchNorm2d和nn.GroupNorm（批量归一化和组归一化层），权重被初始化为1（
            # 这是常见的做法，因为归一化层会除以权重的平方根，初始化为1使得初始化时不会改变输入
            # 的规模），偏置项被初始化为0。
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # 对于nn.Linear（全连接层），权重使用均值为0、标准差为0.01的正态分布进行初始化，偏置项被初
            # 始化为0。
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    # 受保护方法:x(b,c,h,w)
    def _forward_impl(self, x: Tensor) -> Tensor:
        # 这种做法通常是为了与 PyTorch 的 TorchScript 兼容性有关，因为 TorchScript 在处理继承时有一些限制，
        # 特别是在子类中覆盖父类的 forward 方法时。通过将实际的前向传播逻辑放在 _forward_impl 这样的方法中，
        # 并在 forward 方法中调用它，可以在不牺牲 TorchScript 兼容性的情况下，更容易地在子类中重写或扩展前向传
        # 播逻辑。
        x = self.features(x) # 提取特征,encoder
        # Cannot use "squeeze" as batch-size can be 1
        # nn.functional.adaptive_avg_pool2d 是一个非常有用的函数，它可以根据输入张量的尺寸动态地调整池化
        # 窗口的大小，以确保输出张量具有指定的尺寸（在你的例子中是 (1, 1)）。这对于从特征图中提取全局平均池化
        # 特征非常有用，特别是在分类任务中。
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)) # 平均池化
        x = torch.flatten(x, 1) # 扁平化,从第二个轴（索引为1）开始扁平化
        x = self.classifier(x) # 输出层,多少分类
        return x
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
# _IMAGENET_CATEGORIES 会是一个包含 ImageNet 数据集所有类别的列表或类似结构。
_COMMON_META = {
    "num_params": 3504872, # num_params 表示模型的参数数量
    "min_size": (1, 1), # min_size 表示模型可以接受的最小输入尺寸，这里设置为 (1, 1)
    "categories": _IMAGENET_CATEGORIES, 
}
class InvertedResidualConfig: # 倒残差配置
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        # 设定输入通道数(确保是8的倍数)
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel # 核大小
        # 扩张通道数
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult) # 输出通道数
        self.use_se = use_se # 是否用se模块
        self.use_hs = activation == "HS" # 是否用nn.Hardswish
        self.stride = stride # 步长
        self.dilation = dilation # 膨胀率
    # 静态方法可以通过类来调用，也可以通过类的实例来调用，但它们不访问或修改类的属性或状态。
    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)
# “Squeeze”操作指的是通过全局平均池化等方式来压缩特征图的空间维度，从而得到一个通道描述
# 符，这个描述符可以看作是每个通道的全局信息。“Excitation”操作则是一个简单的自门控机制，
# 通过参数学习来为每个通道分配不同的权重，这个权重可以看作是特征通道的重要性。最终，这些权
# 重被用于重新标定原始特征图，以增强有用的特征并抑制不重要的特征。
class SqueezeExcitation(torch.nn.Module): # 压缩激励
    def __init__(
        self,
        input_channels: int, # 输入通道数
        squeeze_channels: int, #压缩点卷积通道数
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__() # 调用父类的初始化方法
        _log_api_usage_once(self)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1) # 平均池化
        # 压缩通道的点卷积
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()
    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input) # 平均池化
        scale = self.fc1(scale) # 压缩点卷积切换通道
        scale = self.activation(scale) # 过滤掉负数
        scale = self.fc2(scale) # 扩张点卷积把通道还原到c
        # 用sigmoid激活函数来让模型学习给每个通道评分,来表示通道的重要性
        return self.scale_activation(scale) 
    # input:(b,c,h,w)
    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input) # 压缩激励
        # 对特征图加权
        return scale * input 
class InvertedResidual2(nn.Module): # 倒残差块
    def __init__(
        self,
        cnf: InvertedResidualConfig, # 残差配置
        norm_layer: Callable[..., nn.Module], # 标准化层
        se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation,scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        # 步长必须为1或者2,否则报错
        if not (1 <= cnf.stride <= 2): 
            raise ValueError("illegal stride value")
        # 使用残差的条件是步长是1,并且输入输出通道一致
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        # 序列化栈初始化
        layers: List[nn.Module] = []
        # 激活函数看cnf.use_hs,为True用Hardswish,否则用ReLU
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        # 如果扩张通道!= 输入通道,用自定义点卷积切换通道
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
        # 如果膨胀率大于1,设定步长为1,否则设定步长为cnf.stride
        stride = 1 if cnf.dilation > 1 else cnf.stride
        # 深度卷积
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels, 
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se: # 如果设定要用se模块,用到深度卷积之后
            # 压缩点卷积通道数
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))
        # 添加自定义点卷积,没有激活函数,是因为卷积是线性,如果引入非线性的激活函数
        # 可能会影响残差连接,如果步长为1,会残差连接
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, 
                norm_layer=norm_layer, activation_layer=None
            )
        )
        # 单个倒置残差块
        self.block = nn.Sequential(*layers) 
        self.out_channels = cnf.out_channels # 设定输出通道数
        self._is_cn = cnf.stride > 1
    def forward(self, input: Tensor) -> Tensor: 
        # 扩张点卷积-->深度卷积-->se-->收缩点卷积
        result = self.block(input) 
        if self.use_res_connect: # 如果要使用残差连接
            result += input
        return result
class MobileNetV3(nn.Module):
    def __init__(
        self,
        # 残差设置列表
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int, # 最后的输出通道
        num_classes: int = 1000, # 分类数
        # 倒置残差块
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None, # 标准化层
        dropout: float = 0.2, # dropout
        **kwargs: Any, 
    ) -> None:
        super().__init__() # 调用父类的初始化方法
        _log_api_usage_once(self)
        # 如果没有残差设置的话,抛出错误
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        # 如果有配置,但是配置不是Sequence的实例,或者里面的元素不是InvertedResidualConfig的实例的话,抛出类型错误
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")
        # 如果block是None,指定残差块类
        if block is None: 
            block = InvertedResidual2
        if norm_layer is None: # 设置norm_layer
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        # encoder:提取特征模块
        layers: List[nn.Module] = []
        # 设定第一个卷积层的输出通道,它是残差设置列表里第一个残差配置实例的输入通道数
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append( # 第一个自定义普通卷积下采样块 (112,112)
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )
        # 遍历inverted_residual_setting中的每一行设置,每一行设置都是一个残差配置类实例
        for cnf in inverted_residual_setting: # 加很多次
            layers.append(block(cnf, norm_layer)) # 在列表里加倒置残差块
        # 设定最后的卷积块的输入和输出通道
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append( # 最后的自定义点卷积块
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )
        self.features = nn.Sequential(*layers) # encoder
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.classifier = nn.Sequential( # 分类器
            # 线性层转换
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True), # 激活函数
            nn.Dropout(p=dropout, inplace=True), 
            nn.Linear(last_channel, num_classes), # 分类层
        )
        for m in self.modules(): # 初始化模型的所有模块的权重
            # 如果是Conv2d的实例
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # nn.init.normal_() 函数从一个均值为 0、标准差为 0.01 的正态分布中抽取数值，并将这些数值填充到 m.weight 张
            # 量中。这个操作直接改变了 m.weight 的值，而不是创建一个新的张量来保存结果
            # 就地操作通常比非就地操作（返回新张量的操作）更高效，因为它不需要分配新的内存空间来存储结果。
            # 就地操作直接修改传入的张量，而不是返回一个新的张量
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x) # 特征提取
        x = self.avgpool(x) # 平均池化
        x = torch.flatten(x, 1) # 扁平化(b,d)
        x = self.classifier(x) # 分类
        return x
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
def _mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    # reduce_divider 变量的作用是在某些情况下减少网络尾部的通道数。当 reduced_tail 参数为 True 时，reduce_divider 
    # 被设置为2，这意味着在构建 MobileNet V3 Large 架构的尾部时，每一层的输出通道数都会被除以2。这是为了减少模型的参数
    # 数量和计算复杂度，从而可能提高效率或者减少过拟合的风险。
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1 # 如果设定使用膨胀,dilation = 2
    # functools.partial 函数被设计用来“部分应用”一个函数，这意味着你可以预先填充（或“冻结”）函数
    # 的一些参数，并返回一个新的函数，这个新函数在被调用时会使用这些预填充的参数。虽然functools.
    # partial通常与普通的函数一起使用，但它同样可以应用于类构造函数（即类的__init__方法），只要这
    # 个类是通过其构造函数（或任何可以被调用的方法）以函数式方式被设计的。
    # 残差配置对象
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    # 把通道变成8的倍数
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)
    # arch:一个标记字符串,用来走不同的条件分支
    if arch == "mobilenet_v3_large":
        # (in_c,k_s,e_c,o_c,u_se,act,s,d)
        # 输入通道,核大小,扩张通道,输出通道,是否用se,激活函数,步长,膨胀率
        # 前面的特征提前没有用se模块
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),# 其中的一个残差配置类实例
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1,(56,56)
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2 (28,28)
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3 (14,14)
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            # 在标准的卷积操作中，卷积核内的元素是紧密相连的，而在膨胀卷积中，这些元素之间会有空洞。例如，如果膨
            # 胀率为2，且卷积核大小为3x3，则卷积核实际上只会与输入特征图上的非连续像素进行卷积操作，中间会跳过一些像素点。
            # 膨胀卷积被用于最后一个特征图提取阶段（标记为 "C4"）。在这个阶段，使用膨胀率为2的膨胀卷积意味着每个卷积核将
            # 每隔一个像素点进行采样，从而在不改变输出特征图尺寸的情况下增加了感受野。这样做可以在保持相同的空间分辨率的同
            # 时，使模型能够更好地捕捉到更广泛的信息。
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4 (7,7)
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        # 设定最后的输出通道数
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1  (56,56)
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2 (28,28)
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3 (14,14)
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4 (7,7)
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")
    return inverted_residual_setting, last_channel
def _mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig], # 残差配置列表
    last_channel: int, # 最后的输出通道数
    weights: Optional[WeightsEnum], # 可选,权重枚举
    progress: bool,
    **kwargs: Any,
) -> MobileNetV3:
    # 如果weights不是None
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model


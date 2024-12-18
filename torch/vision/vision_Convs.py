# __all__ 是一个可选的列表，定义在模块级别。当使用 from ... import * 语句时，如果模块中定义了
# __all__，则只有 __all__ 列表中的名称会被导入。这是模块作者控制哪些公开API被导入的一种方式。
# 使用 * 导入的行为
# 如果模块中有 __all__ 列表：只有 __all__ 列表中的名称会被导入
# 如果模块中没有 __all__ 列表：Python 解释器会尝试导入模块中定义的所有公共名称（即不
# 是以下划线 _ 开头的名称）。但是，这通常不包括以单下划线或双下划线开头的特殊方法或变量
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x): # (b,c1,...)
        # (b,c1,...)-->(b,c2,...),卷积块
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))
class Conv2(Conv):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv
    def forward(self, x): # (b,c1,...)
        # self.cv2(x):(b,c1,...)-->(b,c2,...) # 点卷积
        # self.conv(x):(b,c1,...)-->(b,c2,...) # 普通卷积
        # 先把两种卷积的处理结果做残差,之后批次标准化,激活函数
        return self.act(self.bn(self.conv(x) + self.cv2(x)))
    def forward_fuse(self, x): # (b,c1,...)
        return self.act(self.bn(self.conv(x)))
    def fuse_convs(self):
        # 具有conv权重w形状的全0张量
        w = torch.zeros_like(self.conv.weight.data) 
        i = [x // 2 for x in w.shape[2:]]  # (1,1)
        # 将w中最后两维1:2的数据用cv2的权重替换
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        # 用conv.weight和w做残差,结果做为conv的权重
        self.conv.weight.data += w
        self.__delattr__("cv2") # 删除cv2属性
        self.forward = self.forward_fuse # 更改对象的forward方法
class DWConv(Conv):
    # 假设 c1 = 6（输入通道数），c2 = 8（输出通道数），并且最大公约数 g = math.gcd(6, 8) = 2
    # g是指组数,输出通道数 (8)：表示最终的输出通道数。输入通道组数 (3)：表示每个输出通道对应的输入通道的一个子集（组），
    # 这里每个组包含 3 个输入通道。卷积核大小 (3x3)：表示卷积核的大小，这里是 3x3。
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
class LightConv(nn.Module):
    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False) # 点卷积
        self.conv2 = DWConv(c2, c2, k, act=act) # 深度卷积

    def forward(self, x): # (b,c1,...)
        # (b,c1,...)-->(b,c2,...)
        # 先点卷积切换通道,之后深度卷积处理
        # 用light_conv.conv1.conv.weight来访问属性权重
        # conv2_weight.shape[18, 1, 3, 3]
        # 18指输出和输入通道被分成18组,每组包含1个通道,
        # 每个组包含 1个输出和输入通道。18表示最终的输出通道数。
        # 输入通道组数 (18)：表示每个输出通道对应的输入通道的一个子集（组），这里每个组包含 1个输入通道。
        # 卷积核大小 (3x3)：表示卷积核的大小，这里是 3x3。
        return self.conv2(self.conv1(x))
class DWConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))
class ConvTranspose(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv_transpose(x)))
    def forward_fuse(self, x):
        return self.act(self.conv_transpose(x))
class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
    def forward(self, x):
        # 这里的切片切分空间操作,是先切分奇数行奇数列,之后是偶数行奇数列，之后奇数行偶数列，之后是偶数行偶数列
        # 之后在通道维度合并特征,整个空间采样每个像素位置都被恰好取样了一次。没有任何像素被重复取样,没有任何像素被遗漏
        # 合并特征后经过卷积处理,我第一感觉是这样采样,有助于模型发现图片数据的行梯度和列梯度的变化
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)  # 核大小为5的深度卷积
    def forward(self, x): # (b,c1,...)
        # gc=GhostConv(6,12,k=3)时,cv1.conv.weight.shape[6, 6, 3, 3]
        y = self.cv1(x) # (b,c_,...)
        # self.cv2(y):(b,c_,...)-->(b,c_,...)
        # 之后在通道维度合并通道,变成(b,2c_,...)
        # cv2是步长为1,核大小为5的深度卷积
        return torch.cat((y, self.cv2(y)), 1)
class RepConv(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # 使用批次标准化的条件
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False) #普通卷积
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False) # 点卷积
    def forward_fuse(self, x): # (b,c1,...) 卷积块处理
        return self.act(self.conv(x))
    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        # 如果id_out =self.bn(x),普通卷积和点卷积和原数据的批次标准化做残差连接
        return self.act(self.conv1(x) + self.conv2(x) + id_out)
    # 将各个卷积核及其偏置融合成一个等效的卷积核和偏置。
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    # 对于 batch normalization 层，如果存在 self.bn，则构造一个中间变量 self.id_tensor，
    # 这是一个形状为 (c1, input_dim, 3, 3) 的张量，其中心位置为 1，其余位置为 0。
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    # 这个方法创建一个新的卷积层，该层包含了等效的卷积核和偏置，并删除不再需要的旧卷积层。
    def fuse_convs(self):
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
class ChannelAttention(nn.Module):
    # 这个通道注意力少了一个压缩点卷积,有意为之吗?
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True) # 点卷积
        self.act = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor: # (b,c,...)
        # 平均池化后,每个通道经过点卷积混合通道信息,sigmoid为通道打重要性分数
        # 之后对原数据加权,这样随着训练的进行,重要的特征会越来越明显,不重要的特征会被忽略
        return x * self.act(self.fc(self.pool(x)))
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
    def forward(self, x):# (b,c,...)
        # 首先对x在索引1的维度求均值和最大值,就是对通道维度聚合操作,之后聚合后形状为(b,1,...),之后在特征轴合并特征
        # 之后经过卷积处理,通道变成1,之后sigmoid处理,这样sigmoid会给每个空间位置打分,这个分数表示空间位置对当前任务的重要性
        # 之后与原数据加权,这样空间的重要性被分配到各个通道的空间中
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1) # 通道注意力
        self.spatial_attention = SpatialAttention(kernel_size) # 空间注意力
    def forward(self, x): #(b,c1,...)
        # 先做通道注意力,之后做空间注意力
        return self.spatial_attention(self.channel_attention(x))
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x): # (b,c1)
        return torch.cat(x, self.d) # 对列表元素在通道维度合并特征
# GhostBottleneck 类是一个用于卷积神经网络中的瓶颈模块
class GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            # GhostConv：用于减少参数数量的模块。
            GhostConv(c1, c_, 1, 1),  # 点卷积切换通道,之后和深度卷积后的数据在通道维度合并
            # DWConv：深度可分离卷积，用于降低计算复杂度
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        # nn.Identity() 是 PyTorch 中的一个类，它代表一个不执行任何操作的模块。当你实例化 nn.Identity() 
        # 并将其作为模块的一部分时，它实际上会返回输入而不做任何改变。shortcut：用于实现残差连接的路径。
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 
            else nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False) if c1 != c2 else nn.Identity()
        )
    def forward(self, x): # (b,c1,...)
        return self.conv(x) + self.shortcut(x)

class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        # 点卷积
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float) 
        # 设置conv的权重
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1)) # (1,c,1,1)
        self.c1 = c1
    def forward(self, x):
        b, _, a = x.shape  # batch, channels, anchors
        # (b,c,a)-->(b,4,16,a)-->(b,16,4,a)-->softmax-->conv-->(b,4,a)
        # softmax在索引1的轴归一化
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
class Proto(nn.Module):
    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3) # 普通卷积,核大小3
        # 上采样,核大小是2
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  
        self.cv2 = Conv(c_, c_, k=3) # 普通卷积,核大小3
        self.cv3 = Conv(c_, c2)# 点卷积
    def forward(self, x):
        # (b,h,w,c1)-->(b,h,w,c_)-->(b,2h,2w,c_)-->(b,2h,2w,c_)-->(b,2h,2w,c2)
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))
class HGStem(nn.Module):
    def __init__(self, c1, cm, c2):
        super().__init__()
        # 核大小3,步长2
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        # 核大小2,步长1
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU()) # 核大小2,步长1
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU()) # 核大小3,步长2
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU()) # 点卷积
        # ceil_mode=True 指的是当计算输出尺寸的时候，使用向上取整而不是向下取整（默认行为）。这在某些情
        # 况下可能会导致输出的大小比通常预期的大一点。
        # 举个例子，如果输入的尺寸为 (N, C, H, W)，其中 N 是batch size，C 是通道数，H 和 W 分别是高
        # 度和宽度。假设 H 和 W 都是偶数，比如 (4, 4)，那么应用上述的 MaxPool2d 层之后，如果使用默认的 
        # floor 模式，输出的高度和宽度将是 (3, 3)；但如果使用了 ceil_mode=True，输出的高度和宽度将会是 (4, 4)。
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)
    def forward(self, x): # (b,h,w,c1)
        x = self.stem1(x) # (b,h/2,w/2,cm)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x) # (b,h/2,w/2,cm/2)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2) # (b,h/2,w/2,cm)
        x1 = self.pool(x) # (b,h/2,w/2,cm)
        x = torch.cat([x1, x2], dim=1) # (b,h/2,w/2,2cm)
        x = self.stem3(x)  # (b,h/4,w/4,cm)
        x = self.stem4(x) # (b,h/4,w/4,c2)
        return x
class HGBlock(nn.Module):
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        # 在nn.ModuleList中存储的对象是不同的对象，它们在内存中的地址也是不同的。nn.ModuleList是一个有序集合，
        # 用于存储子模块，它可以像普通的Python列表一样索引，但它确保每个添加的项都是torch.nn.Module的实例或其子类。
        # 每次调用block()时，都会创建一个新的实例，并将其添加到ModuleList中。这意味着每个block对象都有自己的状态和参数，
        # 并且在模型的前向传播过程中会被独立地调用。
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # 点卷积
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # 点卷积
        self.add = shortcut and c1 == c2 # 残差条件
    def forward(self, x):
        y = [x]
        # 这些block对象依次应用于输入数据
        # 这段代码遍历self.m中的每个block对象，并将上一个block的输出作为当前block的输入，从而形成一个序
        # 列化的处理流程。因为每次m(y[-1])处理后的结果都被添加到y中,而y[-1]是取出列表最后一个元素
        # 这样每次m处理后的结果都被当成了下次的输入
        y.extend(m(y[-1]) for m in self.m) # 6个卷积块处理
        # 在通道轴合并特征
        y = self.ec(self.sc(torch.cat(y, 1))) 
        # 满足残差条件,返回残差,否则返回处理后结果
        return y + x if self.add else y
class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1) # 点卷积
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1) # 点卷积
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
    def forward(self, x): # (b,h,w,c1)
        x = self.cv1(x) # (b,h,w,c_)
        # (b,h,w,c_)-->(b,h,w,4c_)-->(b,h,w,c2)
        # 这里不同尺寸的最大池化核会处理相同的x,之后列表内的对象在索引1的轴(通道轴)合并
        # self.m中的每个m处理的都是相同的输入x，而不是上一个最大池化的输出
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # 点卷积
        self.cv2 = Conv(c_ * 4, c2, 1, 1) # 点卷积
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        # (b,h,w,c1)-->(b,h,w,c_)
        y = [self.cv1(x)] 
        # 通过列表推导式和extend方法将多次迭代的结果追加到y列表中。每次迭代都会使用self.m
        # 处理列表y的最后一个元素，并将结果追加到y列表中。因此，y[-1]始终是指向列表中最新追加的元素
        y.extend(self.m(y[-1]) for _ in range(3))
        # 将列表y中的所有元素沿着通道维度（维度1）拼接起来
        # (b,4c_,h,w)-->(b,c2,h,w)
        return self.cv2(torch.cat(y, 1)) 
class C1(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1) # 点卷积
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))
    def forward(self, x):
        # (b,c1,h,w)-->(b,c2,h,w)
        y = self.cv1(x)
        # 序列化栈前后残差
        return self.m(y) + y
class Bottleneck(nn.Module): # 标准瓶颈模块
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1) # 普通卷积
        self.cv2 = Conv(c_, c2, k[1], 1, g=g) # g是groups
        # 残差条件:shortcut为True并且c1 == c2
        self.add = shortcut and c1 == c2 
    def forward(self, x): # (b,c1,h,w)
        # self.cv2(self.cv1(x)):(b,c1,h,w)-->(b,c_,h,w)-->(b,c2,h,w)
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2fAttn(nn.Module):
    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1) # 点卷积
        # optional act=FReLU(c2)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # 点卷积 
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)
    # x:(b,c1,...),guide:(b,d)
    def forward(self, x, guide):
        # (b,c1,...)-->(b,2c,...),之后在通道维度拆分
        y = list(self.cv1(x).chunk(2, 1)) # 这里有两个,每个的通道数都是c
        # 每次m处理的都是y列表中的最后一个元素
        y.extend(m(y[-1]) for m in self.m) # nc
        y.append(self.attn(y[-1], guide)) # (b,c,h,w)
        # 之后把y中的元素在通道维度合并特征,之后经过点卷积切换通道
        # (b,(n+3)c,...)-->(b,c2,...)
        return self.cv2(torch.cat(y, 1))
    def forward_split(self, x, guide):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

class BottleneckCSP(nn.Module): # 瓶颈模块
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1) # 点卷积
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False) 
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):# (b,c1,...)
        # (b,c1,...)-->(b,c_,...)-->(b,c_,...)-->b,c_,...)
        y1 = self.cv3(self.m(self.cv1(x)))
        # (b,c1,...)-->(b,c_,...)
        y2 = self.cv2(x)
        # (b,2c_,...)-->(b,c2,...)-
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
# 在神经网络设计中，激活函数的应用原则如下：
# 在卷积层之后通常使用激活函数，以引入非线性。
# 这是因为卷积操作本身是线性的，只有加上非线性激活函数，网络才能学习到复杂的特征。
# 在网络的最后一层通常不使用激活函数。
# 在网络的最后一层（通常是分类或回归任务的输出层），通常不使用激活函数。这是因为最后一层的任务是输出概率分布
# （分类任务）或者直接预测目标值（回归任务），而激活函数可能会干扰这些值的计算。
# 在残差连接之前的卷积层通常不使用激活函数，以便于保持残差项的线性特性。
# 在残差连接之后使用全局激活函数，以确保整体非线性。
# 在残差连接（residual connection）中，通常会在残差分支的末尾不使用激活函数，而在残差连接之后使用激活函数。
# 这是为了避免在残差连接之前过早引入非线性，从而影响残差项的线性特性。
class ResNetBlock(nn.Module): # 具有标准卷积层的ResNet块。
    def __init__(self, c1, c2, s=1, e=4):
        super().__init__()
        c3 = e * c2
        # cv1：第一个卷积层使用激活函数（act=True），这是为了让网络能够学到更丰富的特征表示。
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True) # 点卷积
        # 第二个卷积层也使用激活函数（act=True），继续增强非线性表示能力。
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True) # 普通卷积
        # 第三个卷积层不使用激活函数（act=False），这是因为这个卷积层之后会有残差连接，并且残差连接之后会有一
        # 个全局的激活函数（这里使用的是 ReLU）。
        self.cv3 = Conv(c2, c3, k=1, act=False) # 点卷积
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()
    def forward(self, x): # (b,c1,...)
        print(self.shortcut)
        # (b,c1,...)-->(b,c2,...)-->(b,c2,...)-->(b,c3,...)
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))
class ResNetLayer(nn.Module):
    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        super().__init__()
        self.is_first = is_first
        if self.is_first:
            # 3-->填充1,5-->填充2,7-->填充3
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)
    def forward(self, x): # (b,c1,...)
        #(b,c1,...)-->(b,c3)
        return self.layer(x)

class C2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1) # 点卷积
        self.cv2 = Conv(2 * self.c, c2, 1)  
        self.m = nn.Sequential(*(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
    def forward(self, x): # (b,c1,h,w)
        # (b,c1,h,w)-->(b,2c,h,w)
        # chunk(2, 1)：将输出张量沿着通道维度（维度1）分割成两个部分，分别命名为a和b。
        # 2是指分成两个部分,1指索引1的轴
        a, b = self.cv1(x).chunk(2, 1)
        # 将分割出的a传递给模块self.m,将处理后的a与未处理的b沿通道维度拼接起来。
        # 将拼接后的张量再次通过卷积层cv2进行处理
        # (b,2c,...)-->(b,c2,...)
        return self.cv2(torch.cat((self.m(a), b), 1))
# 引导向量的空间注意力机制
class MaxSigmoidAttnBlock(nn.Module):
    # gc 是引导向量（guide）的维度。scale 决定是否使用可学习的缩放因子
    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False): 
        super().__init__()
        self.nh = nh # 注意力头的数量
        self.hc = c2 // nh 
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None # 点卷积切换通道
        # ec 是嵌入通道数，用于调整输入特征图的维度
        self.gl = nn.Linear(gc, ec) # 创建一个线性层 gl 
        self.bias = nn.Parameter(torch.zeros(nh)) # 创建一个可学习的截距项 bias
        # 创建一个卷积层 proj_conv 用于调整输入的维度到期望的输出通道数 c2
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)  # 普通卷积
        # 如果 scale 参数为 True，则创建一个可学习的缩放因子；否则，使用常数值 1.0
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0
    def forward(self, x, guide): # x:(b,c1,h,w),guide:(b,gc),引导向量
        bs, _, h, w = x.shape 
        guide = self.gl(guide) # (b,gc)-->(b,ec)
        # (b,ec)-->(b,n,m,dc)
        guide = guide.view(bs, -1, self.nh, self.hc)
        # (b,c1,h,w)-->(b,ec,...),用来切换原特征图通道到嵌入通道
        embed = self.ec(x) if self.ec is not None else x
        # (b,ec,h,w)-->(b,m,dc,h,w)
        embed = embed.view(bs, self.nh, self.hc, h, w)
        # (b,m,dc,h,w)@(b,n,m,dc)-->(b,m,h,w,n)
        # 匹配索引：在 einsum 的子句中，m 和 dc 在两个输入张量中都有出现，这意味着它们将在计算中进行匹配。
        # 乘法操作：对于 embed 和 guide 中相同索引的维度，将进行逐元素乘法操作。
        # 求和操作：在 einsum 子句中重复的索引（如 m 和 dc）将在乘法之后进行求和操作
        # 输出形状：最终输出的形状由 -> bmhwn 指定，这意味着输出张量将保留 b, m, h, w, n 这些维度，并按照这个顺序排列
        # 乘法发生在 dc 和 m 上，即 embed[b, m, dc, h, w] * guide[b, n, m, dc]。
        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        # 在n的维度聚合最大值,[0]表示不要argmax
        aw = aw.max(dim=-1)[0] 
        # 缩放
        aw = aw / (self.hc**0.5)
        # [None, :, None, None]就是[1,m,1,1]
        aw = aw + self.bias[None, :, None, None]
        # 给空间位置打分,区分空间位置的重要性
        aw = aw.sigmoid() * self.scale # (b,m,h,w)
        # (b,c1,h,w)-->(b,c2,h,w)
        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w) # (b,m,dc,h,w)
        # (b,m,dc,h,w)*(b,m,1,h,w)-->(b,m,dc,h,w)
        # 用aw对x做加权,这样得到的空间数据就受到空间注意力权重的影响
        # 图片数据应该会和引导向量相关
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w) # (b,c2,h,w)

class ImagePoolingAttn(nn.Module):
    """通过图像感知信息增强文本嵌入"""
    # 旨在通过结合图像的信息来增强文本的嵌入表示，使得文本嵌入能够更好地捕捉到与图像相关的内容。
    # 这种方法通常用于多模态任务中，比如图像字幕生成、视觉问答（VQA）等场景。
    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        super().__init__()
        nf = len(ch) # 不同尺寸特征图的个数
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec # 嵌入维度
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh # dc
        self.k = k
    def forward(self, x, text):
        # x:是不同尺寸的特征图列表,text:(b,s,ct)
        bs = x[0].shape[0] # 批次大小
        assert len(x) == self.nf # 不同尺寸特征图的个数
        num_patches = self.k**2 
        # x 是一个列表：x 包含了不同尺寸的特征图，每个特征图的形状可能是不同的。
        # self.projections 是一个 ModuleList：包含了一系列的 nn.Conv2d 层，用于将不同尺寸的特征图转换到相同的维度 ec。
        # self.im_pools 是一个 ModuleList：包含了一系列的 nn.AdaptiveMaxPool2d 层，用于将不同尺寸的特征图转换为固定大小 (k, k)。
        # feat 是原始特征图列表中的一个元素。proj 是 self.projections 中的一个卷积层,pool 是 self.im_pools 中的一个池化层。
        # pool(proj(x)) 应用了池化操作，将卷积后的特征图转换为固定大小 (k, k)
        # 对于 x 列表中的每一个特征图，先通过相应的卷积层调整维度，然后通过相应的池化层调整大小
        # 每个特征图经过上述操作后，都被展平成 (b,ec,num_patches) 的形状
        x = [pool(proj(x)).view(bs, -1, num_patches) for (feat, proj, pool) in zip(x, self.projections, self.im_pools)]
        # 在最后一个维度合并,(b,ec,nf*p)-->(b,nf*p,ec)
        x = torch.cat(x, dim=-1).transpose(1, 2)
        # (b,s,ct)-->(b,s,ec)
        q = self.query(text)
        # 线性转换和点卷积切换通道不一样的地方是:点卷积切换通道,通道在索引1的轴,而线性转换,线性特征维度是在最后一个维度
        k = self.key(x) # (b,nf*p,ec)
        v = self.value(x) # (b,nf*p,ec)
        q = q.reshape(bs, -1, self.nh, self.hc) # (b,s,m,dc)
        k = k.reshape(bs, -1, self.nh, self.hc) # (b,nf*p,m,dc)
        v = v.reshape(bs, -1, self.nh, self.hc) # (b,nf*p,m,dc)
        # 对通道维度做元素乘积并聚合,nf*p 表示不同尺寸特征图经过池化后的特征数量
        # 乘法发生在 dc 维度上,在 dc 维度上进行求和，得到的结果形状为 (b, m, s, nf*p)
        # 第一个 torch.einsum 操作计算了注意力权重矩阵 aw，其形状为 (b, m, s, nf*p)。
        # 这个矩阵用于描述每个 token (s) 在每个注意力头 (m) 上与其他特征图 (nf*p) 的相关性。
        aw = torch.einsum("bnmc,bkmc->bmnk", q, k) # (b,m,s,nf*p)
        aw = aw / (self.hc**0.5) # 缩放
        # 对浓缩合并的特征图空间向量做softmax归一化
        aw = F.softmax(aw, dim=-1)
        # (b,m,s,nf*p) (b,nf*p,m,dc)-->(b,s,m,dc)
        # nf*p 表示不同尺寸特征图经过池化后的特征数量。
        # 对nf*p向量做的对应元素相乘之后聚合相加的操作
        # 就是相当于对文本序列中的token表示做了加权
        # 对于 aw 和 v 中的每一个 (b, m, s, nf*p) 和 (b,nf*p,m,dc)，我们将进行逐元素乘法操作
        # 乘法发生在 nf*p 维度上,在 nf*p 维度上进行求和，得到的结果形状为 (b, s, m, dc)。
        # 第二个 torch.einsum 操作使用计算出的注意力权重矩阵 aw 对值向量 v 进行加权求和，得到的结果形状为 
        # (b, s, m, dc)。这个操作相当于对文本序列中的 token 表示进行了加权，从而增强了文本嵌入中的图像感知信息。
        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        # b,s,m,dc)-->(b,s,ec)-->(b,s,ct)
        x = self.proj(x.reshape(bs, -1, self.ec))
        # 残差连接
        return x * self.scale + text

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):# (b,c1,...)
        # (b,c1,...)-->(b,2c,...),在通道维度分成两份
        y = list(self.cv1(x).chunk(2, 1))
        # 依次追加self.m中每个m处理后的结果,y[-1]是上个处理结果
        y.extend(m(y[-1]) for m in self.m)
        # 将列表中的所有特征图在通道轴合并
        # (b,(2 + n)c,...)-->(b,c2,...)
        return self.cv2(torch.cat(y, 1))
    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1) # 点卷积
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
    def forward(self, x): # (b,c1,...)
        # (b,c1,...)-->(b,c2,...)-->(b,c2,...)
        # 先点卷积切换通道,之后深度卷积
        return self.cv2(self.cv1(x))
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads # h
        self.head_dim = dim // num_heads # dk
        self.key_dim = int(self.head_dim * attn_ratio) 
        self.scale = self.key_dim**-0.5 # 缩放系数
        nh_kd = self.key_dim * num_heads # 所有头加起来的维数
        h = dim + nh_kd * 2  # 中间维度
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False) # 点卷积
        # 深度卷积
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
    def forward(self, x): # (b,d,...)
        B, C, H, W = x.shape
        N = H * W 
        # (b,d,...)-->(b,h,...)
        qkv = self.qkv(x) 
        # (b,h,...)-->(b,h,dk+k_dk*2,s),在通道维度进行拆分,q,k特征维度相同
        # v单独一个特征维度
        # q和k主要用于计算注意力权重，因此减少它们的特征维度可以减少冗余信息，提高计算效率。
        # v用于根据注意力权重进行加权求和，保持较大的特征维度可以保留更多的信息。
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        # (b,h,s_q,k_dk)@(b,h,k_dk,s_k)-->(b,h,s_q,s_k)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1) # 在s_k上归一化
        # (b,h,dk,s_v)@(b,h,s_k,s_q)-->(b,h,dk,s_q)-->(b,d,h,w)
        # v:(b,h,dk,s)-->(b,d,h,w)-->(b,d,...)
        # 注意力后的v和经过深度卷积的v做残差
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        # (b,d,h,w)-->(b,d,h,w)
        x = self.proj(x)
        return x

# 对比损失（contrastive loss）的头部模块，通常用于对比学习任务中。对比学习的目标是拉近正样本之间的距离，
# 同时推开负样本之间的距离。这个模块通过计算归一化的特征图和权重之间的相似度，并对其进行缩放和偏移，从而生成
# 最终的对比损失分数。
class ContrastiveHead(nn.Module):
    def __init__(self):
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        # self.bias：一个可学习的偏置项，初始值设置为 -10.0，目的是为了保持初始分类损失与其他损失的一致性
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # self.logit_scale：一个可学习的缩放因子，初始值设置为 log(1/0.07)，这是为了让缩放因子的初始值接近 
        # 14（1/0.07 ≈ 14），这是一个常见的选择。
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())
    def forward(self, x, w):
        # x：形状为 (b, c, h, w) 的特征图，其中 b 是批量大小，c 是通道数，h 和 w 是高度和宽度。
        # w：形状为 (b, k, c) 的权重，其中 b 是批量大小，k 是权重的数量，c 是通道数。
        x = F.normalize(x, dim=1, p=2) # x 被归一化，使其在第 1 维（通道维度）具有单位范数。
        w = F.normalize(w, dim=-1, p=2) # w 被归一化，使其在最后一维（通道维度）具有单位范数。
        #　计算相似度,使用 torch.einsum("bchw,bkc->bkhw", x, w) 计算特征图 x 和权重 w 之间的点积。
        # 这里 x 的形状为 (b, c, h, w)，w 的形状为 (b, k, c)，结果的形状为 (b, k, h, w)。
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        # x * self.logit_scale.exp()：将结果乘以缩放因子的指数形式。+ self.bias：加上偏置项。
        # 这个模块的目的是通过计算特征图和权重之间的相似度，并对其进行归一化、缩放和偏移，从而生成最终的对比损失分数。
        # 这些分数可以用于后续的损失计算，以促进正样本之间的相似性和负样本之间的差异性。
        return x * self.logit_scale.exp() + self.bias
class BNContrastiveHead(nn.Module):
    def __init__(self, embed_dims: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
    def forward(self, x, w):
        # x:(b,c,h,w),w:(b,k,c)
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2) # w 被归一化
        # (bchw)@bkc-->(b,k,h,w)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias
class RepBottleneck(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)
class RepCSP(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
class RepNCSPELAN4(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
class ELAN1(RepNCSPELAN4):
    def __init__(self, c1, c2, c3, c4):
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
# 它首先对输入张量进行平均池化，然后通过一个卷积层进行卷积操作。这种设计可以在一定程度上增加模型的感
# 受野，并且通过平均池化可以减少特征图的大小，从而降低计算复杂度。
class AConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # self.cv1 是一个卷积层，输入通道数为 c1，输出通道数为 c2，卷积核大小为 3，步长为 2，填充为 1。
        self.cv1 = Conv(c1, c2, 3, 2, 1)
    def forward(self, x):
        # x 是输入张量。kernel_size=2：池化窗口大小为 2x2,步长为 1。
        # 不使用填充,不使用向上取整模式。在计算平均值时包含边界填充。
        # 计算填充和不计算填充的内部数据区别不大,接近填充的边缘区别很大
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        # print(x.shape) # ([1, 6, 4, 4])
        #将经过平均池化后的输入通过卷积层 cv1。
        return self.cv1(x)

class ADown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1) # 普通卷积中的下采样
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0) # 点卷积
    def forward(self, x): # (b,c1,...)
        # 平均池化,计算填充数据,未填充
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1) # 在通道轴拆分
        x1 = self.cv1(x1) # c1-->c,下采样,特征图尺寸减半
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1) # 最大值池化,尺寸减半
        x2 = self.cv2(x2) # c1-->c
        # 在通道轴合并特征(b,2c,...)
        return torch.cat((x1, x2), 1)
class SPPELAN(nn.Module):
    def __init__(self, c1, c2, c3, k=5):
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1) # 点卷积
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1) # 点卷积
    def forward(self, x): # (b,c1,...)
        y = [self.cv1(x)] # 用来存放各种处理结果c1-->c3
        # 把m(y[-1])处理的结果追加进列表,y[-1]是列表中的最后一个
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        # 在通道上合并y列表中的特征图,4 * c3-->c2
        return self.cv5(torch.cat(y, 1))
class CBLinear(nn.Module):
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)
    def forward(self, x): # c1
        # c1-->sum(c2s),之后在通道维度拆分
        return self.conv(x).split(self.c2s, dim=1)
# CBFuse 类实现了一个用于选择性特征融合的模块。这个模块接收一个索引列表 idx，用于从输入的特征列表 xs 
# 中选择特定的特征图，并将它们上采样到相同的尺寸，然后进行融合。
class CBFuse(nn.Module):
    def __init__(self, idx):
        super().__init__() # 调用父类的构造函数。
        self.idx = idx #存放要融合的特征图的索引
    def forward(self, xs):
        target_size = xs[-1].shape[2:] # 获取最后一个特征图的尺寸（高度 h 和宽度 w)
        #　选择需要的特征图
        selected_features = [xs[i] for i in self.idx]
        # F.interpolate 用于将所有选定的特征图调整到与最后一个特征图相同的尺寸。由于最后一个特征图的尺寸是最小的，因此实际上是进行下采样操作
        res = [F.interpolate(feature, size=target_size, mode="nearest") for feature in selected_features]
        stack_res=torch.stack(res + xs[-1:]) # [3, 1, 32, 8, 8]
        print(stack_res.shape)
        return torch.sum(stack_res, dim=0)

class PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1) # 点卷积切通道
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))
    def forward(self, x): # (b,c1,h,w)
        # (b,c1,h,w)-->(b,2c,...),split会在通道维度拆分
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        # 对b做注意力前后的残差连接
        b = b + self.attn(b) # (b,c,...)
        # (b,c,...)-->(b,2c,...)-->(b,c,...)
        # 整个多头自注意力过程没有层标准化和dropout
        b = b + self.ffn(b)  # 前馈前后残差
        # 在通道维度合并a,b的特征,之后经过卷积处理
        # (b,2c,...)-->(b,c1,...)
        return self.cv2(torch.cat((a, b), 1))
class C2fCIB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1) # 点卷积
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
    def forward(self, x): # (b,c1,...)
        # x经过cv1,之后被m模块处理,另一个x经过cv2处理,之后两者在通道维度合并特征
        # (b,c1,...)-->(b,c_,...)-->(b,c,...)
        # b,c1,...)-->(b,c_,...)
        # 之后在通道维度合并,-->(b,2c,...)-->(b,c2,...)
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
class C3x(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))
class C3TR(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)
class C3Ghost(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))
def fuse_conv_and_bn(conv, bn):
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )
    # Prepare filters
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False) # 普通卷积,填充
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()
    def forward(self, x): # (b,ed)
        # 对输入做不同核大小的卷积操作,之后残差连接
        return self.act(self.conv(x) + self.conv1(x))
    def forward_fuse(self, x): # (b,ed)
        return self.act(self.conv(x))
    @torch.no_grad()
    def fuse(self):
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1
class CIB(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv1(x) if self.add else self.cv1(x)

class RepC3(nn.Module):
    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()
    def forward(self, x):
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))

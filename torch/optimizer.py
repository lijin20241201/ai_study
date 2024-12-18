# 这个激活函数通过结合Softplus函数和几个可学习的参数（如lambda (lambd) 和 kappa (kappa)）来增强模型的灵活性和表达能力
class AGLU(nn.Module):
    def __init__(self, device=None, dtype=None) -> None:
        # 通过super().__init__()调用父类nn.Module的初始化方法，确保模块的基本功能被正确设置
        super().__init__()
        # 使用nn.Softplus(beta=-1.0)定义了一个Softplus激活函数，其中beta参数被设置为-1.0。
        self.act = nn.Softplus(beta=-1.0)
        # 通过nn.Parameter包装并初始化为均匀分布的随机数。这个参数在后续的前向传播中会被用来调整激活函数的形状或灵敏度。
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda parameter
        # 另一个可学习的参数，同样初始化为均匀分布的随机数。它用于缩放输入x，从而在激活函数中引入更多的灵活性。
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 首先，将lambd（lambda）参数通过torch.clamp函数限制在最小值为0.0001，以避免在后续计算中出现除以零的情况。
        lam = torch.clamp(self.lambd, min=0.0001)
        # 首先通过self.kappa * x缩放输入x，然后减去torch.log(lam)，接着将这个结果传递给Softplus函数。最后，
        # 整个表达式被torch.exp函数以1/lam为底数进行指数运算。这个操作序列产生了最终的激活输出。
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))
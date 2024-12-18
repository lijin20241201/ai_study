# 创建一个随机张量
x = torch.randn(3, 3)
print("Original x:", x)
# 使用 inplace=False
relu_non_inplace = nn.ReLU(inplace=False)
y = relu_non_inplace(x)
print("Non-inplace operation (y):", y)
print("x after non-inplace operation:", x)
# 使用 inplace=True
relu_inplace = nn.ReLU(inplace=True)
z = relu_inplace(x)
print("Inplace operation (z):", z)
print("x after inplace operation:", x) # x的值被修改
def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)
from torchvision.models.detection.image_list import ImageList
images = [torch.rand(3, h, w) for h, w in [(200, 300), (150, 100), (180, 120)]]
tensors, image_sizes = pad_to_max(images)
image_list = ImageList(tensors, image_sizes)
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        feature_maps = []
        for layer in self.conv_layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps
model = SimpleConvNet()
# 随机生成一批输入图像
batch_size = 4
input_data = torch.randn(batch_size, 3, 224, 224)
feature_maps = model(input_data)
for i, feature_map in enumerate(feature_maps):
    print(f"Feature map {i + 1} size: {feature_map.size()}")
outputs=anchors(image_list,feature_maps[1:])

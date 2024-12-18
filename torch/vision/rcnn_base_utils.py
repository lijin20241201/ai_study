class ImageList:
    # tensors: 包含图像数据的张量。这些图像已经被填充（pad）到了相同的大小，因此可以作为一个批处理输入到模型中。
    # image_sizes: 是一个列表，其中每个元素是一个元组 (height, width)，表示每个图像的原始尺寸。
    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes
    # to() 方法: 允许将 ImageList 中的张量移动到指定的设备（例如 CPU 或 GPU）。这在进行深度学习操作
    # 时是非常有用的，因为模型和数据需要位于同一个设备上。
    def to(self, device: torch.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)
# 在某些场景下，保持图像的原始尺寸可能更为重要，这时就需要在批次级别进行填充（padding）
# 保持原始比例：如果调整图像大小会导致严重的失真或者信息丢失，那么保留原始尺寸并在批次中进行填充可能是更好的选择。
# 动态输入：对于某些模型，特别是那些设计用于处理可变输入尺寸的模型，如一些目标检测模型（如 Faster R-CNN），
# 保持图像原始尺寸并在批次内部进行填充是必要的。
# 增强模型泛化能力：在训练过程中使用不同尺寸的图像可以增强模型对不同输入尺寸的鲁棒性。
def pad_to_max(images: List[torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    max_height = max([img.shape[-2] for img in images])
    max_width = max([img.shape[-1] for img in images])
    padded_images = []
    image_sizes = []
    for img in images:
        padding = (0, max_width - img.shape[-1], 0, max_height - img.shape[-2])
        padded_img = torch.nn.functional.pad(img, padding)
        padded_images.append(padded_img)
        image_sizes.append((img.shape[-2], img.shape[-1]))
    return torch.stack(padded_images), image_sizes

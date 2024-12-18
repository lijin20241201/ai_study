# id: 2, name: "bicycle" 自行车
# id: 3, name: "car" 汽车
# id: 4, name: "motorcycle" 摩托车
# id: 5, name: "airplane" 飞机
# id: 6, name: "bus" 公交车
# id: 7, name: "train" 火车
# id: 8, name: "truck" 卡车
# id: 9, name: "boat" 船
# image_id: 558840 - 表示该标注对象属于图像ID为558840的那张图像。
# category_id: 58 - 表示该标注对象属于类别ID为58的那个类别。
# id: 156 - 表示该标注对象的唯一标识符为156，这使得我们可以在同一张图像中区分不同的标注对象。
# 人物（Person）：与人类相关的类别。
# 车辆（Vehicle）：包括各种类型的交通工具。
# 户外（Outdoor）：户外环境中常见的物体。
# 动物（Animal）：各种动物。
# 器具（Appliance）：家用电器。
# 家具（Furniture）：家里的家具。
# 电子设备（Electronic）：如手机、电视等。
# 食物（Food）：食物和饮品。
# 运动器材（Sports equipment）：如球类、滑板等。
# supercategory: 超类别是指一类相似物品的大类。例如，'electronic'表示电子设备，'appliance'表示家用
# 电器，而'indoor'可能指的是室内物品或者室内场景的一部分。超类别可以帮助识别类别之间的关系，并且在某些情
# 况下，可以用来聚合相关类别。
# id: 这是该类别在整个数据集中唯一的整数标识符。在进行模型训练或者评估时，类别ID通常用于标记图像中的对
# 象。不同的ID对应不同的类别。
# name: 类别的名称，这是一个字符串，用来描述具体是什么类型的物体。例如，'cell phone'代表手机，
# 'microwave'代表微波炉等。 
def swap_xy(boxes):
    # 交换边界框的x和y坐标
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
def convert_to_xywh(boxes):
    # 将框的格式更改为中心,宽和高。
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )
def convert_to_corners(boxes):
    """将框格式更改为角坐标(x1,y1,x2,y2) """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )
# 计算两个边界框集合之间交并比的函数,IoU是物体检测中常用的一个指标，用于衡量两个边界框的重叠程度。
def compute_iou(boxes1, boxes2):
     # 输入的boxes格式为(x, y, width, height)，其中(x, y)为中心点坐标
    boxes1_corners = convert_to_corners(boxes1) # (x1,y1,x2,y2)
    boxes2_corners = convert_to_corners(boxes2) # (x1,y1,x2,y2)
    # 设定两个框左上角点坐标值的较大者
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2]) 
    # 设定两个框右下角点坐标值的较小者,这两步是为了取两个框的重叠部分
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    # 计算交集的宽度和高度，如果两个框不相交，则宽度或高度为0
    intersection = tf.maximum(0.0, rd - lu) 
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1] #交集面积
    boxes1_area = boxes1[:, 2] * boxes1[:, 3] # boxes1的面积=中心点形式中boxes1的宽高乘积
    boxes2_area = boxes2[:, 2] * boxes2[:, 3] # boxes2的面积=中心点形式中boxes2的宽高乘积
    union_area = tf.maximum( # boxes1和boxes2的并集是两者面积和-两者的交集,下面返回两者的较大值
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    # 返回裁剪后的值，这个值就是交并比
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)
# 用于可视化图像中的检测结果。函数接收图像、边界框、类别标签和置信度得分作为输入，并在图像上绘制
# 边界框及相应的标签信息。
# 输入的图像，通常是一个NumPy数组。包含检测到的边界框的列表或数组，每个边界框是一个四维向量 
# [x1, y1, x2, y2]，表示边界框的左上角和右下角坐标。
# 包含每个边界框对应的类别标签的列表或数组。包含每个边界框对应的置信度得分的列表或数组。
# Matplotlib绘图窗口的尺寸，默认为 (7, 7)。边界框线条的宽度，默认为 1。边界框的颜色，默认为红色 [0, 0, 1]
def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    # 将输入的图像转换为NumPy数组，并确保其数据类型为 uint8，这是图像数据常见的存储格式。
    image = np.array(image, dtype=np.uint8) # ndarray
    # 创建一个指定大小的绘图窗口，并关闭坐标轴显示。接着，将图像绘制到窗口中，并获取当前的绘图区域。
    plt.figure(figsize=figsize) # 画布大小
    plt.axis("off") # 关闭轴显示
    plt.imshow(image)
    ax = plt.gca() # 获取当前视图
    # 遍历每个检测到的边界框、类别和得分。
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score) # 格式化类别，分数文本串
        x1, y1, x2, y2 = box # 拆包box坐标
        w, h = x2 - x1, y2 - y1 # 获取框的宽，高
        # 对每个边界框，构造一个描述类别和得分的文本字符串，并计算边界框的宽度和高度。然后，使用Rectangle对
        # 象绘制边界框，并将其添加到绘图区域。
        patch = plt.Rectangle( # 画边界框
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        
        ax.add_patch(patch) # 加进当前视图
        # 在每个边界框的左上角位置添加一个文本标签，显示类别和得分。使用一个带有颜色和透明度的矩形
        # 框包围文本，以增强可读性。
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax
# 用于目标检测的预设框
class AnchorBox: 
   # aspect_ratios：一个浮点数值列表，表示特征图上每个位置处锚框的宽高比。
    # scales：一个浮点数值列表，表示特征图上每个位置处锚框的缩放比例。
    # num_anchors：特征图上每个位置处的锚框数量。
    # areas：一个浮点数值列表，对应不同尺寸特征图下锚框的面积(特征图越小,面积越大)
    # strides：一个浮点数值列表，想对于原图,下采样的步伐,步长越长,对应的特征图越小
    def __init__(self):
        # 不同宽高比,对应面积相同,宽高却不同的锚框
        self.aspect_ratios = [0.5, 1.0, 2.0] 
        # 缩放系数,用于对同一尺寸特征图下面积相同的锚框进行缩放
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]] 
        # 对应特征图中每个位置的9个锚框
        self._num_anchors = len(self.aspect_ratios) * len(self.scales) 
        # 步长对应不同尺寸的特征图,步长越长,特征图尺寸越小
        self._strides = [2 ** i for i in range(3, 8)] 
        # 面积对应不同尺寸的特征图,小的面积用来在大尺寸的特征图上检查目标,大的面积用来在小的特征图上检查目标
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]] 
        # 计算不同尺寸特征图下的9种不同尺寸和宽高比的锚框
        self._anchor_dims = self._compute_dims() 
    def _compute_dims(self): # 带下划线是protected方法,不推荐让外部调用
        anchor_dims_all = [] # 列表,用来装5种特征图尺寸下形状为(1,9,2)的锚框
        # 遍历5种不同面积的锚框(对应不同尺寸的特征图)
        for area in self._areas:
            # 用来装9种缩放系数和宽高比不同的锚框
            anchor_dims = [] 
            # 对于每个给定的面积，代码通过循环遍历所有的长宽比来计算相应的锚框宽度和高度。
            for ratio in self.aspect_ratios: 
                anchor_height = tf.math.sqrt(area / ratio) # 锚框高
                anchor_width = area / anchor_height # 锚框宽
                # 变形(1,1,2)
                dims = tf.reshape( 
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                # 对于当前面积的锚框缩放,最终获得某个位置的9个大小形状不同的锚框
                for scale in self.scales: 
                    anchor_dims.append(scale * dims) 
            # 在锚框数维度堆叠,形状变成(1,1,9,2),因为索引是从列表项开始算的
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2)) #(1,9,2)
        # print(f'{anchor_dims_all[0].shape=}') #[1, 1, 9, 2]
        # 返回的是5种面积下的9种锚框,列表长度5,表示5种不同的面积(对应5种不同程度的特征图)下的9种特征图
        return anchor_dims_all
    # 参数:特征图的高,特征图的宽,特征图的层级
    def _get_anchors(self, feature_height, feature_width, level):
        # 生成中心点坐标：rx 和 ry 分别生成了特征图宽度和高度方向上每个像素点中心的 X 和 Y 坐标。
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5 
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5 
        # 调整中心点坐标：使用 tf.meshgrid(rx, ry) 生成网格坐标，得到 (feature_height, feature_width) 
        # 形状的张量，分别表示每个像素点中心的 X 和 Y 坐标。
        # tf.stack(tf.meshgrid(rx, ry), axis=-1) 将这两个网格坐标堆叠在一起，形成一个形状为 
        # (feature_height, feature_width, 2) 的张量，其中最后一个维度存储了每个像素点的 (x, y) 坐标。
        # 乘于步长是因为要还原到原图的计算单位
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2) # (f_h,f_w,1,2)
        # 复制中心点坐标以匹配锚框数量：centers = tf.tile(centers, [1, 1, self._num_anchors, 1]) 
        # 复制中心点坐标，使得每个位置有 _num_anchors 个相同的坐标，形状变为 (feature_height, 
        #                                           feature_width, _num_anchors, 2)。
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        # 对于特定层级的特征图,在每个位置(这里的位置指单位位置1x1)都对应9个锚框,形状为(feat_h,feat_w,9,2)
        # 也就是说,每个中心点都对应9种不同尺寸的锚框(14, 14, 9, 2)
        dims = tf.tile( 
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        # 组合中心点坐标和尺寸：将中心点坐标和尺寸拼接在一起，形成 最后一个轴(x, y, width, height) 格式的锚框。
        anchors = tf.concat([centers, dims], axis=-1) 
        # 调整锚框形状：将所有锚框展平成一个二维张量，其中每一行代表一个锚框的信息，形状为 (h*w*n_anchors, 4)。
        return tf.reshape( # 变形成(h*w*n_anchors,4)
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )
    # 最终返回的张量包含了所有特征层的所有锚框信息，形状为 (total_anchors, 4)，其中 total_anchors 
    # 是所有特征层的锚框总数，4 表示每个锚框的 (x, y, width, height) 信息。
    # 这段代码有效地处理了不同尺度的特征图，并且确保每个特征图位置都有多个不同大小和长宽比的锚框，这
    # 有助于模型在不同尺度上检测目标。
    def get_anchors(self, image_height, image_width):
        # 循环生成每个特征层的锚框：循环遍历特征金字塔中的特征层，这里特征金字塔有5 层（第 3 层到第 7 层）
        # 计算特征层的高宽：tf.math.ceil(image_height / 2 ** i) 和 tf.math.ceil(
        # image_width / 2 ** i) 计算每个特征层的高和宽。这里使用了向上取整操作 (tf.math.ceil)，
        # 以确保得到的特征图尺寸是一个整数。除以 2 ** i 表示随着特征层的深入，特征图的尺寸会逐渐减小
        # ，这是因为每一层都是上一层尺寸的一半（假设使用了 2 倍的下采样率）
        # 调用 _get_anchors 方法：对于每个特征层，调用 _get_anchors 方法来生成该层的锚框。
        # 传入的参数包括特征层的高、宽以及层的索引 i。
        # 收集所有特征层的锚框：通过列表推导式 [...] for i in range(3, 8)] 收集所有特征层的锚框列表。
        # 合并所有锚框：使用 tf.concat 函数沿第一个轴（默认为 0 轴）将所有特征层的锚框合并成一个大的张量。
        # (64,32,16,8,4),比方说:4x4的特征图上就有4*4*9个锚框
        # 越小的特征图上使用的锚框尺寸越大。这是因为特征图的尺寸随着层次的加深而减小，而每一层特征图对应的原
        # 始图像区域也在增大。因此，深层特征图上的单个像素点代表了更大的图像区域，更适合用来检测较大的物体。
        # 特征金字塔与锚框尺寸的关系:特征图层级通常是从浅到深排列的。浅层特征图尺寸较大，而深层特征图尺寸较小。
        # 在特征金字塔中，通常在较浅的特征图上使用较小的锚框，而在较深的特征图上使用较大的锚框。
        # 较小的目标在图像中占据的空间较小，因此在特征金字塔的浅层更容易检测到。
        # 较大的目标在图像中占据的空间较大，因此在特征金字塔的深层更容易检测到。
        # 浅层特征图保留了更多的细节信息，这对于检测小目标非常有用。
        # 深层特征图提供了更好的全局上下文信息，这对于检测大目标非常有用。
        # 随着特征图尺寸的减小，锚框的面积通常会逐渐增大。这种设计是为了让不同尺度的目标能够在特征金字塔的不同层
        # 级上得到有效的检测，从而提高整体检测性能。
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i), # ceil:向上取整
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        # (total_features*n_anchors,4) n_anchors在这里是9,表示某个中心点位置的9个锚框
        # total_features是5个特征图中的空间点的数目总和(也可以称作锚框中心点的总数)
        return tf.concat(anchors, axis=0) 
def random_flip_horizontal(image, boxes): # 随机水平翻转
    # 这行代码生成一个介于0和1之间的随机数，并检查该随机数是否大于0.5。如果是，则执行水平翻转；否则，不进行翻转。
    if tf.random.uniform(()) > 0.5:
        # 如果满足翻转条件，则使用TensorFlow的 tf.image.flip_left_right 函数来水平翻转图像。
        # 该函数将图像的左右部分交换。
        image = tf.image.flip_left_right(image) 
        # 当图像水平翻转时，边界框的位置也需要相应更新。边界框通常表示为 (xmin, ymin, xmax, ymax)，
        # 表示边界框的左上和右下角坐标。
        # boxes[:, 2] 表示 xmax右边界,boxes[:, 0] 表示 xmin左边界,boxes[:, 1] 表示 ymin顶部边界,
        # boxes[:, 3] 表示 ymax底部边界
        # 当图像水平翻转时，水平坐标需要进行调整：新的 xmin 应该是 1 - xmax；
        # 新的 xmax 应该是 1 - xmin；ymin 和 ymax 不变，因为垂直坐标没有改变。
        # 因此，新的边界框坐标可以通过 tf.stack 函数重新组织，得到 [new_xmin, ymin, new_xmax, ymax]。
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes
# 调整图像的大小并添加填充以适应特定的输入要求。这个函数可以用于预处理图像数据
# image: 输入的图像张量 min_side: 图像的较短边调整后的最小长度，默认为500.0。
# max_side: 图像较长边的最大长度，以防止图像过大的情况，默认为800.0。
# jitter: 包含缩放比例的最小值和最大值的列表，用于在范围内随机选择缩放比例。
# stride: 特征金字塔中最小特征图的步长，用于确定最终图像的尺寸。
# 这样做的目的是为了保持图像的宽高比，同时让图像能够适配模型的输入要求，并且通过 jitter 参
# 数引入一些随机性，增加数据增强的效果。
def resize_and_pad_image(
    image, min_side=500.0, max_side=800.0, jitter=[400,700], stride=128.0
):
    # 获取图像的高度和宽度。
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32) # h,w
    # 如果设置了 jitter，则从给定的范围内随机选择一个 min_side。
    if jitter is not None: 
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    # 计算缩放比率 ratio，使得图像的较短边等于 min_side。
    ratio = min_side / tf.reduce_min(image_shape) 
    # 如果应用该比率后，图像的较长边超过了 max_side，则重新计算比率以保证较长边不超过 max_side。
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    # 使用计算出的比率调整图像的大小。
    image_shape = ratio * image_shape 
    # 计算填充后的图像尺寸，以确保图像尺寸能被 stride 整除。
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    # 使用 pad_to_bounding_box 方法对图像进行填充，使其达到所需的尺寸。
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio
# 用于对单个训练样本进行预处理。
def preprocess_data(sample):
    # 获取图像和边界框信息,image 是样本中的图像。
    # bbox 是样本中的边界框坐标，通过 swap_xy 函数进行坐标变换
    image = sample["image"] 
    bbox = swap_xy(sample["objects"]["bbox"]) 
    # 获取类别ID：class_id 是图像中目标的类别，类型转换为 int32。
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32) 
    # 随机水平翻转：使用 random_flip_horizontal 函数随机水平翻转图像及其对应的边界框。
    image, bbox = random_flip_horizontal(image, bbox) 
    # 调整图像大小并填充：使用 resize_and_pad_image 函数调整图像大小，并确保图像尺寸能够被特
    # 征金字塔中的步长整除
    image, image_shape, _ = resize_and_pad_image(image) 
    # 更新边界框坐标：将边界框坐标从归一化的形式转换为0像素值，并更新为 (x1, y1, x2, y2) 形式。
    # 再次转换边界框坐标格式为 (x, y, w, h)
    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],# x1,这时边界框角点坐标值转换成像素值
            bbox[:, 1] * image_shape[0], # y1
            bbox[:, 2] * image_shape[1], # x2
            bbox[:, 3] * image_shape[0], # y2
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox) # (x1,y1,x2,y2)-->(x,y,w,h)
    return image, bbox, class_id
# 由边界框和类别id组成的原始标签需要转换为训练的目标。此转换由以下步骤组成:为给定的图像尺寸生成锚框
# 将真实框分配给锚框,没有被分配任何对象的锚框，要么被分配背景类，要么根据IOU被忽略.使用锚框生成分类和回归目标
class LabelEncoder: 
    def __init__(self):
        self._anchor_box = AnchorBox() # 锚框类对象
        # 边界框方差
        self._box_variance = tf.convert_to_tensor( 
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )
    # anchor_boxes:(n,4),gt_boxes:(m,4)
    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        # 计算n个锚框和m个真实框的iou,形状(n,m),每一行表示当前锚框和匹配到的m个真实框的iou
        iou_matrix = compute_iou(anchor_boxes, gt_boxes) 
        # 聚合求与锚框最匹配的那个真实框的iou,形状变成(n,)
        max_iou = tf.reduce_max(iou_matrix, axis=1) 
        # 获取与锚框最匹配的那个真实框的索引(n,),值是真实框索引
        matched_gt_idx = tf.argmax(iou_matrix, axis=1) 
        # 获取正掩码,max_iou中值大于等于match_iou的,表示匹配程度高,(n,)
        positive_mask = tf.greater_equal(max_iou, match_iou) 
        # 小于ignore_iou的称为负掩码,掩码中True表示匹配程度低,(n,)
        negative_mask = tf.less(max_iou, ignore_iou) 
        # 在ignore_iou和match_iou值之间的会成为忽略掩码,(n,)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        # 返回每个锚框匹配到的真实框的idx索引,正掩码--->真正例,忽略掩码
        return (
            matched_gt_idx, 
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),  
        )
    # 这段代码描述的是在目标检测算法中，如何从锚框（anchor boxes）和匹配上的真实框（ground truth boxes）
    # 计算回归目标的过程。
    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        # 计算偏移量：首先计算匹配的真实框中心点相对于锚框中心点的偏移量。这通过 (matched_gt_boxes[:, :2] 
        # - anchor_boxes[:, :2]) 实现，这里 [:, :2] 表示选取了 x 和 y 坐标。
        # 归一化偏移量：接着将上述得到的偏移量除以锚框的宽度和高度进行归一化，即 / anchor_boxes
        # [:, 2:]。这里 [:, 2:] 表示选取了宽度和高度。
        # 计算比例：接下来计算真实框的宽度和高度与锚框的宽度和高度的比例，并取自然对数 tf.math.log(
        # matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:])。这一步是为了让网络更容易学习不同大
        # 小的目标，因为直接使用宽度和高度的比例可能会导致数值不稳定。
        # 拼接结果：然后将归一化的偏移量和对数比例拼接到一起，形成一个新的张量 box_target，这个张量包
        # 含了每个匹配的真实框相对于其对应锚框的归一化偏移量和对数比例。
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        # 应用方差：最后，box_target 被除以一个预先定义好的方差 _box_variance。这个方差通常是在训练过
        # 程中根据经验或者通过调整设置的，它可以帮助平衡不同回归目标的重要性，从而加速训练过程并提高模型性能。
        # 处理后的 box_target 张量会被用来作为网络中的回归分支的监督信号，帮助模型学会预测正确的目标位置。
        # box_target 被除以预先定义好的方差 _box_variance，以平衡不同回归目标的重要性。
        box_target = box_target / self._box_variance
        return box_target
    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        # 获取指定形状的图片中的所有特征图下的所有锚框,形状(total_anchors,4)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32) # (n,),类别ids
        # 匹配的真实框索引,匹配iou超过正样本阈值的,匹配iou在阈值之间的
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        # 从gt_boxes（真实框的集合）中根据matched_gt_idx（匹配到的真实框的索引）获取真实框
        # 并将这些收集到的真实框赋值给matched_gt_boxes。结果是一个形状为[n,4]的张量，
        # 表示所有匹配到的真实框,形状和anchor_boxes形状一致
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx) 
        # (n,4),计算回归目标：调用 _compute_box_target 方法来计算回归目标 (box_target)，
        # 即每个锚框对应的偏移量，这些偏移量是基于匹配的真实框相对于锚框的位置计算得出的。
        # 锚框和它匹配到的最大iou的真实框之间的偏移做目标边界框标签,形状为[n,4]
        # box_target：表示每个锚框对应的归一化偏移量和比例，形状为 (n, 4)。
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        # 提取匹配的真实类别ID,使用 matched_gt_idx 从 cls_ids 中提取匹配的真实框的类别ID
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx) # (n,)
        # 构建分类标签：
        # 使用 positive_mask 来标记正样本，非正样本的位置赋值 -1，表示背景或忽略
        cls_target = tf.where( 
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        # 使用 ignore_mask 来标记需要忽略的样本，这些位置赋值 -2；
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        # 最终得到的 cls_target 是一个包含每个锚框类别的张量，其中 -1 表示负样本（背景）
        # ，-2 表示忽略的样本。
        # 将 cls_target 的维度扩展到 (m, 1)，以便能够与 box_target 在最后一个轴上进行合并
        cls_target = tf.expand_dims(cls_target, axis=-1) # (m,1)
        # 形成最终的标签张量 label，形状为 (m, 5)，其中前四个元素代表边界框的回归目标，第五
        # 个元素代表类别的标签
        label = tf.concat([box_target, cls_target], axis=-1)
        return label
    # 这段代码描述了一个用于物体检测任务的数据预处理函数 encode_batch，它负责将一批图像及其对应的真实
    # 框（ground truth boxes）和类别ID（class IDs）转换成适合模型训练的格式。
    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        # 获取批次图像形状：使用 tf.shape() 函数获取批次图像的形状信息 images_shape，这包括批次大小
        # batch_size 以及其他维度信息（如高度 h，宽度 w 等）。
        images_shape = tf.shape(batch_images) # (b,h,w,...)
        batch_size = images_shape[0] # 批次大小
        #初始化 TensorArray：创建一个 tf.TensorArray 对象 labels，用于保存每一张图像经过 
        # _encode_sample 处理后得到的标签信息。tf.TensorArray 是一种可以动态增长的数组，在循环中非常有用。
        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        # 处理每一张图像：遍历整个批次中的每一张图像，对于每一张图像，调用 _encode_sample 方法来生成对应的标签
        # label。这包括计算回归目标以及分类标签，并将它们组合在一起。
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            # 写入 TensorArray：将每张图像的标签信息写入到 labels TensorArray 中。
            labels = labels.write(i, label) 
        # 预处理图像数据：使用 ResNet 预处理函数 tf.keras.applications.resnet.preprocess_input 对图像数据进行
        # 预处理，使其符合 ResNet 模型的输入要求。这通常涉及到对图像像素值的标准化处理。
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        # 返回处理后的数据：最后，返回经过预处理的图像数据 batch_images 以及从 TensorArray 中收集并堆叠起来的标签信息 
        # labels.stack()。labels.stack() 会将所有单个样本的标签信息组合成一个完整的批次标签。
        return batch_images, labels.stack()
# 建立ResNet50骨干网
# retanet使用基于ResNet的骨干网，利用ResNet骨干网构造特征金字塔网络。在本例中，
# 我们使用ResNet50作为主干，并在步长8,16和32处返回特征图。
def get_backbone(input_shape=[None, None, 3]):
    backbone = keras.applications.ResNet50(
        include_top=False, input_shape=input_shape
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    # 返回这三个尺寸的特征图(64, 64, 512),...
    return keras.Model( 
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )
# 构建特征图金字塔网络作为自定义层
# 在特征金字塔网络（FPN）中，不同尺度的特征图服务于不同大小的目标检测。较大的特征图（如 P3）
# 更适合捕捉较小的目标，而较小的特征图（如 P5 或 P6）更适合捕捉较大的目标。
# 较大的特征图保留了更多的细节信息，因此更适合检测图像中的小物体。
# 较小的特征图具有更高的抽象级别，更适合检测图像中的大物体。
# 在这个上下文中：
# p3_output（较高的分辨率，如 64x64）主要用于检测图像中的小物体。
# p5_output 和 p6_output（较低的分辨率，如 16x16 或更小）主要用于检测图像中的大物体。
class FeaturePyramid(keras.layers.Layer):
    # 用上面resnet返回的特征图构建特征金字塔
    def __init__(self, backbone=None, **kwargs):
        super().__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same") # 点卷积
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same") # 点卷积
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same") # 点卷积
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same") # 普通卷积 
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same") # 普通卷积 
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same") # 普通卷积 
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same") # 普通卷积,下采样
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same") # 普通卷积,下采样
        self.upsample_2x = keras.layers.UpSampling2D(2) # 上采样层
    # FPN 是用于物体检测的一种架构，旨在通过不同尺度的特征图来捕捉多尺度的物体信息。
    def call(self, images, training=False): # images:(b,h,w,c)
        # 三个下采样特征图:(64, 64, 512),(32, 32, 1024),(16, 16, 2048),尺寸不同
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        # 通道数统一：接着，通过 1x1 卷积层（conv_c3_1x1, conv_c4_1x1, conv_c5_1x1）将这三个特征图的通
        # 道数统一为 256。这样做的目的是使特征图之间能够更好地融合。
        p3_output = self.conv_c3_1x1(c3_output) # (64,64,256)
        p4_output = self.conv_c4_1x1(c4_output) # (32,32,256)
        p5_output = self.conv_c5_1x1(c5_output) # (16,16,256)
        # 自顶向下路径（Top-down pathway）：然后，从最高层的特征图 p5_output 开始，将其上采样到与下
        # 一层特征图相同的尺寸，并与之相加。这样做是为了将高层抽象的特征信息与低层更具体的细节信息相结合，从
        # 而得到更丰富的多尺度特征。
        # 这种操作叫做 自顶向下的特征融合（Top-down Feature Fusion）。在这个过程中，较高层次（较深、
        # 分辨率较低）的特征图通过上采样（upsample_2x）与较低层次（较浅、分辨率较高）的特征图相加，从
        # 而融合了较深层特征图中的抽象特征和较浅层特征图中的细节特征。
        # 这个操作的专业术语可以称为 自顶向下的特征融合 或者 特征金字塔融合
        # 这种融合方式有几个好处： 
        # 多尺度特征提取：通过融合不同层次的特征图，模型可以在多个尺度上检测物体，增强了模型的鲁棒性和泛化能力。
        # 信息传递：较深层的特征图包含了更抽象的信息，而较浅层的特征图则保留了更多细节信息。通过相加操作，可以
        # 将这些信息结合起来，使得每一层的特征都更加丰富。
        # 减少信息损失：通过逐层融合特征，可以减少信息的丢失，特别是在从深层到浅层的过程中，保持特征的完整性。
        p4_output = p4_output + self.upsample_2x(p5_output) # (32,32,256) 
        p3_output = p3_output + self.upsample_2x(p4_output) # (64,64,256)
        # 额外的卷积操作：在完成特征融合之后，再次对 p3_output, p4_output, p5_output 进行 3x3 卷积操作（
        # conv_c3_3x3, conv_c4_3x3, conv_c5_3x3），进一步提取特征。
        p3_output = self.conv_c3_3x3(p3_output) # (...,256)
        p4_output = self.conv_c4_3x3(p4_output) # (...,256)
        p5_output = self.conv_c5_3x3(p5_output) # (...,256)
        # 额外的金字塔层次：此外，还增加了两个额外的层次 p6_output 和 p7_output。p6_output 是通过对 
        # c5_output 进行 3x3 卷积操作得到的，而 p7_output 则是通过对 p6_output 应用 ReLU 激活函数后
        # 再进行 3x3 卷积操作得到的。这两个额外层次进一步增加了特征图的多样性，有助于捕捉更大的检测目标。
        p6_output = self.conv_c6_3x3(c5_output) # (8,8,256) 尺寸会减半
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output)) # (4,4,256) 尺寸减半
        # 返回5种不同尺寸的特征图,这个对应锚框类中的5种不同的步长,也对应5种锚框面积
        # 较小的锚框(意味着面积较小)用于捕捉较浅层中的特征图中的较小的物体,因为这个时候细节多
        # 而较大的锚框(意味着面积较大)用于捕捉较深层的特征图中较大的物体
        # 对于512x512的输入图片,返回(64,64,256),(32,32,256),(16,16,256)
        # (8,8,256),(4,4,256)
        return p3_output, p4_output, p5_output, p6_output, p7_output
# 构建分类和边界框回归头
# RetinaNet模型具有用于边界框回归和预测对象类别概率的独立头。这些头在不同尺寸的特征图
# 之间是共享的。
# 这个预测头主要用于两个目的：
# 类别预测（Classification Head）：如果 output_filters 设置为类别数加上背景类（例如
# ，对于 COCO 数据集，类别数为 80，则 output_filters 可能是 81）。
# 边界框回归（Box Regression Head）：如果 output_filters 设置为边界框的参数数量（
# 通常是 4，即 x, y, width, height）。
# 通过这个预测头，模型可以从提取的特征图中预测出每个位置上的类别概率分布和边界框的偏移量
# ，从而实现物体检测的任务。
# 预测头的作用是在提取的特征图上进行最终的预测，包括类别预测和边界框回归。具体来说，这段代码构
# 建了一个包含多个卷积层的序列模型，用于生成最终的预测输出
def build_head(output_filters, bias_init,shape=[None, None, 256]):
    # 构建类/框预测头。
    head = keras.Sequential([keras.Input(shape=shape)])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01) # 随机标准正太初始化
    for _ in range(4): # 添加4次卷积块
        head.add(
            keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(keras.layers.ReLU()) # 在每个卷积之后只加了激活函数,并没有加批次标准化
    head.add(
        keras.layers.Conv2D(
            output_filters, # 输出通道数由参数output_filters指定,这取决于要预测的类别数量或边界框的数量。
            3, # 核大小
            1, # 步长
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head # (b,h,w,output_filters)
# 继承keras.Model构建RetinaNet
class RetinaNet(keras.Model):
    # num_classes:模型要识别的目标类别数量,backbone:作为特征提取基础的主干网络
    def __init__(self, num_classes, backbone=None, **kwargs):
        super().__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone) 
        self.num_classes = num_classes 
        # 分类头 (cls_head) 和 回归头 (box_head): 这两个组件分别负责预测物体的类别和边界框的位置。
        # 分类头使用了一个初始化器来初始化权重，这个初始化器基于先验概率，假设正类的概率为0.01，负类的概
        # 率为0.99。这样的初始化有助于在训练初期应对类别不平衡的问题。回归头使用了零初始化。
        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        # 类别头,bias_init使用先验概率初始化,9 * num_classes表示每个锚框中心点对应9个
        # 锚框,num_classes 是类别数量,而这个输出通道表示一个中心点(锚点)对应的所有预测
        self.cls_head = build_head(9 * num_classes, prior_probability) 
        # 边界框头,bias_init使用0向量初始化,9*4表示一个中心点(锚点)对应的所有预测
        self.box_head = build_head(9 * 4, "zeros") 
    def call(self, image, training=False):
        # 获取输入的批次数据对应的不同尺寸的特征图
        features = self.fpn(image, training=training) 
        N = tf.shape(image)[0] # 批次大小
        cls_outputs = []
        box_outputs = []
        for feature in features:
            # self.box_head(feature)输出的形状是(b,h,w,9 * 4)
            # box_outputs添加的是变形后的,形状(N,h*w*9,4)
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            # self.cls_head(feature)输出的形状是(b,h,w,9 * num_classes)
            # -->(b,h*w×9,num_classes)
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        # 在索引1的轴合并,形状为(N,total_anchors,num_classes)
        cls_outputs = tf.concat(cls_outputs, axis=1)
        # 在索引1的轴合并,形状为(N,total_anchors,4)
        box_outputs = tf.concat(box_outputs, axis=1)
        # 之后在最后一个轴合并,形状变成(N,total_anchors,4+num_classes)
        return tf.concat([box_outputs, cls_outputs], axis=-1)
# 实现一个用于解码预测的自定义层
# 在深度学习模型中，特别是处理如目标检测、语义分割等任务时，模型的最终输出往往需要经过一个解码过程
# 来转换为人类可理解的形式（如边界框坐标、类别标签等）。
# 能够有效地筛选出高质量的检测框，并去除那些冗余或重叠的框，最终返回一组最优的边界框和类别预测。
# 当模型预测出多个边界框及其对应的类别概率时，只有当某个边界框的类别概率高于 confidence_threshold
# 时，才会被保留下来进入下一步处理（如非极大值抑制）
class DecodePredictions(tf.keras.layers.Layer):
    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )
    # anchor_boxes:,box_predictions:(N,total_anchors,4)
    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        # 在训练过程中，模型预测的边界框调整参数通常被缩放了特定的方差值，这一行代码恢复了这些调整参
        # 数至其原始尺度。(b,total_anchors,4)
        boxes = box_predictions * self._box_variance 
        # 代码通过调整锚点框来生成最终的边界框。这里使用了两种调整方式：
        # 对于边界框的中心点（boxes[:, :, :2]），通过将预测的偏移量乘以锚点框的宽度和高度（
        # anchor_boxes[:, :, 2:]），然后将结果加到锚点框的中心点坐标（anchor_boxes[:, :, :2]）
        # 上，来计算新的中心点坐标。
        # 对于边界框的宽度和高度（boxes[:, :, 2:]），首先对这些预测值应用指数函数（tf.math.exp），这是
        # 因为在训练时通常会使用对数尺度来预测这些值，以便处理它们可能具有的广泛范围。然后，将结果乘以锚点
        # 框的宽度和高度，以得到最终的宽度和高度。
        # 这里首先处理边界框的中心点坐标 (boxes[:, :, :2])，将其与锚框的宽度和高度 (anchor_boxes[:, :, 2:]) 相乘后
        # ，加上锚框的中心点坐标 (anchor_boxes[:, :, :2])，从而获得新的中心点坐标。
        # 接着处理边界框的宽度和高度 (boxes[:, :, 2:])，通过对其应用指数函数 tf.math.exp，这是因为宽度和高度在训练时是
        # 以对数形式预测的，以处理它们可能的广泛范围。之后，将指数化后的结果乘以锚框的宽度和高度，以得到最终的宽度和高度。
        # 使用 tf.concat 将调整后的中心点坐标和宽度高度合并，形成最终的边界框表示。
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        # 将边界框从中心点加宽度高度的形式转换为角点形式，即 (x1, y1, x2, y2)，这通常是目标检测中常见的边
        # 界框表示方式。(b,total_anchors,4)
        boxes_transformed = convert_to_corners(boxes) #转换成角点形式：(x1,y1,x2,y2)
        return boxes_transformed
    # images:(1,h,w,3),predictions:(b,total_anchors,4+num_classes)
    def call(self, images, predictions):
        # 这一行获取输入图像的形状，并将其转换为浮点类型
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        # 根据输入图片的h和w获取5种不同尺寸下的特征图对应的所有锚框,(total_anchors,4)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        # 提取模型输出中关于边界框调整参数的部分，即预测的边界框偏移量。
        box_predictions = predictions[:, :, :4] 
        # 对模型输出中的类别预测部分应用 Sigmoid 函数，将输出转换为概率值，因为类别预测通常是二值或多
        # 值逻辑回归问题。(b,total_anchors,num_classes)
        # cls_predictions 包含了每个锚框针对每个类别的预测概率。
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:]) 
        # 调用 _decode_box_predictions 方法来解码边界框预测。anchor_boxes[None, ...] 
        # 增加了一个维度来匹配 box_predictions 的形状。(b,total_anchors,4)
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)
        # 使用tf.image.combined_non_max_suppression函数对解码后的边界框和类别预测进行非极大值抑制处理。
        # 这个函数会保留高置信度的边界框，同时抑制与它们重叠度（IoU）超过阈值的低置信度边界框。参数包括：
        # boxes：边界框坐标，需要增加一个额外的维度以匹配函数的要求
        # cls_predictions：类别预测
        # max_detections_per_class：每个类别的最大检测数。
        # max_detections：图像中的最大总检测数。
        # nms_iou_threshold,非极大值抑制的IoU阈值
        # confidence_threshold：置信度阈值，低于此阈值的边界框将被忽略。
        # clip_boxes：是否将边界框裁剪到图像边界内，这里设置为False。
        # batch_size：批量处理的图像数量。total_anchors：所有特征图中的锚框总数。
        # num_classes：类别数量。
        # 这意味着对于每一个锚框，都有一个长度为 num_classes 的向量，表示该锚框属于各个类别的概率。这个
        # 向量是经过 Sigmoid 函数处理后的结果，因此每个元素的值都在 [0, 1] 之间，可以解释为该锚框属于
        # 相应类别的概率。
        return tf.image.combined_non_max_suppression( # 非极大值抑制（NMS）
            tf.expand_dims(boxes, axis=2), # (b,total_anchors,1,4)
            cls_predictions, # (b,total_anchors,num_classes)
            # 控制每个类别最多保留的边界框数量，确保不会有过量的检测框。
            self.max_detections_per_class,
            # 控制总的检测框数量，即使每个类别都达到最大保留数量，总的检测框数量也不会超过这个限制。
            self.max_detections,
            # 设定的 IoU 阈值用于判断边界框之间的重叠程度，从而决定哪些框需要被抑制。
            self.nms_iou_threshold,
            # 用于筛选出足够高的得分边界框，只对那些得分较高的框进行 NMS。
            self.confidence_threshold,
            # 决定是否将边界框裁剪到图像范围内，这里设置为 False 表示不裁剪。
            clip_boxes=False,
        )
# 平滑 L1 损失（Smooth L1 Loss），这是一种常用于目标检测任务中的边界框回归损失函数
class RetinaNetBoxLoss(tf.losses.Loss):
    # 初始化基类 tf.losses.Loss，设置 reduction 参数为 "none"，这意味着损失函数不
    # 会对批处理中的样本进行聚合，而是返回每个样本的损失值。
    def __init__(self, delta):
        super().__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        # 设置平滑 L1 损失中的 delta 参数，该参数用于区分损失函数在不同误差区间内的计算方式。
        self._delta = delta
    def call(self, y_true, y_pred):
        # 计算真实值y_true和预测值y_pred之间的差异
        difference = y_true - y_pred 
        # 计算这个差异的绝对值，这实际上与MAE的计算方式相似
        absolute_difference = tf.abs(difference) 
        # 计算差异的平方，这是MSE的计算方式。
        squared_difference = difference ** 2 
        #使用 tf.where 函数根据 absolute_difference 与 self._delta 的比较结果来选择损失的计
        # 算方式：如果 absolute_difference < self._delta，则损失为 0.5 * squared_difference
        # （接近 MSE 的形式，但乘以 0.5 以调整量纲）。否则，损失为 absolute_difference - 0.5（
        # 接近 MAE 的形式，但在误差较大时提供线性增长）。
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        # 沿着最后一个维度对损失进行求和，得到每个样本的总损失值。结果的形状为 
        # (n,)，其中 n 是批处理中的样本数量。
        return tf.reduce_sum(loss, axis=-1) 
# 它实现了焦点损失（Focal Loss）。Focal Loss 是一种改进的交叉熵损失函数，专门设计用于解决类别
# 不平衡问题，尤其是在前景（正样本）和背景（负样本）数量差异极大的情况下。
class RetinaNetClassificationLoss(tf.losses.Loss):
    # 初始化基类 tf.losses.Loss，设置 reduction 参数为 "none"，这意味着损失函数不会对批处理中
    # 的样本进行聚合，而是返回每个样本的损失值。
    def __init__(self, alpha, gamma):
        super().__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        # 设置用于调节正负样本权重的系数 alpha。
        # 如果你的 alpha 是一个标量，那么 self._alpha 将是一个标量值，用于调整正负样本的权重。
        # 在这种情况下，alpha 的值通常介于 0 和 1 之间，用来平衡正样本和负样本的损失贡献。
        # 如果你的 alpha 是一个向量，那么 self._alpha 将是一个向量，其中每个元素对应一个类别。
        # 在这种情况下，alpha 向量的长度应该与类别数量 num_classes 相同。
        # 这里 alpha 是作为构造函数的一个参数传入的，并且被赋值给了 self._alpha。既然 alpha 
        # 是作为一个单独的参数传递的，而不是一个列表或数组，我们可以认为它是一个标量。
        self._alpha = alpha 
        # 设置用于调节易分类样本损失下降速率的系数 gamma
        self._gamma = gamma 
        # 调节因子，用于调节易分类样本损失下降的速率。当 gamma=0 时，Focal Loss 退化为
        # 标准的交叉熵损失
    # y_true:(batch_size, total_anchors, num_classes)
    # 每个锚框对应一个长度为 num_classes 的 one-hot 向量，表示该锚框的真实类别。
    # y_pred 是模型预测的原始输出（logits），通常表示为一个张量，其中每个元素表示一个锚框对于
    # 每个类别的预测分数
    # 同样假设 num_classes 是类别数量，total_anchors 是所有特征图中的锚框总数，batch_size 是
    # 批量处理的图像数量，那么 y_pred 的形状应该是：(batch_size, total_anchors, num_classes)
    # 每个锚框对应一个长度为 num_classes 的向量，表示该锚框对于每个类别的预测分数（logits）。
    # 对于每个锚框，都有一个长度为 num_classes 的向量，表示该锚框的真实类别标签（对于 y_true）
    # 或预测分数（对于 y_pred）。
    def call(self, y_true, y_pred):
        # 计算真实标签 y_true 与预测的 logit 值 y_pred 之间的二元交叉熵损失。
        # 这里的 y_pred 是未经激活的原始输出，而 labels 是真实的标签（通常是 0 或 1）
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        # 计算预测概率 probs，即对 y_pred 应用 Sigmoid 函数得到的结果。probs 
        # 表示预测为正类的概率。
        probs = tf.nn.sigmoid(y_pred)
        # 根据 y_true 的值为每个样本分配不同的权重。如果 y_true 为 1.0（正样本），
        # 则权重为 self._alpha；否则为 1.0 - self._alpha。这一步是为了调整正负样本的贡献度。
        # alpha 被用于动态调整正样本和负样本的权重
        # 由于 self._alpha 被用作一个标量来调整正负样本的权重，我们可以确认它是一个标量值。
        # self._alpha 是一个标量，通常介于 0 和 1 之间。
        # 它用来调整正样本和负样本的权重，通常用于解决类别不平衡的问题
        # self._alpha 是一个标量。
        # alpha 的形状与 y_true 的形状相同，即 (batch_size, total_anchors, num_classes)。
        # alpha 的值根据 y_true 的值动态选择 self._alpha 或 1.0 - self._alpha。
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        # 计算pt,y_true中等于1.0的会是probs中对应的值,否则是1-probs中对应的值
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        # 根据 Focal Loss 的公式计算每个样本的损失。损失由类别权重 alpha、焦点调节
        # 项 (1.0 - pt)**self._gamma 和二元交叉熵 cross_entropy 的乘积组成。
        # alpha和y_true形状相同
        # _gamma (self._gamma)：是一个标量，用于调节易分类样本损失下降的速率。当 gamma 设置为 0 
        # 时，Focal Loss 退化为标准的交叉熵损失。
        # gamma (self._gamma) 是一个标量，用于调节易分类样本损失下降的速率。
        # pt (pt) 的形状与 y_true 的形状相同。
        # cross_entropy (cross_entropy) 的形状与 y_true 的形状相同。
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
# RetinaNetLoss 类实现了 RetinaNet 的综合损失函数，它结合了分类损失（Focal Loss）
# 和边界框回归损失（Smooth L1 Loss）。
class RetinaNetLoss(tf.losses.Loss):
    # num_classes：类别数量，默认为 80。alpha：Focal Loss 中调整正负样本权重的系数，
    # 默认为 0.25。gamma：Focal Loss 中的调节因子，默认为 2.0。
    # delta：Smooth L1 Loss 中的调节因子，默认为 1.0。
    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        super().__init__(reduction="auto", name="RetinaNetLoss")
        # 焦点损失
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma) 
        self._box_loss = RetinaNetBoxLoss(delta)  # 平滑 L1 损失
        self._num_classes = num_classes # 类别
    def call(self, y_true, y_pred):
        # y_true 的形状为 (batch_size, total_anchors, 5)。
        # 前 4 列表示边界框的真实偏移（如 (dx, dy, dw, dh)）
        # 第 5 列表示类别标签，通常为整数
        # y_pred 的形状为 (batch_size, total_anchors, 4 + num_classes)。
        # 前 4 列表示预测的边界框偏移值。后面的列表示预测的类别概率。
        # 转换类型
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        # 提取边界框真实的偏移标签(batch_size, total_anchors, 4)
        box_labels = y_true[:, :, :4] 
        # 模型预测的边界框偏移值(batch_size, total_anchors, 4)
        box_predictions = y_pred[:, :, :4]
        # 对真实类别标签的 one-hot 编码
        cls_labels = tf.one_hot( 
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes, #目标类别总数
            dtype=tf.float32,
        )
        # 模型中每个样本的所有锚框预测类别的概率分布
        cls_predictions = y_pred[:, :, 4:] 
        # 目标类别属于正类的掩码,大于-1的
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        # 忽略类别掩码,y_true中等于-2的
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        # 计算分类损失和边界框损失
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions) 
        # 忽略分类损失中tf.equal(ignore_mask, 1.0)为True的位置(b, total_anchors)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss) # 判断要不要忽略损失
        # 忽略iou小于0.5的(b, total_anchors)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0) 
        # 计算每个样本中正样本（IoU > 0.5 的锚框）的数量，形状为 (batch_size,)
        # positive_mask 是一个布尔张量，形状为 (batch_size, total_anchors)，表示哪些锚
        # 框是正样本（即与真实边界框的 IoU 大于某个阈值，通常是 0.5）。
        # 这里，axis=-1 表示沿着最后一个维度（即锚框的数量维度）进行求和，得到的结果是一个
        # 形状为 (batch_size,) 的张量，表示每个样本中正样本的数量。
        normalizer = tf.reduce_sum(positive_mask, axis=-1) 
        # 这里 clf_loss 的原始形状为 (batch_size, total_anchors)。通过 tf.reduce_sum(..., axis=-1)，
        # 我们沿着最后一个轴（即 total_anchors 轴）进行求和，得到的结果是一个形状为 (batch_size,) 
        # 的张量，表示每个样本的总分类损失
        # tf.math.divide_no_nan 用于安全地进行除法操作，即使分母为零也不会导致错误，而是返回 NaN（Not a Number
        # clf_loss 和 box_loss 的形状均为 (batch_size,)。
        # 这意味着每个样本的分类损失和边界框损失都被归一化到了同一个尺度上，方便后续的损失计算和优化。
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss

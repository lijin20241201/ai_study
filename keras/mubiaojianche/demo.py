# 假设的图像和检测结果
image = np.random.randint(0, 256, size=(300, 400, 3), dtype=np.uint8)
boxes = np.array([[100, 100, 200, 200], [150, 150, 250, 250]])
classes = ['person', 'car']
scores = [0.9, 0.8]
# 调用可视化函数
ax = visualize_detections(image, boxes, classes, scores)
# 示例图像（假设为 256x256）
image = tf.ones((256, 256, 3))
# 示例边界框坐标
boxes = tf.constant([
    [0.1, 0.2, 0.3, 0.4],  # (xmin, ymin, xmax, ymax)
    [0.5, 0.6, 0.7, 0.8],
])
# 调用函数
image_flipped, boxes_flipped = random_flip_horizontal(image, boxes)
# 输出结果
print("Flipped Image Shape:", image_flipped.shape)
print("Updated Boxes Coordinates:", boxes_flipped.numpy())

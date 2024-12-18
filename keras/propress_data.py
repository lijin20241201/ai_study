class Augment(layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.randomFlip_inputs = layers.RandomFlip(mode="horizontal", seed=seed)
        self.randomFlip_labels =layers.RandomFlip(mode="horizontal", seed=seed)
    def call(self, inputs, labels):
        x =self.randomFlip_inputs(inputs)
        y = self.randomFlip_labels(labels)
        return x, y
def load_paths(path, split_ratio):# 加载图片掩码路径,切分路径
    images = sorted(glob(os.path.join(path, "DUTS-TE-Image/*"))) # 排序后的
    masks = sorted(glob(os.path.join(path, "DUTS-TE-Mask/*")))
    len_ = int(len(images) * split_ratio)
    return (images[:len_], masks[:len_]), (images[len_:], masks[len_:])#返回训练集,验证集
def read_image(path, size,mode):
    x = keras.utils.load_img(path, target_size=size, color_mode=mode)
    x = keras.utils.img_to_array(x)
    if mode=='rgb': # 图片数据模型内部会归一处理,这里不用变
        x=x.astype(np.float32)
    elif mode=='grayscale': # 掩码数据转成0--1区间
        x = (x / 255.0).astype(np.float32)
    return x
def preprocess(x_batch, y_batch, img_size, out_classes):
    def f(_x, _y):
        _x, _y = _x.decode(), _y.decode()
        _x = read_image(_x,(img_size, img_size), mode="rgb")  # image
        _y = read_image(_y,(img_size, img_size), mode="grayscale")  # mask
        return _x, _y
    images, masks = tf.numpy_function(f, [x_batch, y_batch], [tf.float32, tf.float32])
    images.set_shape([img_size, img_size, 3])
    masks.set_shape([img_size, img_size,out_classes])
    return images, masks
def load_dataset(image_paths, mask_paths, img_size, out_classes, batch,is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(
        lambda x, y: preprocess(x, y, img_size, out_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if is_train:
        dataset=dataset.map(Augment())
        dataset = dataset.cache().shuffle(buffer_size=100)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
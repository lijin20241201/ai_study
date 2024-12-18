# 骨干网络的基础类。
# 骨干网络是在标准任务上训练的可重用模型层，例如在Imagenet分类任务上训练的模型层，它们可以在其他任务中复用。
# 用户可以通过指定的模块路径 keras_cv.models.Backbone 来访问该类。这意味着当用户安装了 Keras CV 库后，
# 就可以通过这个路径来导入并使用 Backbone 类。
@keras_cv_export("keras_cv.models.Backbone")
class Backbone(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # 调用父类的初始化方法
        self._pyramid_level_inputs = {} # 金字塔层级的输入
        # 用于存储当前模型中所有层的唯一标识符（ID）。这些 ID 将用于过滤掉那些不应该出现在 __dir__ 方法
        # 返回结果中的层属性。
        self._functional_layer_ids = set(
            id(layer) for layer in self._flatten_layers()
        )
    # __dir__ 方法覆盖了默认的行为，用于过滤掉那些由函数式 API 自动创建的内部层对象，使得动态属性列表更清晰、
    # 更简洁。这有助于在交互式环境中（如 Python shell 或 Jupyter Notebook）查看类的可用属性时，避免列出那
    # 些内部实现细节。
    def __dir__(self):
        def filter_fn(attr):
            try:
                return id(getattr(self, attr)) not in self._functional_layer_ids
            except:
                return True

        return filter(filter_fn, super().__dir__())
    # get_config 方法返回一个配置字典，包含了当前实例的一些关键属性
    # 这里没有调用父类的 get_config 方法，因为默认的功能性模型的 get_config 方法返回的是嵌套结构，
    # 不适合传递给 Backbone 构造函数。
    def get_config(self):
        # 在功能型模型中，get_config 方法通常会返回一个包含模型结构的信息的嵌套字典。对于 Backbone 类来说，这样的配
        # 置可能过于复杂，无法直接用于重新构造一个新的 Backbone 实例。因此，Backbone 类需要覆盖 get_config 方法，
        # 以便返回一个简化后的配置，这个配置可以直接传递给 Backbone 的构造函数。
        return {
            "name": self.name,
            "trainable": self.trainable,
        }
    # from_config 是一个类方法，用于从配置字典中重建一个 Backbone 实例。默认的功能性模型的 from_config 方法会返回一
    # 个普通的 keras.Model 实例，而这里我们覆盖它以确保返回的是 Backbone 的子类实例。
    @classmethod
    def from_config(cls, config):
        # 功能型模型的默认 `from_config()` 方法会返回一个普通的 `keras.Model` 实例。
        # 我们覆盖它是为了得到一个子类的实例。
        return cls(**config)
    # presets 类属性,
    @classproperty
    def presets(cls):
        return {}
    # 这个字典用于存储带有权重的预设配置，即那些已经经过预训练并带有权重的模型配置
    @classproperty
    def presets_with_weights(cls):
        return {}
    # 这个字典通过从 presets 中排除 presets_with_weights 中的项来生成，即所有不在 presets_with_weights 中的预设配置。
    # 示例：如果 presets 包含 "base"、"large"、"pretrained_base"、"pretrained_large"，而 presets_with_weights 包含 
    # "pretrained_base"、"pretrained_large"，那么 presets_without_weights 将只包含 "base" 和 "large"。
    @classproperty
    def presets_without_weights(cls):
        return {
            preset: cls.presets[preset]
            for preset in set(cls.presets) - set(cls.presets_with_weights)
        }
    # preset：预设配置的名称，用于标识一个特定的模型配置。load_weights：可选参数，指定是否加载权重。
    # 默认为 None，表示按照预设配置决定是否加载权重。
    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=None,
        **kwargs,
    ):
        # 如果提供的 preset 名称存在于 cls.presets 字典中，则从中提取 "kaggle_handle" 
        # 字段的值作为新的 preset 名称。
        if preset in cls.presets:
            preset = cls.presets[preset]["kaggle_handle"]
        # 检查预设配置,调用 check_preset_class 函数，检查 preset 是否适用于当前类 cls。
        check_preset_class(preset, cls)
        # 调用 load_from_preset 函数，传入 preset 名称、是否加载权重的标志以及任何需要覆盖的配置参数。
        # config_overrides 参数将传递 **kwargs 中的所有关键字参数，用于覆盖预设配置中的某些设置。
        return load_from_preset(
            preset,
            load_weights=load_weights,
            config_overrides=kwargs,
        )

    def __init_subclass__(cls, **kwargs):
        # 当一个子类被创建时，首先调用基类的 __init_subclass__ 方法，这样可以确保基类的初始化逻辑被执行。
        super().__init_subclass__(**kwargs)
        # 检查子类的字典 __dict__ 是否包含 from_preset 键，如果没有，则定义一个默认的 from_preset 方法。
        if "from_preset" not in cls.__dict__:
            # 这个默认的 from_preset 方法会调用父类的 from_preset 方法，并传递相同的参数。
            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)
            # 将定义好的 from_preset 方法作为类方法（classmethod）赋值给子类的 from_preset 属性
            # 这里定义了一个名为 from_preset 的函数，并将其转换为类方法后赋值给 cls.from_preset。
            cls.from_preset = classmethod(from_preset)
        # 如果子类的 presets 字典为空（即没有预设配置），则为 from_preset 方法设置一个默认的文档字符串，指示
        # 该类没有可用的预设配置。
        if not cls.presets:
            cls.from_preset.__func__.__doc__ = """Not implemented.

            No presets available for this class.
            """
        # 这里检查 from_preset 方法的文档字符串是否为空。如果是空的，说明子类没有覆盖或重写这个方法的文档字符串。
        if cls.from_preset.__doc__ is None:
            # 如果文档字符串为空，则从基类 Backbone 中复制 from_preset 方法的文档字符串到子类的 from_preset 方法中。
            # 这一步是为了保证至少有一个默认的文档字符串，即使子类没有显式定义。
            cls.from_preset.__func__.__doc__ = Backbone.from_preset.__doc__
            # 使用 format_docstring 函数来格式化 from_preset 方法的文档字符串。
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets_with_weights), ""),
                preset_names='", "'.join(cls.presets),
                preset_with_weights_names='", "'.join(cls.presets_with_weights),
            )(cls.from_preset.__func__)

    @property
    def pyramid_level_inputs(self):
        # 中间模型输出用于特征提取。
        # 格式是一个字典，其键为字符串，值为层名。
        # 字符串键代表特征输出的层级。一个典型的特征金字塔有五个层级，对应于骨干网络中的尺度 "P3"、"P4"、"P5"、
        # "P6"、"P7"。尺度 Pn 代表比输入图像宽度和高度小 2^n 倍的特征图。
        # 示例：
        # {
        #     'P3': 'v2_stack_1_block4_out',
        #     'P4': 'v2_stack_2_block6_out',
        #     'P5': 'v2_stack_3_block3_out',
        # }
        return self._pyramid_level_inputs
    @pyramid_level_inputs.setter
    def pyramid_level_inputs(self, value):
        self._pyramid_level_inputs = value
# vgg卷积池化块(特征提取)
def apply_vgg_block(
    x,
    num_layers,
    filters,
    kernel_size,
    activation,
    padding,
    max_pool,
    name,
):
    # 卷积的层数,卷积中带设置激活函数
    for num in range(1, num_layers + 1):
        x = layers.Conv2D(
            filters,
            kernel_size,
            activation=activation,
            padding=padding,
            name=f"{name}_conv{num}",
        )(x)
    # 如果设置了max_pool,那就应用最大池化(步长是2,下采样)
    if max_pool: 
        x = layers.MaxPooling2D((2, 2), (2, 2), name=f"{name}_pool")(x)
    return x

@keras_cv_export("keras_cv.models.VGG16Backbone")
class VGG16Backbone(Backbone):
    def __init__(
        self,
        include_rescaling,
        include_top,
        input_tensor=None,
        num_classes=None,
        input_shape=(224, 224, 3),
        pooling=None,
        classifier_activation="softmax",
        name="VGG16",
        **kwargs,
    ):
        # 检验输入
        # 如果include_top是True,表示使用预训练模型的分类头,这时就要指定num_classes
        if include_top and num_classes is None:
            raise ValueError(
                "If `include_top` is True, you should specify `num_classes`. "
                f"Received: num_classes={num_classes}"
            )
        # 如果指定了带分类头,就不应该指定pooling
        if include_top and pooling:
            raise ValueError(
                f"`pooling` must be `None` when `include_top=True`."
                f"Received pooling={pooling} and include_top={include_top}. "
            )
        # 获取模型输入
        img_input = utils.parse_model_inputs(input_shape, input_tensor)
        x = img_input # (224,224,3)
        # 归一化
        if include_rescaling:
            x = layers.Rescaling(scale=1 / 255.0)(x)
        # 经过第一个卷积池化块的处理,同时下采样
        x = apply_vgg_block( # (112,112,3)
            x=x,
            num_layers=2,
            filters=64,
            kernel_size=(3, 3),
            activation="relu", 
            padding="same", # 填充
            max_pool=True,
            name="block1",
        )
        
        x = apply_vgg_block( # (56,56,3)
            x=x,
            num_layers=2,
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            max_pool=True,
            name="block2",
        )

        x = apply_vgg_block( # (28,28,3)
            x=x,
            num_layers=3,
            filters=256,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            max_pool=True,
            name="block3",
        )

        x = apply_vgg_block( # (14,14,3)
            x=x,
            num_layers=3,
            filters=512,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            max_pool=True,
            name="block4",
        )

        x = apply_vgg_block( # (7,7,3)
            x=x,
            num_layers=3,
            filters=512,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            max_pool=True,
            name="block5",
        )
        # 如果包括分类头
        if include_top:
            x = layers.Flatten(name="flatten")(x) # 扁平化,变成向量(b,d)
            # 投影,浓缩特征
            x = layers.Dense(4096, activation="relu", name="fc1")(x) 
            x = layers.Dense(4096, activation="relu", name="fc2")(x)
            # 分类,softmax归一化
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions",
            )(x)
        # 如果不包括分类头,看传入的pooling
        else:
            if pooling == "avg": # 如果是avg,就是全局平均池化
                x = layers.GlobalAveragePooling2D()(x)
            elif pooling == "max":# 如果是max,就是全局最大池化
                x = layers.GlobalMaxPooling2D()(x)
        # 调用父类的初始化方法
        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)
        # 设置这些属性为实例属性
        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.num_classes = num_classes
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.classifier_activation = classifier_activation
    # 获取配置
    def get_config(self):
        return {
            "include_rescaling": self.include_rescaling, # 是否包含归一化
            "include_top": self.include_top, # 是否包含分类头
            "name": self.name,
            "input_shape": self.input_shape[1:], # (h,w,c)
            "input_tensor": self.input_tensor, 
            "pooling": self.pooling, # 传入的池化方式
            "num_classes": self.num_classes, # 几分类
            # 分类器激活函数
            "classifier_activation": self.classifier_activation,
            #当一个模型或其内部的层的 trainable 属性设置为 False 时，意味着这些层的权重在训练过程中不会被更新
            # 。这意味着这些层是“冻结的”，它们的参数在训练过程中保持不变。
            "trainable": self.trainable,  
        }

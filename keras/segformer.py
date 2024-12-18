from keras_cv.src.utils.preset_utils import load_from_preset

class Backbone(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pyramid_level_inputs = {}
        self._functional_layer_ids = set(
            id(layer) for layer in self._flatten_layers()
        )
    # __dir__ 方法实现了一个过滤机制，用于在列出对象的属性时排除掉那些其ID在 
    # self._functional_layer_ids 集合中的属性。
    def __dir__(self):
        def filter_fn(attr):
            try:
                return id(getattr(self, attr)) not in self._functional_layer_ids
            except:
                return True
        return filter(filter_fn, super().__dir__())
    # 获取配置
    def get_config(self):
        return {
            "name": self.name,
            "trainable": self.trainable,
        }
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    @classproperty
    def presets(cls):
        return {}

    @classproperty
    def presets_with_weights(cls):
        return {}

    @classproperty
    def presets_without_weights(cls):
        return {
            preset: cls.presets[preset]
            for preset in set(cls.presets) - set(cls.presets_with_weights)
        }
    # @classmethod 装饰器用于将一个方法定义为类方法。类方法接收类本身作为第一个参数（通常是 cls），而不是
    # 类的实例。这使得类方法能够访问类属性（即定义在类级别而不是实例级别的变量）和类本身的其他方法，而无需
    # 创建类的实例。
    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=None,
        **kwargs,
    ):
        # We support short IDs for official presets, e.g. `"bert_base_en"`.
        # Map these to a Kaggle Models handle.
        if preset in cls.presets:
            preset = cls.presets[preset]["kaggle_handle"]

        check_preset_class(preset, cls)
        return load_from_preset(
            preset,
            load_weights=load_weights,
            config_overrides=kwargs,
        )
    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to set up a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)
        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have a distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        if not cls.presets:
            cls.from_preset.__func__.__doc__ = """Not implemented.

            No presets available for this class.
            """

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = Backbone.from_preset.__doc__
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets_with_weights), ""),
                preset_names='", "'.join(cls.presets),
                preset_with_weights_names='", "'.join(cls.presets_with_weights),
            )(cls.from_preset.__func__)
    # @property装饰器将其变成了一个特殊的实例方法，这个方法被用作属性的访问器
    # 当通过实例访问这个被@property装饰的方法时，你实际上是在访问一个属性，而不是直接调用
    # 方法。这种机制提供了一种更简洁、更直观的方式来访问和修改对象的属性，同时可以在访问或修改
    # 属性时执行一些额外的逻辑。
    @property
    def pyramid_level_inputs(self):
        return self._pyramid_level_inputs
    # @pyramid_level_inputs.setter 是一个特殊的装饰器，它与 @property 装饰器一起使用，用于定义设
    # 置属性值的方法。当你使用 @property 装饰了一个方法之后，该方法就被当作了一个只读属性。但是，如果你
    # 还想允许外部代码设置这个属性的值，你就需要使用 @property.setter 装饰器来定义一个设置器（setter）。
    @pyramid_level_inputs.setter
    def pyramid_level_inputs(self, value):
        self._pyramid_level_inputs = value
class OverlappingPatchingAndEmbedding(keras.layers.Layer):
    def __init__(self, project_dim=32, patch_size=7, stride=4, **kwargs):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.patch_size = patch_size
        self.stride = stride
        self.proj = keras.layers.Conv2D(
            filters=project_dim,
            kernel_size=patch_size,
            strides=stride,
            padding="same",
        )
        self.norm = keras.layers.LayerNormalization()
    def call(self, x):
        x = self.proj(x) # 下采样
        # B, H, W, C
        shape = x.shape
        x = ops.reshape(x, (-1, shape[1] * shape[2], shape[3])) # (b,hw,c)
        x = self.norm(x)
        return x
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "patch_size": self.patch_size,
                "stride": self.stride,
            }
        )
        return config
class SegFormerMultiheadAttention(keras.layers.Layer):
    def __init__(self, project_dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.scale = (project_dim // num_heads) ** -0.5
        self.q = keras.layers.Dense(project_dim)
        self.k = keras.layers.Dense(project_dim)
        self.v = keras.layers.Dense(project_dim)
        self.proj = keras.layers.Dense(project_dim)
        if sr_ratio > 1:
            self.sr = keras.layers.Conv2D( # 下采样
                filters=project_dim,
                kernel_size=sr_ratio,
                strides=sr_ratio,
                padding="same",
            )
            self.norm = keras.layers.LayerNormalization()

    def call(self, x):
        input_shape = ops.shape(x) # (b,s,c)
        # h,w
        H, W = int(math.sqrt(input_shape[1])), int(math.sqrt(input_shape[1]))
        B, C = input_shape[0], input_shape[2] # b,c
        q = self.q(x) # q
        q = ops.reshape( # (b,s,h,dk)
            q,
            (
                input_shape[0],
                input_shape[1],
                self.num_heads,
                input_shape[2] // self.num_heads,
            ),
        )
        q = ops.transpose(q, [0, 2, 1, 3]) # (b,h,s,dk)
        if self.sr_ratio > 1:
            # print(x.shape) (None, 3136, 32)
            # 这里先换轴,再变形,得到的图片会变成把通道内单个图拎出来
            # 排列的情况
            x = ops.reshape(  # (None, 56, 56, 32)
                ops.transpose(x, [0, 2, 1]),
                (B, H, W, C),
            )
            # print(x.shape)
            # 这时候卷积操作会对上面变形得到的图做卷积
            x = self.sr(x) # 卷积 (None, 7, 7, 32)
            # print(x.shape)
            # 变形成(b,c,s)
            x = ops.reshape(x, [input_shape[0], input_shape[2], -1]) 
            # print(x.shape),下面换轴成(b,s,c)
            x = ops.transpose(x, [0, 2, 1]) 
            # 这时得到的和原图顺序一致
            x = self.norm(x)
        k = self.k(x)
        v = self.v(x)
        k = ops.transpose(
            ops.reshape( # (b,s,h,dk)
                k,
                [B, -1, self.num_heads, C // self.num_heads],
            ),
            [0, 2, 1, 3], # (b,h,s,dk)
        )

        v = ops.transpose( 
            ops.reshape(
                v,
                [B, -1, self.num_heads, C // self.num_heads],
            ),
            [0, 2, 1, 3],# (b,h,s,dk)
        )
        # (b,h,s_q,dk)@(b,h,dk,s_k)-->(b,h,s_q,s_k)
        attn = (q @ ops.transpose(k, [0, 1, 3, 2])) * self.scale
        # 在s_k上求softmax,之后对v做加权和
        attn = ops.nn.softmax(attn, axis=-1)
        # (b,h,s_q,s_k)@(b,h,s_v,dk)-->(b,h,s_q,dk)
        attn = attn @ v
        # transpose之后是(b,s_q,h,dk),之后变形为(b,s_q,d)
        attn = ops.reshape(
            ops.transpose(attn, [0, 2, 1, 3]),
            [input_shape[0], input_shape[1], input_shape[2]],
        )
        x = self.proj(attn)  #(b,s_q,d)
        return x
class HierarchicalTransformerEncoder(keras.layers.Layer):
    def __init__(
        self,
        project_dim,
        num_heads,
        sr_ratio=1,
        drop_prob=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.drop_prop = drop_prob

        self.norm1 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.attn = SegFormerMultiheadAttention(
            project_dim, num_heads, sr_ratio
        )
        self.drop_path = DropPath(drop_prob)
        self.norm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.mlp = self.MixFFN(
            channels=project_dim,
            mid_channels=int(project_dim * 4),
        )
    def build(self, input_shape):
        super().build(input_shape)
        self.H = ops.sqrt(ops.cast(input_shape[1], "float32"))
        self.W = ops.sqrt(ops.cast(input_shape[2], "float32"))
    def call(self, x):
        # 自注意力前后残差
        x = x + self.drop_path(self.attn(self.norm1(x))) 
        # 前馈前后残差,层标准化在前
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mlp": keras.saving.serialize_keras_object(self.mlp),
                "project_dim": self.project_dim,
                "num_heads": self.num_heads,
                "drop_prop": self.drop_prop,
            }
        )
        return config
    # 自定义前馈层
    class MixFFN(keras.layers.Layer):
        def __init__(self, channels, mid_channels):
            super().__init__()
            self.fc1 = keras.layers.Dense(mid_channels)
            self.dwconv = keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=1,
                padding="same",
            )
            self.fc2 = keras.layers.Dense(channels)
        def call(self, x):
            x = self.fc1(x) # (b,s,4c)
            shape = ops.shape(x)
            H, W = int(math.sqrt(shape[1])), int(math.sqrt(shape[1]))
            B, C = shape[0], shape[2]
            x = ops.reshape(x, (B, H, W, C)) # (b,h,w,4c)
            x = self.dwconv(x) #(b,h,w,4c),深度卷积
            x = ops.reshape(x, (B, -1, C)) # (b,s,4c)
            x = ops.nn.gelu(x) # 激活函数处理
            x = self.fc2(x) #(b,s,c)
            return x

class MiTBackbone(Backbone):
    def __init__(
        self,
        include_rescaling,
        depths,
        input_shape=(224, 224, 3),
        input_tensor=None,
        embedding_dims=None,
        **kwargs,
    ):
        drop_path_rate = 0.1
        # drop率等差数列
        dpr = [x for x in np.linspace(0.0, drop_path_rate, sum(depths))]
        blockwise_num_heads = [1, 2, 5, 8]
        blockwise_sr_ratios = [8, 4, 2, 1]
        num_stages = 4
        cur = 0
        patch_embedding_layers = []
        transformer_blocks = []
        layer_norms = []
        # 单循环内流程:patch_embed_layer-->多次transformer_block
        # -->LayerNormalization
        for i in range(num_stages):
            patch_embed_layer = OverlappingPatchingAndEmbedding(
                project_dim=embedding_dims[i],
                patch_size=7 if i == 0 else 3, # 第一个核大
                stride=4 if i == 0 else 2, # 第一次下采样步幅大
                name=f"patch_and_embed_{i}",
            )
            patch_embedding_layers.append(patch_embed_layer)
            transformer_block = [
                HierarchicalTransformerEncoder(
                    project_dim=embedding_dims[i],
                    num_heads=blockwise_num_heads[i],
                    sr_ratio=blockwise_sr_ratios[i],
                    drop_prob=dpr[cur + k],
                    name=f"hierarchical_encoder_{i}_{k}",
                )
                for k in range(depths[i]) 
            ]
            transformer_blocks.append(transformer_block)
            cur += depths[i]
            layer_norms.append(keras.layers.LayerNormalization())
        inputs = parse_model_inputs(input_shape, input_tensor)
        x = inputs
        if include_rescaling: # 归一化
            x = keras.layers.Rescaling(scale=1 / 255)(x) # (None, 224, 224, 3)
        pyramid_level_inputs = []
        for i in range(num_stages):
            stride = 4 if i == 0 else 2 # 第一次步长是4
            new_height, new_width = ( 
                int(ops.shape(x)[1] / stride),
                int(ops.shape(x)[2] / stride),
            )
            x = patch_embedding_layers[i](x) # (b,s,c)  
            for blk in transformer_blocks[i]: # transformer encoder
                x = blk(x)
            x = layer_norms[i](x) # 层标准化
            x = keras.layers.Reshape(  
                (new_height, new_width, -1), name=f"output_level_{i}"
            )(x) # (None, 56, 56, 32)
            pyramid_level_inputs.append(get_tensor_input_name(x))
        super().__init__(inputs=inputs, outputs=x, **kwargs)
        self.depths = depths
        self.embedding_dims = embedding_dims
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        self.pyramid_level_inputs = {
            f"P{i + 1}": name for i, name in enumerate(pyramid_level_inputs)
        }
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "embedding_dims": self.embedding_dims,
                "include_rescaling": self.include_rescaling,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
            }
        )
        return config
class Task(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backbone = None
        self._functional_layer_ids = set(
            id(layer) for layer in self._flatten_layers()
        )
    def __dir__(self):
        def filter_fn(attr):
            if attr in ["backbone", "_backbone"]:
                return False
            try:
                return id(getattr(self, attr)) not in self._functional_layer_ids
            except:
                return True
        return filter(filter_fn, super().__dir__())
    @property
    def backbone(self):
        return self._backbone
    @backbone.setter
    def backbone(self, value):
        self._backbone = value
    def get_config(self):
        # Don't chain to super here. The default `get_config()` for functional
        # models is nested and cannot be passed to our Task constructors.
        return {
            "name": self.name,
            "trainable": self.trainable,
        }
    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        return cls(**config)
    @classproperty
    def presets(cls):
        """Dictionary of preset names and configs."""
        return {}
    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configs that include weights."""
        return {}
    @classproperty
    def presets_without_weights(cls):
        """Dictionary of preset names and configs that don't include weights."""
        return {
            preset: cls.presets[preset]
            for preset in set(cls.presets) - set(cls.presets_with_weights)
        }
    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configs for compatible backbones."""
        return {}
    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=None,
        input_shape=None,
        **kwargs,
    ):
        if preset in cls.presets:
            preset = cls.presets[preset]["kaggle_handle"]
        preset_cls = check_preset_class(preset, (cls, Backbone))
        # Backbone case.
        if issubclass(preset_cls, Backbone):
            backbone = load_from_preset(
                preset,
                load_weights=load_weights,
            )
            return cls(backbone=backbone, **kwargs)

        # Task case.
        return load_from_preset(
            preset,
            load_weights=load_weights,
            input_shape=input_shape,
            config_overrides=kwargs,
        )
    @property
    def layers(self):
        # Some of our task models don't use the Backbone directly, but create
        # a feature extractor from it. In these cases, we don't want to count
        # the `backbone` as a layer, because it will be included in the model
        # summary and saves weights despite not being part of the model graph.
        layers = super().layers
        if hasattr(self, "backbone") and self.backbone in layers:
            # We know that the backbone is not part of the graph if it has no
            # inbound nodes.
            if len(self.backbone._inbound_nodes) == 0:
                layers.remove(self.backbone)
        return layers
    def __setattr__(self, name, value):
        # Work around torch setattr for properties.
        if name in ["backbone"]:
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to set up a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have a distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        if not cls.presets:
            cls.from_preset.__func__.__doc__ = """Not implemented.

            No presets available for this class.
            """
        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = Task.from_preset.__doc__
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets_with_weights), ""),
                preset_names='", "'.join(cls.presets),
                preset_with_weights_names='", "'.join(cls.presets_with_weights),
            )(cls.from_preset.__func__)
def get_feature_extractor(model, layer_names, output_keys=None):
    if not output_keys:
        output_keys = layer_names
    items = zip(output_keys, layer_names)
    outputs = {key: model.get_layer(name).output for key, name in items}
    return keras.Model(inputs=model.inputs, outputs=outputs)
class SegFormer(Task):
    def __init__(
        self,
        backbone,
        num_classes,
        projection_filters=256,
        **kwargs,
    ):
        if not isinstance(backbone, keras.layers.Layer) or not isinstance(
            backbone, keras.Model
        ):
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance "
                f" or `keras.Model`. Received instead "
                f"backbone={backbone} (of type {type(backbone)})."
            )
        inputs = backbone.input # (None, 224, 224, 3)
        feature_extractor = get_feature_extractor( # (p1--p4)
            backbone, list(backbone.pyramid_level_inputs.values())
        )
        # feature_extractor(inputs)的输出是多级别的字典形式
        # (None, 56, 56, 32),(28, 28, 64),(14, 14, 128),(7, 7, 256)
        features = list(feature_extractor(inputs).values())
        # Get H and W of level one output
        _, H, W, _ = features[0].shape
        # Project all multi-level outputs onto the same dimensionality
        # and feature map shape
        multi_layer_outs = []
        # backbone.embedding_dims  [32, 64, 128, 256]
        for feature_dim, feature in zip(backbone.embedding_dims, features):
            # (None, 56, 56, 256),(None, 28, 28, 256),(None, 14, 14, 256),
            # (None, 7, 7, 256)
            out = keras.layers.Dense(
                projection_filters, name=f"linear_{feature_dim}"
            )(feature)
            # 调整到同一尺寸
            out = keras.layers.Resizing(H, W, interpolation="bilinear")(out)
            # (None, 56, 56, 256)
            multi_layer_outs.append(out)
        # 将现在（或当前处理阶段）大小相等的特征图（feature maps）进行拼接（concatenation）
        # 操作.因为列表内经过倒序,所以未resize前较小尺寸的特征图在前,大尺寸的在后
        concatenated_outs = keras.layers.Concatenate(axis=3)(
            multi_layer_outs[::-1]
        )
        # Fuse concatenated features into a segmentation map
        seg = keras.Sequential(
            [
                keras.layers.Conv2D( # 点卷积
                    filters=projection_filters, kernel_size=1, use_bias=False
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
            ]
        )(concatenated_outs) # (b,h,w,256)
        seg = keras.layers.Dropout(0.1)(seg)
        # 点卷积输出层,softmax处理
        seg = keras.layers.Conv2D(
            filters=num_classes, kernel_size=1, activation="softmax"
        )(seg)
        output = keras.layers.Resizing(
            height=inputs.shape[1],
            width=inputs.shape[2],
            interpolation="bilinear",
        )(seg)
        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )
        self.num_classes = num_classes
        self.projection_filters = projection_filters
        self.backbone = backbone
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "projection_filters": self.projection_filters,
                "backbone": keras.saving.serialize_keras_object(self.backbone),
            }
        )
        return config

class SegFormer(Task):
    def __init__(
        self,
        backbone,
        num_classes,
        projection_filters=256,
        **kwargs,
    ):
        if not isinstance(backbone, keras.layers.Layer) or not isinstance(
            backbone, keras.Model
        ):
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance "
                f" or `keras.Model`. Received instead "
                f"backbone={backbone} (of type {type(backbone)})."
            )
        inputs = backbone.input # (None, 224, 224, 3)
        feature_extractor = get_feature_extractor( # (p1--p4)
            backbone, list(backbone.pyramid_level_inputs.values())
        )
        # feature_extractor(inputs)的输出是多级别的字典形式
        # 不同尺寸的特征图
        # (None, 56, 56, 32),(28, 28, 64),(14, 14, 128),(7, 7, 256)
        features = list(feature_extractor(inputs).values())
        # Get H and W of level one output
        _, H, W, _ = features[0].shape
        # Project all multi-level outputs onto the same dimensionality
        # and feature map shape
        multi_layer_outs = []
        # backbone.embedding_dims  [32, 64, 128, 256]
        # 把不同尺寸的特征图投影到相同维度
        # 较浅层的特征图包含更多的细节信息（如边缘、纹理），而较深层的特征图则包含更高级别
        # 的语义信息（如对象部分、整体形状）。
        for feature_dim, feature in zip(backbone.embedding_dims, features):
            # (None, 56, 56, 256),(None, 28, 28, 256),(None, 14, 14, 256),
            # (None, 7, 7, 256)
            out = keras.layers.Dense(
                projection_filters, name=f"linear_{feature_dim}"
            )(feature)
            # 调整到同一尺寸
            out = keras.layers.Resizing(H, W, interpolation="bilinear")(out)
            # (None, 56, 56, 256)
            multi_layer_outs.append(out)
        # 将现在（或当前处理阶段）大小相等的特征图（feature maps）进行拼接（concatenation）
        # 操作.因为列表内经过倒序,所以未resize前较小尺寸的特征图在前,大尺寸的在后
        # 特征维度是从比较抽象-->具体
        concatenated_outs = keras.layers.Concatenate(axis=3)(
            multi_layer_outs[::-1]
        )
        # Fuse concatenated features into a segmentation map
        seg = keras.Sequential(
            [
                keras.layers.Conv2D( # 点卷积
                    filters=projection_filters, kernel_size=1, use_bias=False
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
            ]
        )(concatenated_outs) # (b,h,w,256)
        seg = keras.layers.Dropout(0.1)(seg)
        # 点卷积输出层,softmax处理
        seg = keras.layers.Conv2D(
            filters=num_classes, kernel_size=1, activation="softmax"
        )(seg)
        output = keras.layers.Resizing(
            height=inputs.shape[1],
            width=inputs.shape[2],
            interpolation="bilinear",
        )(seg)
        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )
        self.num_classes = num_classes
        self.projection_filters = projection_filters
        self.backbone = backbone
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "projection_filters": self.projection_filters,
                "backbone": keras.saving.serialize_keras_object(self.backbone),
            }
        )
        return config




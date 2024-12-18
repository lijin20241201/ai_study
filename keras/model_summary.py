# 用于验证并标准化 reduction 参数。这个参数通常在机器学习框架中用于指定损失函数（
# loss function）如何将输出值聚合为单个标量值。
def standardize_reduction(reduction):
    allowed = {"sum_over_batch_size", "sum", None, "none"} # 集合
    # 使用 if 语句检查 reduction 是否不在 allowed 集合中。如果是这样，函数会抛出一个 ValueError 异常，
    # 指出传入了无效的 reduction 值，并列出所有允许的值。
    if reduction not in allowed:
        raise ValueError(
            "Invalid value for argument `reduction`. "
            f"Expected one of {allowed}. Received: "
            f"reduction={reduction}"
        )
    return reduction

def reduce_values(values, reduction="sum_over_batch_size"):
    if (
        reduction is None
        or reduction == "none"
        or tuple(values.shape) == ()
        or tuple(values.shape) == (0,)
    ):
        return values
    # 计算 values 的总和，然后除以 values 中元素的总数，以得到平均值。
    loss = ops.sum(values)
    # ops.prod这个操作计算张量中所有元素的乘积。ops.shape(values)这个操作将 values 的形状转换为一个张量，
    # 并指定数据类型为 int32。
    if reduction == "sum_over_batch_size":
        loss /= ops.cast(
            ops.prod(ops.convert_to_tensor(ops.shape(values), dtype="int32")),
            loss.dtype,
        )
    return loss

# 您提到的公式涉及在损失函数中使用样本权重（sample_weight）进行加权平均。这个过程通常在处理不平
# 衡数据集或特定样本需要不同权重的情况下非常有用。
# 您提供的 apply_mask 函数用于在损失函数中应用掩码（mask），并根据指定的 reduction 方法调整样本权重（
# sample_weight）。这个函数特别适用于处理不平衡数据集或需要对某些样本赋予不同权重的情况。
def apply_mask(sample_weight, mask, dtype, reduction):
    if mask is not None:
        mask = ops.cast(mask, dtype=dtype)
        if reduction == "sum_over_batch_size":
            # Valid entries have weight `total/valid`, while invalid ones
            # have 0. When summed over batch, they will be reduced to:
            #
            # mean(loss * sample_weight * total / valid)
            #   = sum(loss * sample_weight * total / valid) / total
            #   = sum(loss * sample_weight) / total * total / valid
            #   = sum(loss * sample_weight) / valid
            total = ops.cast(
                ops.prod(ops.convert_to_tensor(ops.shape(mask), dtype="int32")),
                dtype,
            )
            valid = ops.sum(mask)  # May be 0!
            mask *= total / (valid + backend.epsilon())
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, dtype=dtype)
            mask, sample_weight = squeeze_or_expand_to_same_rank(
                mask, sample_weight
            )
            sample_weight *= mask
        else:
            sample_weight = mask
    return sample_weight
def reduce_weighted_values(
    values,
    sample_weight=None,
    mask=None,
    reduction="sum_over_batch_size",
    dtype=None,
):
    reduction = standardize_reduction(reduction) # 检验reduction
    # 转换成tensor
    values = ops.convert_to_tensor(values, dtype=dtype)
    if sample_weight is not None:
        sample_weight = ops.convert_to_tensor(sample_weight, dtype=dtype)
    if mask is not None:
        mask = ops.convert_to_tensor(mask, dtype=dtype)
    # Merge mask and sample weight into sample weight.
    sample_weight = apply_mask(
        sample_weight, mask, dtype=values.dtype, reduction=reduction
    )
    if sample_weight is not None:
        sample_weight = ops.cast(sample_weight, values.dtype)
        # Update dimensions of `sample_weight` to match `losses`.
        values, sample_weight = squeeze_or_expand_to_same_rank(
            values, sample_weight
        )
        values = values * sample_weight
    # Apply reduction function to the individual weighted losses.
    loss = reduce_values(values, reduction)
    return loss
# 这个 Loss 类提供了一个基础框架，用于定义和管理 Keras 损失函数。通过继承这个类并实现 call 方法，可以创建自定义的损失函数。
# 这个装饰器用于将类导出到 Keras API，使得用户可以通过 tf.keras.Loss 或 tf.keras.losses.Loss 来访问这个类。
@keras_export(["keras.Loss", "keras.losses.Loss"])
class Loss:
    def __init__(self, name=None, reduction="sum_over_batch_size", dtype=None):
        self.name = name or auto_name(self.__class__.__name__) # 损失函数的名称，默认为类名。
        self.reduction = standardize_reduction(reduction) # 指定如何聚合损失值，默认为 "sum_over_batch_size"。
        self.dtype = dtype or backend.floatx() # 损失值的数据类型，默认为 Keras 后端的浮点类型

    def __call__(self, y_true, y_pred, sample_weight=None):
        # 获取 y_pred 的掩码 in_mask。
        in_mask = getattr(y_pred, "_keras_mask", None)
        # 使用 ops.name_scope 为损失计算创建一个命名空间。
        with ops.name_scope(self.name):
            # 将 y_true 和 y_pred 转换为指定数据类型的张量。
            y_pred = tree.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_pred
            )
            y_true = tree.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_true
            )
            # 调用 self.call 方法计算损失值。
            losses = self.call(y_true, y_pred)
            # 获取 losses 的掩码 out_mask。
            out_mask = getattr(losses, "_keras_mask", None)
            # 根据 in_mask 和 out_mask 计算最终的掩码 mask。
            if in_mask is not None and out_mask is not None:
                mask = in_mask & out_mask
            elif in_mask is not None:
                mask = in_mask
            elif out_mask is not None:
                mask = out_mask
            else:
                mask = None
            # 调用 reduce_weighted_values 函数对损失值进行加权和聚合。
            return reduce_weighted_values(
                losses,
                sample_weight=sample_weight,
                mask=mask,
                reduction=self.reduction,
                dtype=self.dtype,
            )
    # 抽象方法 这个方法需要在子类中实现，用于计算具体的损失值
    def call(self, y_true, y_pred):
        raise NotImplementedError
    # 返回配置字典
    def get_config(self):
        return {"name": self.name, "reduction": self.reduction}
    # 类方法
    @classmethod
    def from_config(cls, config):
        return cls(**config) # 返回指定配置的实例
class CompileLoss(losses_module.Loss):
    def __init__(
        self,
        loss,
        loss_weights=None,
        reduction="sum_over_batch_size",
        output_names=None,
    ):
        if loss_weights and not isinstance(loss_weights, (list, tuple, dict)):
            raise ValueError(
                "Expected `loss_weights` argument to be a list, tuple, or "
                f"dict. Received instead: loss_weights={loss_weights} "
                f"of type {type(loss_weights)}"
            )
        self._user_loss = loss
        self._user_loss_weights = loss_weights
        self.built = False
        self.output_names = output_names
        super().__init__(name="compile_loss", reduction=reduction)

    def build(self, y_true, y_pred):
        if self.output_names:
            output_names = self.output_names
        elif isinstance(y_pred, dict):
            output_names = sorted(list(y_pred.keys()))
        elif isinstance(y_pred, (list, tuple)):
            num_outputs = len(y_pred)
            if all(hasattr(x, "_keras_history") for x in y_pred):
                output_names = [x._keras_history.operation.name for x in y_pred]
            else:
                output_names = None
        else:
            output_names = None
            num_outputs = 1
        if output_names:
            num_outputs = len(output_names)

        y_pred = self._flatten_y(y_pred)
        loss = self._user_loss
        loss_weights = self._user_loss_weights
        flat_losses = []
        flat_loss_weights = []

        if isinstance(loss, dict):
            for name in loss.keys():
                if name not in output_names:
                    raise ValueError(
                        "In the dict argument `loss`, key "
                        f"'{name}' does not correspond to any model output. "
                        f"Received:\nloss={loss}"
                    )
        if num_outputs == 1:
            if isinstance(loss, dict):
                loss = tree.flatten(loss)
            if isinstance(loss, list) and len(loss) == 1:
                loss = loss[0]
            if not is_function_like(loss):
                raise ValueError(
                    "When there is only a single output, the `loss` argument "
                    "must be a callable. "
                    f"Received instead:\nloss={loss} of type {type(loss)}"
                )
            if isinstance(y_pred, list) and len(y_pred) == 1:
                y_pred = y_pred[0]

        if is_function_like(loss) and tree.is_nested(y_pred):
            # The model has multiple outputs but only one loss fn
            # was provided. Broadcast loss to all outputs.
            loss = tree.map_structure(lambda x: loss, y_pred)

        # Iterate over all possible loss formats:
        # plain function, list/tuple, dict
        if is_function_like(loss):
            flat_losses.append(get_loss(loss, y_true, y_pred))
            if loss_weights:
                if not isinstance(loss_weights, float):
                    raise ValueError(
                        "When there is only a single output, the "
                        "`loss_weights` argument "
                        "must be a Python float. "
                        f"Received instead: loss_weights={loss_weights} of "
                        f"type {type(loss_weights)}"
                    )
                flat_loss_weights.append(loss_weights)
            else:
                flat_loss_weights.append(1.0)
        elif isinstance(loss, (list, tuple)):
            loss = tree.flatten(loss)
            if len(loss) != len(y_pred):
                raise ValueError(
                    "For a model with multiple outputs, "
                    "when providing the `loss` argument as a list, "
                    "it should have as many entries as the model has outputs. "
                    f"Received:\nloss={loss}\nof length {len(loss)} "
                    f"whereas the model has {len(y_pred)} outputs."
                )
            if not all(is_function_like(e) for e in loss):
                raise ValueError(
                    "For a model with multiple outputs, "
                    "when providing the `loss` argument as a list, "
                    "each list entry should be a callable (the loss function "
                    "corresponding to that output). "
                    f"Received: loss={loss}"
                )
            flat_losses = [
                get_loss(fn, y_true, y_pred) for fn in loss if fn is not None
            ]
            if loss_weights:
                if not isinstance(loss_weights, (list, tuple)):
                    raise ValueError(
                        "If the `loss` argument is provided as a list/tuple, "
                        "the `loss_weight` argument should also be provided as "
                        "a list/tuple, of equal length. "
                        f"Received: loss_weights={loss_weights}"
                    )
                if len(loss_weights) != len(y_pred):
                    raise ValueError(
                        "For a model with multiple outputs, "
                        "when providing the `loss_weights` argument as a list, "
                        "it should have as many entries as the model has "
                        f"outputs. Received: loss_weights={loss_weights} of "
                        f"length {len(loss_weights)} whereas the model has "
                        f"{len(y_pred)} outputs."
                    )
                if not all(isinstance(e, (int, float)) for e in loss_weights):
                    raise ValueError(
                        "For a model with multiple outputs, when providing "
                        "the `loss_weights` argument as a list, "
                        "each list entry should be a Python int or float (the "
                        "weighting coefficient corresponding to the loss for "
                        f"that output). Received: loss_weights={loss_weights}"
                    )
                flat_loss_weights = list(loss_weights)
            else:
                flat_loss_weights = [1.0 for _ in loss]
        elif isinstance(loss, dict):
            if output_names is None:
                raise ValueError(
                    "Argument `loss` can only be provided as a dict "
                    "when the model also returns a dict of outputs. "
                    f"Received loss={loss}"
                )
            for name in loss.keys():
                if isinstance(loss[name], list) and len(loss[name]) == 1:
                    loss[name] = loss[name][0]
                if not is_function_like(loss[name]):
                    raise ValueError(
                        "For a model with multiple outputs, "
                        "when providing the `loss` argument as a dict, "
                        "each dict entry should be a callable (the loss "
                        "function corresponding to that output). "
                        f"At key '{name}', received invalid type:\n{loss[name]}"
                    )
            for name, yt, yp in zip(output_names, y_true, y_pred):
                if name in loss:
                    if loss[name]:
                        flat_losses.append(get_loss(loss[name], yt, yp))
                    else:
                        flat_losses.append(None)
                else:
                    flat_losses.append(None)
            if loss_weights:
                if not isinstance(loss_weights, dict):
                    raise ValueError(
                        "If the `loss` argument is provided as a dict, "
                        "the `loss_weight` argument should also be provided as "
                        f"a dict. Received: loss_weights={loss_weights}"
                    )
                for name in loss_weights.keys():
                    if name not in output_names:
                        raise ValueError(
                            "In the dict argument `loss_weights`, key "
                            f"'{name}' does not correspond to any model "
                            f"output. Received: loss_weights={loss_weights}"
                        )
                    if not isinstance(loss_weights[name], float):
                        raise ValueError(
                            "For a model with multiple outputs, "
                            "when providing the `loss_weights` argument as a "
                            "dict, each dict entry should be a Python float "
                            "(the weighting coefficient corresponding to the "
                            f"loss for that output). At key '{name}', "
                            f"received invalid type:\n{loss_weights[name]}"
                        )
                for name in output_names:
                    if name in loss_weights:
                        flat_loss_weights.append(loss_weights[name])
                    else:
                        flat_loss_weights.append(1.0)
            else:
                flat_loss_weights = [1.0 for _ in flat_losses]
        self.flat_losses = flat_losses
        self.flat_loss_weights = flat_loss_weights
        self.built = True

    def __call__(self, y_true, y_pred, sample_weight=None):
        with ops.name_scope(self.name):
            return self.call(y_true, y_pred, sample_weight)

    def _flatten_y(self, y):
        if isinstance(y, dict) and self.output_names:
            result = []
            for name in self.output_names:
                if name in y:
                    result.append(y[name])
            return result
        return tree.flatten(y)

    def call(self, y_true, y_pred, sample_weight=None):
        if not self.built:
            self.build(y_true, y_pred)

        y_true = self._flatten_y(y_true)
        y_pred = self._flatten_y(y_pred)

        if sample_weight is not None:
            sample_weight = self._flatten_y(sample_weight)
            # For multi-outputs, repeat sample weights for n outputs.
            if len(sample_weight) < len(y_true):
                sample_weight = [sample_weight[0] for _ in range(len(y_true))]
        else:
            sample_weight = [None for _ in y_true]

        loss_values = []
        for loss, y_t, y_p, loss_weight, sample_weight in zip(
            self.flat_losses,
            y_true,
            y_pred,
            self.flat_loss_weights,
            sample_weight,
        ):
            if loss:
                value = loss_weight * ops.cast(
                    loss(y_t, y_p, sample_weight), dtype=backend.floatx()
                )
                loss_values.append(value)
        if loss_values:
            total_loss = sum(loss_values)
            return total_loss
        return None

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
def print_summary(
    model,
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    show_trainable=False,
    layer_range=None,
):
    # 这两行导入了 Keras 中的 Functional 和 Sequential 模型类
    from keras.src.models import Functional
    from keras.src.models import Sequential
    # 设置默认的打印函数
    # 如果没有提供 print_fn 并且 Keras 的交互式日志记录未启用，则使用 io_utils.print_msg 作为默认的打印函数。
    if not print_fn and not io_utils.is_interactive_logging_enabled():
        print_fn = io_utils.print_msg
    # 判断模型类型
    # 首先判断模型是否为 Sequential 类型，如果是，则认为模型是顺序的，并获取其层列表。
    if isinstance(model, Sequential):
        sequential_like = True
        layers = model.layers
    # 如果模型不是 Functional 类型，将其视为子类化模型，并同样认为模型是顺序的
    elif not isinstance(model, Functional):
        # We treat subclassed models as a simple sequence of layers, for logging
        # purposes.
        sequential_like = True
        layers = model.layers
    # 如果模型是 Functional 类型，则进一步检查模型是否有多个节点或共享层。如果有，则模型不再是顺序的。
    else:
        layers = model._operations
        sequential_like = True
        nodes_by_depth = model._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or (
                len(v) == 1 and len(tree.flatten(v[0].input_tensors)) > 1
            ):
                # if the model has multiple nodes
                # or if the nodes have multiple inbound_layers
                # the model is no longer sequential
                sequential_like = False
                break
            nodes += v
        
        if sequential_like:
            # search for shared layers
            for layer in model.layers:
                flag = False
                for node in layer._inbound_nodes:
                    if node in nodes:
                        if flag:
                            sequential_like = False
                            break
                        else:
                            flag = True
                if not sequential_like:
                    break
    # 设置默认的行长度和位置
    # 根据模型是否为顺序的，设置默认的行长度和列位置。
    # 定义表头和对齐方式。如果模型不是顺序的，收集所有相关的节点。
    if sequential_like:
        default_line_length = 88
        positions = positions or [0.45, 0.80, 1.0]
        # header names for the different log elements
        header = ["Layer (type)", "Output Shape", "Param #"]
        alignment = ["left", "left", "right"]
    else:
        default_line_length = 108
        positions = positions or [0.3, 0.56, 0.74, 1.0]
        # header names for the different log elements
        header = ["Layer (type)", "Output Shape", "Param #", "Connected to"]
        alignment = ["left", "left", "right", "left"]
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v
    # 处理显示可训练状态的情况
    # 如果需要显示可训练状态，增加行长度，并调整列位置，添加新的表头和对齐方式。
    if show_trainable:
        default_line_length += 12
        positions = [p * 0.90 for p in positions] + [1.0]
        header.append("Trainable")
        alignment.append("center")

    # Compute columns widths
    # 获取终端的宽度，并计算每列的宽度。如果某列的宽度小于4，则抛出异常。
    default_line_length = min(
        default_line_length, shutil.get_terminal_size().columns - 4
    )
    line_length = line_length or default_line_length
    column_widths = []
    current = 0
    for pos in positions:
        width = int(pos * line_length) - current
        if width < 4:
            raise ValueError("Insufficient console width to print summary.")
        column_widths.append(width)
        current += width

    # Render summary as a rich table.创建表格
    # 使用 rich 库创建表格，设置列名、对齐方式和宽度。
    columns = []
    # Right align parameter counts.
    for i, name in enumerate(header):
        column = rich.table.Column(
            name,
            justify=alignment[i],
            width=column_widths[i],
        )
        columns.append(column)

    table = rich.table.Table(*columns, width=line_length, show_lines=True)
    # 定义辅助函数,获取层的连接信息。
    def get_connections(layer):
        connections = ""
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue
            for kt in node.input_tensors:
                keras_history = kt._keras_history
                inbound_layer = keras_history.operation
                node_index = highlight_number(keras_history.node_index)
                tensor_index = highlight_number(keras_history.tensor_index)
                if connections:
                    connections += ", "
                connections += (
                    f"{inbound_layer.name}[{node_index}][{tensor_index}]"
                )
        if not connections:
            connections = "-"
        return connections
    # 获取层的字段信息，包括名称、输出形状、参数数量、连接信息和可训练状态。
    def get_layer_fields(layer, prefix=""):
        output_shape = format_layer_shape(layer)
        name = prefix + layer.name
        cls_name = layer.__class__.__name__
        name = rich.markup.escape(name)
        name += f" ({highlight_symbol(rich.markup.escape(cls_name))})"

        if not hasattr(layer, "built"):
            params = highlight_number(0)
        elif not layer.built:
            params = highlight_number(0) + " (unbuilt)"
        else:
            params = highlight_number(f"{layer.count_params():,}")

        fields = [name, output_shape, params]
        if not sequential_like:
            fields.append(get_connections(layer))
        if show_trainable:
            if layer.weights:
                fields.append(
                    bold_text("Y", color=34)
                    if layer.trainable
                    else bold_text("N", color=9)
                )
            else:
                fields.append(bold_text("-"))
        return fields
    # 打印层的信息，支持嵌套层的展开
    def print_layer(layer, nested_level=0):
        if nested_level:
            prefix = "   " * nested_level + "└" + " "
        else:
            prefix = ""

        fields = get_layer_fields(layer, prefix=prefix)

        rows = [fields]
        if expand_nested and hasattr(layer, "layers") and layer.layers:
            nested_layers = layer.layers
            nested_level += 1
            for i in range(len(nested_layers)):
                rows.extend(
                    print_layer(nested_layers[i], nested_level=nested_level)
                )
        return rows

    # Render all layers to the rich table.渲染所有层到表格
    # 根据指定的层范围，将所有层的信息添加到表格中。
    layer_range = get_layer_index_bound_by_layer_name(layers, layer_range)
    for layer in layers[layer_range[0] : layer_range[1]]:
        for row in print_layer(layer):
            table.add_row(*row)
    # 计算参数数量和大小
    # After the table, append information about parameter count and size.
    if hasattr(model, "_collected_trainable_weights"):
        trainable_count = count_params(model._collected_trainable_weights)
        trainable_memory_size = weight_memory_size(
            model._collected_trainable_weights
        )
    else:
        trainable_count = count_params(model.trainable_weights)
        trainable_memory_size = weight_memory_size(model.trainable_weights)
    
    non_trainable_count = count_params(model.non_trainable_weights)
    non_trainable_memory_size = weight_memory_size(model.non_trainable_weights)

    if model.compiled and model.optimizer and model.optimizer.built:
        optimizer_weight_count = count_params(model.optimizer.variables)
        optimizer_memory_size = weight_memory_size(model.optimizer.variables)
        optimizer_built = True
    else:
        optimizer_weight_count = 0
        optimizer_memory_size = 0
        optimizer_built = False

    total_count = trainable_count + non_trainable_count + optimizer_weight_count
    total_memory_size = (
        trainable_memory_size
        + non_trainable_memory_size
        + optimizer_memory_size
    )
    # 计算可训练参数、不可训练参数和优化器参数的数量和内存大小。
    # Create a rich console for printing. Capture for non-interactive logging.
    if print_fn:
        console = rich.console.Console(
            highlight=False, force_terminal=False, color_system=None
        )
        console.begin_capture()
    else:
        console = rich.console.Console(highlight=False)

    # Print the to the console.
    # 创建控制台并打印信息
    # 创建 rich 控制台，打印模型名称、表格和参数信息。
    console.print(bold_text(f'Model: "{rich.markup.escape(model.name)}"'))
    console.print(table)
    console.print(
        bold_text(" Total params: ")
        + highlight_number(f"{total_count:,}")
        + f" ({readable_memory_size(total_memory_size)})"
    )
    console.print(
        bold_text(" Trainable params: ")
        + highlight_number(f"{trainable_count:,}")
        + f" ({readable_memory_size(trainable_memory_size)})"
    )
    console.print(
        bold_text(" Non-trainable params: ")
        + highlight_number(f"{non_trainable_count:,}")
        + f" ({readable_memory_size(non_trainable_memory_size)})"
    )
    if optimizer_built:
        console.print(
            bold_text(" Optimizer params: ")
            + highlight_number(f"{optimizer_weight_count:,}")
            + f" ({readable_memory_size(optimizer_memory_size)})"
        )
    # 输出捕获的摘要信息
    # 如果提供了 print_fn，则捕获并输出摘要信息。
    # Output captured summary for non-interactive logging.
    if print_fn:
        print_fn(console.end_capture(), line_break=False)
class Trainer:
    def __init__(self):
        self._lock = False
        self._run_eagerly = False
        self._jit_compile = None
        self.compiled = False
        self.loss = None
        self.steps_per_execution = 1

    @traceback_utils.filter_traceback
    @tracking.no_automatic_dependency_tracking
    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile="auto",
        auto_scale_loss=True,
    ):
        
        self.optimizer = optimizers.get(optimizer)
        if (
            auto_scale_loss
            and self.dtype_policy.name == "mixed_float16"
            and self.optimizer
            and not isinstance(self.optimizer, LossScaleOptimizer)
        ):
            self.optimizer = LossScaleOptimizer(
                self.optimizer, name="loss_scale_optimizer"
            )
        if hasattr(self, "output_names"):
            output_names = self.output_names
        else:
            output_names = None
        if loss is not None:
            self._compile_loss = CompileLoss(
                loss, loss_weights, output_names=output_names
            )
            self.loss = loss
        else:
            self._compile_loss = None
        if metrics is not None or weighted_metrics is not None:
            self._compile_metrics = CompileMetrics(
                metrics, weighted_metrics, output_names=output_names
            )
        else:
            self._compile_metrics = None
        if jit_compile == "auto":
            if run_eagerly:
                jit_compile = False
            else:
                jit_compile = self._resolve_auto_jit_compile()
        if jit_compile and run_eagerly:
            jit_compile = False
            warnings.warn(
                "If `run_eagerly` is True, then `jit_compile` "
                "cannot also be True. Disabling `jit_compile`.",
                stacklevel=2,
            )

        self.jit_compile = jit_compile
        self.run_eagerly = run_eagerly
        self.stop_training = False
        self.compiled = True
        self._loss_tracker = metrics_module.Mean(name="loss")
        self.steps_per_execution = steps_per_execution

        self.train_function = None
        self.test_function = None
        self.predict_function = None

        self._compile_config = serialization_lib.SerializableDict(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
        )

    @property
    def jit_compile(self):
        if self._jit_compile is None:
            # Value was never set. Resolve it now.
            self._jit_compile = self._resolve_auto_jit_compile()
        return self._jit_compile

    @jit_compile.setter
    def jit_compile(self, value):
        if value and not model_supports_jit(self):
            warnings.warn(
                "Model doesn't support `jit_compile=True`. "
                "Proceeding with `jit_compile=False`."
            )
            self._jit_compile = False
        else:
            self._jit_compile = value

    def _resolve_auto_jit_compile(self):
        if backend.backend() == "torch":
            # jit_compile = "auto" with the pytorch backend defaults to eager
            return False

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            devices = tf.config.list_physical_devices()
            if not list(filter(lambda x: x.device_type != "CPU", devices)):
                # Disable XLA on CPU-only machines.
                return False

            if self._distribute_strategy:
                # Disable XLA with tf.distribute
                return False

        if model_supports_jit(self):
            return True
        return False

    @property
    def run_eagerly(self):
        return self._run_eagerly

    @run_eagerly.setter
    def run_eagerly(self, value):
        self._run_eagerly = value

    @property
    def metrics(self):
        metrics = [self._loss_tracker] if self.compiled else []
        metrics.extend(self._metrics[:])
        if self.compiled and self._compile_metrics is not None:
            metrics += [self._compile_metrics]
        return metrics

    @property
    def metrics_names(self):
        return [m.name for m in self.metrics]

    @property
    def metrics_variables(self):
        vars = []
        for metric in self.metrics:
            vars.extend(metric.variables)
        return vars

    def reset_metrics(self):
        for m in self.metrics:
            m.reset_state()

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
      
        del x  # The default implementation does not use `x`.
        losses = []
        if self._compile_loss is not None:
            loss = self._compile_loss(y, y_pred, sample_weight)
            if loss is not None:
                losses.append(loss)
        for loss in self.losses:
            losses.append(ops.cast(loss, dtype=backend.floatx()))
        if not allow_empty and len(losses) == 0:
            raise ValueError(
                "No loss to compute. Provide a `loss` argument in `compile()`."
            )
        if len(losses) == 1:
            total_loss = losses[0]
        elif len(losses) == 0:
            total_loss = ops.zeros(())
        else:
            total_loss = ops.sum(losses)
        return total_loss

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
       
        del x  # The default implementation does not use `x`.
        if self._compile_metrics is not None:
            self._compile_metrics.update_state(y, y_pred, sample_weight)
        return self.get_metrics_result()

    def get_metrics_result(self):
       
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return self._pythonify_logs(return_metrics)

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
       
        raise NotImplementedError

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        
        raise NotImplementedError

    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
       
        raise NotImplementedError

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
       
        raise NotImplementedError

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
       
        raise NotImplementedError

    def predict_on_batch(self, x):
       
        raise NotImplementedError

    def get_compile_config(self):
       
        if self.compiled and hasattr(self, "_compile_config"):
            return self._compile_config.serialize()

    def compile_from_config(self, config):
        
        has_overridden_compile = self.__class__.compile != Trainer.compile
        if has_overridden_compile:
            warnings.warn(
                "`compile()` was not called as part of model loading "
                "because the model's `compile()` method is custom. "
                "All subclassed Models that have `compile()` "
                "overridden should also override "
                "`get_compile_config()` and `compile_from_config(config)`. "
                "Alternatively, you can "
                "call `compile()` manually after loading.",
                stacklevel=2,
            )
            return
        config = serialization_lib.deserialize_keras_object(config)
        self.compile(**config)
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables.
            self.optimizer.build(self.trainable_variables)

    def _should_eval(self, epoch, validation_freq):
        epoch = epoch + 1  # one-index the user-facing epoch.
        if isinstance(validation_freq, int):
            return epoch % validation_freq == 0
        elif isinstance(validation_freq, list):
            return epoch in validation_freq
        else:
            raise ValueError(
                "Expected `validation_freq` to be a list or int. "
                f"Received: validation_freq={validation_freq} of the "
                f"type {type(validation_freq)}."
            )

    def _pythonify_logs(self, logs):
        result = {}
        for key, value in sorted(logs.items()):
            if isinstance(value, dict):
                result.update(self._pythonify_logs(value))
            else:
                try:
                    value = float(value)
                except:
                    pass
                result[key] = value
        return result

    def _flatten_metrics_in_order(self, logs):
        """Turns `logs` dict into a list as per key order of `metrics_names`."""
        metric_names = [m.name for m in self.metrics]
        results = []
        for name in metric_names:
            if name in logs:
                results.append(logs[name])
        for key in sorted(logs.keys()):
            if key not in metric_names:
                results.append(logs[key])
        if len(results) == 1:
            return results[0]
        return results

    def _assert_compile_called(self, method_name=None):
        if not self.compiled:
            msg = "You must call `compile()` before "
            if metrics_module:
                msg += "using the model."
            else:
                msg += f"calling `{method_name}()`."
            raise ValueError(msg)

    def _symbolic_build(self, iterator=None, data_batch=None):
        model_unbuilt = not all(layer.built for layer in self._flatten_layers())
        compile_metrics_unbuilt = (
            self._compile_metrics is not None
            and not self._compile_metrics.built
        )
        optimizer_unbuilt = (
            self.optimizer is not None and not self.optimizer.built
        )
        if model_unbuilt or compile_metrics_unbuilt or optimizer_unbuilt:
            if data_batch is None:
                for _, data in iterator.enumerate_epoch():
                    data_batch = data[0]
                    break

        if model_unbuilt or compile_metrics_unbuilt:
            # Create symbolic tensors matching an input batch.

            def to_symbolic_input(v):
                if v is None:
                    return None
                return backend.KerasTensor(
                    v.shape, backend.standardize_dtype(v.dtype)
                )

            data_batch = tree.map_structure(to_symbolic_input, data_batch)
            (
                x,
                y,
                sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(data_batch)
            # Build all model state with `backend.compute_output_spec`.
            try:
                y_pred = backend.compute_output_spec(self, x)
            except Exception as e:
                raise RuntimeError(
                    "Unable to automatically build the model. "
                    "Please build it yourself before calling "
                    "fit/evaluate/predict. "
                    "A model is 'built' when its variables have "
                    "been created and its `self.built` attribute "
                    "is True. Usually, calling the model on a batch "
                    "of data is the right way to build it.\n"
                    "Exception encountered:\n"
                    f"'{e}'"
                )
            if compile_metrics_unbuilt:
                # Build all metric state with `backend.compute_output_spec`.
                backend.compute_output_spec(
                    self.compute_metrics,
                    x,
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                )
        if optimizer_unbuilt:
            # Build optimizer
            self.optimizer.build(self.trainable_variables)
        self._post_build()
class TensorFlowTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

        # Model must be created under scope of DistStrat it will be trained
        # with.
        if tf.distribute.has_strategy():
            self._distribute_strategy = tf.distribute.get_strategy()
        else:
            self._distribute_strategy = None

        self._distribute_reduction_method = None
        self._supports_reduce_retracing = Version(tf.__version__) >= Version(
            "2.9.0"
        )

    @property
    def distribute_strategy(self):
        return self._distribute_strategy or tf.distribute.get_strategy()

    @property
    def distribute_reduction_method(self):
        return self._distribute_reduction_method or "auto"

    @distribute_reduction_method.setter
    def distribute_reduction_method(self, value):
        self._distribute_reduction_method = value

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        # Forward pass
        with tf.GradientTape() as tape:
            if self._call_has_training_arg:
                y_pred = self(x, training=True)
            else:
                y_pred = self(x)
            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
            )
            self._loss_tracker.update_state(loss)
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        if self.trainable_weights:
            trainable_weights = self.trainable_weights
            gradients = tape.gradient(loss, trainable_weights)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        else:
            warnings.warn("The model does not have any trainable weights.")

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        loss = self.compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
        )
        self._loss_tracker.update_state(loss)
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        return y_pred

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a single training step on a batch of data."""
            return self.train_step(data)

        if not self.run_eagerly:
            kwargs = {"jit_compile": self.jit_compile}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})
            one_step_on_data = tf.function(one_step_on_data, **kwargs)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_iterator(iterator):
            """Runs a single training step given a Dataset iterator."""
            data = next(iterator)
            outputs = self.distribute_strategy.run(
                one_step_on_data, args=(data,)
            )
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_iterator(iterator):
            for _ in range(self.steps_per_execution):
                outputs = one_step_on_iterator(iterator)
            return outputs

        if self.steps_per_execution > 1:
            train_function = multi_step_on_iterator
        else:
            train_function = one_step_on_iterator

        if not self.run_eagerly:
            kwargs = {}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})
            train_function = tf.function(train_function, **kwargs)

        self.train_function = train_function

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a single test step on a batch of data."""
            return self.test_step(data)

        if not self.run_eagerly and self.jit_compile:
            kwargs = {"jit_compile": True}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})
            one_step_on_data = tf.function(one_step_on_data, **kwargs)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_iterator(iterator):
            """Runs a single test step given a Dataset iterator."""
            data = next(iterator)
            outputs = self.distribute_strategy.run(
                one_step_on_data, args=(data,)
            )
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_iterator(iterator):
            for _ in range(self.steps_per_execution):
                outputs = one_step_on_iterator(iterator)
            return outputs

        if self.steps_per_execution > 1:
            test_function = multi_step_on_iterator
        else:
            test_function = one_step_on_iterator

        if not self.run_eagerly:
            kwargs = {}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})
            test_function = tf.function(test_function, **kwargs)

        self.test_function = test_function

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a predict test step on a batch of data."""
            return self.predict_step(data)

        if not self.run_eagerly and self.jit_compile:
            kwargs = {"jit_compile": True}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})
            one_step_on_data = tf.function(one_step_on_data, **kwargs)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data_distributed(data):
            data = data[0]
            outputs = self.distribute_strategy.run(
                one_step_on_data, args=(data,)
            )
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_data(data):
            outputs = one_step_on_data_distributed(data[:1])
            for single_step_data in data[1:]:
                step_outputs = one_step_on_data_distributed([single_step_data])
                outputs = tf.nest.map_structure(
                    lambda t1, t2: concat([t1, t2]), outputs, step_outputs
                )
            return outputs

        if self.steps_per_execution > 1:
            predict_function = multi_step_on_data
        else:
            predict_function = one_step_on_data_distributed

        if not self.run_eagerly:
            kwargs = {}
            if self._supports_reduce_retracing:
                kwargs.update({"reduce_retracing": True})

            predict_function = tf.function(predict_function, **kwargs)

        self.predict_function = predict_function

    @traceback_utils.filter_traceback
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        self._assert_compile_called("fit")
        # TODO: respect compiled trainable state
        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            (
                x,
                y,
                sample_weight,
            ), validation_data = data_adapter_utils.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data is not None:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        # Create an iterator that yields batches for one epoch.
        epoch_iterator = TFEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            distribute_strategy=self.distribute_strategy,
            steps_per_execution=self.steps_per_execution,
        )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.stop_training = False
        self.make_train_function()
        callbacks.on_train_begin()
        training_logs = None
        logs = None
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            with epoch_iterator.catch_stop_iteration():
                for step, iterator in epoch_iterator.enumerate_epoch():
                    callbacks.on_train_batch_begin(step)
                    logs = self.train_function(iterator)
                    callbacks.on_train_batch_end(
                        step, self._pythonify_logs(logs)
                    )
                    if self.stop_training:
                        break

            # Override with model metrics instead of last step logs
            epoch_logs = self.get_metrics_result()

            # Run validation.
            if validation_data is not None and self._should_eval(
                epoch, validation_freq
            ):
                # Create EpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = TFEpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        distribute_strategy=self.distribute_strategy,
                        steps_per_execution=self.steps_per_execution,
                        steps_per_epoch=validation_steps,
                        shuffle=False,
                    )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {
                    "val_" + name: val for name, val in val_logs.items()
                }
                epoch_logs.update(val_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        if (
            isinstance(self.optimizer, optimizers_module.Optimizer)
            and epochs > 0
        ):
            self.optimizer.finalize_variable_values(self.trainable_weights)

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history

    @traceback_utils.filter_traceback
    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        self._assert_compile_called("evaluate")
        # TODO: respect compiled trainable state
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of input/target data.
            epoch_iterator = TFEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                distribute_strategy=self.distribute_strategy,
                steps_per_execution=self.steps_per_execution,
            )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = None
        self.reset_metrics()
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator.enumerate_epoch():
                callbacks.on_test_batch_begin(step)
                logs = self.test_function(iterator)
                callbacks.on_test_batch_end(step, self._pythonify_logs(logs))
                if self.stop_evaluating:
                    break
        logs = self.get_metrics_result()
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    @traceback_utils.filter_traceback
    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = TFEpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            distribute_strategy=self.distribute_strategy,
            steps_per_execution=self.steps_per_execution,
        )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tf.nest.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tree.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        def get_data(iterator):
            """Returns data for the next execution."""
            data = []
            for _ in range(self.steps_per_execution):
                try:
                    single_step_data = next(iterator)
                except (StopIteration, tf.errors.OutOfRangeError) as e:
                    if hasattr(data, "__len__") and len(data) > 0:
                        # Suppress the error when still have remaining data.
                        return data
                    else:
                        # Re-raise the error for
                        # TFEpochIterator.catch_stop_iteration() to catch when
                        # no data left.
                        raise e
                data.append(single_step_data)
            return data

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()
        outputs = None
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator.enumerate_epoch():
                callbacks.on_predict_batch_begin(step)
                data = get_data(iterator)
                batch_outputs = self.predict_function(data)
                outputs = append_to_outputs(batch_outputs, outputs)
                callbacks.on_predict_batch_end(step, {"outputs": batch_outputs})
                if self.stop_predicting:
                    break
        callbacks.on_predict_end()
        outputs = tree.map_structure_up_to(
            batch_outputs, potentially_ragged_concat, outputs
        )
        return tf.nest.map_structure(convert_to_np_if_not_ragged, outputs)

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("train_on_batch")
        self.make_train_function()
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(
                    "Arguments `sample_weight` and `class_weight` "
                    "cannot be specified at the same time. "
                    f"Received: sample_weight={sample_weight}, "
                    f"class_weight={class_weight}"
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        def data():
            yield (x, y, sample_weight)

        logs = self.train_function(data())
        logs = tf.nest.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("test_on_batch")
        self.make_test_function()

        def data():
            yield (x, y, sample_weight)

        logs = self.test_function(data())
        logs = tf.nest.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        self.make_predict_function()
        batch_outputs = self.predict_function([(x,)])
        batch_outputs = tf.nest.map_structure(
            convert_to_np_if_not_ragged, batch_outputs
        )
        return batch_outputs

    # Backwards compatibility shims.
    @property
    def compiled_metrics(self):
        class DeprecatedCompiledMetric:
            def update_state(_, y, y_pred, sample_weight=None):
                return self._compiled_metrics_update_state(
                    y, y_pred, sample_weight=sample_weight
                )

        return DeprecatedCompiledMetric()

    def _compiled_metrics_update_state(self, y, y_pred, sample_weight=None):
        warnings.warn(
            "`model.compiled_metrics()` is deprecated. "
            "Instead, use e.g.:\n"
            "```\n"
            "for metric in self.metrics:\n"
            "    metric.update_state(y, y_pred)\n"
            "```\n",
            stacklevel=2,
        )
        for metric in self.metrics:
            if isinstance(metric, metrics_module.Mean):
                metric.update_state(y_pred, sample_weight=sample_weight)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

    def compiled_loss(
        self, y, y_pred, sample_weight=None, regularization_losses=None
    ):
        warnings.warn(
            "`model.compiled_loss()` is deprecated. "
            "Instead, use `model.compute_loss(x, y, y_pred, sample_weight)`.",
        )
        return self.compute_loss(
            x=None, y=y, y_pred=y_pred, sample_weight=sample_weight
        )

    def loss(self, y, y_pred, sample_weight=None):
        warnings.warn(
            "`model.loss` is deprecated. "
            "Instead, use `model.compute_loss(x, y, y_pred, sample_weight)`.",
        )
        return self.compute_loss(
            x=None, y=y, y_pred=y_pred, sample_weight=sample_weight
        )
@keras_export(["keras.Model", "keras.models.Model"])
class Model(Trainer, Layer):
    def __new__(cls, *args, **kwargs):
        # Signature detection for usage of `Model` as a `Functional`
        if functional_init_arguments(args, kwargs) and cls == Model:
            from keras.src.models import functional

            return functional.Functional(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        Trainer.__init__(self)
        from keras.src.models import functional

        # Signature detection for usage of a `Model` subclass
        # as a `Functional` subclass
        if functional_init_arguments(args, kwargs):
            inject_functional_model_class(self.__class__)
            functional.Functional.__init__(self, *args, **kwargs)
        else:
            Layer.__init__(self, *args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError(
            f"Model {self.__class__.__name__} does not have a `call()` "
            "method implemented."
        )

    @property
    def layers(self):
        return list(self._flatten_layers(include_self=False, recursive=False))

    @layers.setter
    def layers(self, _):
        raise AttributeError(
            "`Model.layers` attribute is reserved and should not be used. "
            "Please use another name."
        )

    @traceback_utils.filter_traceback
    def get_layer(self, name=None, index=None):
        if index is not None and name is not None:
            raise ValueError(
                "Provide only a layer name or a layer index. Received: "
                f"index={index}, name={name}."
            )
        if index is not None:
            if len(self.layers) <= index:
                raise ValueError(
                    f"Was asked to retrieve layer at index {index}"
                    f" but model only has {len(self.layers)}"
                    " layers."
                )
            else:
                return self.layers[index]

        if name is not None:
            for layer in self.layers:
                if layer.name == name:
                    return layer
            raise ValueError(
                f"No such layer: {name}. Existing layers are: "
                f"{list(layer.name for layer in self.layers)}."
            )
        raise ValueError(
            "Provide either a layer name or layer index at `get_layer`."
        )

    @traceback_utils.filter_traceback
    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        expand_nested=False,
        show_trainable=False,
        layer_range=None,
    ):
       
        summary_utils.print_summary(
            self,
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
            show_trainable=show_trainable,
            layer_range=layer_range,
        )

    @traceback_utils.filter_traceback
    def save(self, filepath, overwrite=True, **kwargs):
      
        return saving_api.save_model(self, filepath, overwrite, **kwargs)

    @traceback_utils.filter_traceback
    def save_weights(self, filepath, overwrite=True):
       
        if not str(filepath).endswith(".weights.h5"):
            raise ValueError(
                "The filename must end in `.weights.h5`. "
                f"Received: filepath={filepath}"
            )
        try:
            exists = os.path.exists(filepath)
        except TypeError:
            exists = False
        if exists and not overwrite:
            proceed = io_utils.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
        saving_lib.save_weights_only(self, filepath)

    @traceback_utils.filter_traceback
    def load_weights(self, filepath, skip_mismatch=False, **kwargs):
       
        saving_api.load_weights(
            self, filepath, skip_mismatch=skip_mismatch, **kwargs
        )

    def build_from_config(self, config):
        if not config:
            return
        if "input_shape" in config:
            # Case: all inputs are in the first arg (possibly nested).
            if utils.is_default(self.build):
                status = self._build_by_run_for_single_pos_arg(
                    config["input_shape"]
                )
            else:
                try:
                    self.build(config["input_shape"])
                    status = True
                except:
                    status = False
            self._build_shapes_dict = config

        elif "shapes_dict" in config:
            # Case: inputs were recorded as multiple keyword arguments.
            if utils.is_default(self.build):
                status = self._build_by_run_for_kwargs(config["shapes_dict"])
            else:
                try:
                    self.build(**config["shapes_dict"])
                    status = True
                except:
                    status = False
            self._build_shapes_dict = config["shapes_dict"]

        if not status:
            warnings.warn(
                f"Model '{self.name}' had a build config, but the model "
                "cannot be built automatically in "
                "`build_from_config(config)`. "
                "You should implement "
                "`def build_from_config(self, config)`, "
                "and you might also want to implement the method "
                " that generates the config at saving time, "
                "`def get_build_config(self)`. "
                "The method `build_from_config()` is meant to "
                "create the state of the model (i.e. its variables) "
                "upon deserialization.",
                stacklevel=2,
            )

    def to_json(self, **kwargs):
       
        from keras.src.saving import serialization_lib

        model_config = serialization_lib.serialize_keras_object(self)
        return json.dumps(model_config, **kwargs)

    def export(self, filepath, format="tf_saved_model"):
        
        from keras.src.export import export_lib

        export_lib.export_model(self, filepath)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.src.models.functional import Functional
        functional_config_keys = [
            "name",
            "layers",
            "input_layers",
            "output_layers",
        ]
        is_functional_config = all(
            key in config for key in functional_config_keys
        )
        argspec = inspect.getfullargspec(cls.__init__)
        functional_init_args = inspect.getfullargspec(Functional.__init__).args[
            1:
        ]
        revivable_as_functional = (
            cls in {Functional, Model}
            or argspec.args[1:] == functional_init_args
            or (argspec.varargs == "args" and argspec.varkw == "kwargs")
        )
        if is_functional_config and revivable_as_functional:
            # Revive Functional model
            # (but not Functional subclasses with a custom __init__)
            from keras.src.models.functional import functional_from_config

            return functional_from_config(
                cls, config, custom_objects=custom_objects
            )

        # Either the model has a custom __init__, or the config
        # does not contain all the information necessary to
        # revive a Functional model. This happens when the user creates
        # subclassed models where `get_config()` is returning
        # insufficient information to be considered a Functional model.
        # In this case, we fall back to provide all config into the
        # constructor of the class.
        try:
            return cls(**config)
        except TypeError as e:
            raise TypeError(
                "Unable to revive model from config. When overriding "
                "the `get_config()` method, make sure that the "
                "returned config contains all items used as arguments "
                f"in the  constructor to {cls}, "
                "which is the default behavior. "
                "You can override this default behavior by defining a "
                "`from_config(cls, config)` class method to specify "
                "how to create an "
                f"instance of {cls.__name__} from its config.\n\n"
                f"Received config={config}\n\n"
                f"Error encountered during deserialization: {e}"
            )

    def _get_variable_map(self):
        store = {}
        map_trackable_variables(self, store=store, visited_trackables=set())
        return store
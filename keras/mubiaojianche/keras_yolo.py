def compute_pycocotools_metric(y_true, y_pred, bounding_box_format):
    y_true = bounding_box.to_dense(y_true)
    y_pred = bounding_box.to_dense(y_pred)

    box_pred = y_pred["boxes"]
    cls_pred = y_pred["classes"]
    confidence_pred = y_pred["confidence"]

    gt_boxes = y_true["boxes"]
    gt_classes = y_true["classes"]

    box_pred = bounding_box.convert_format(
        box_pred, source=bounding_box_format, target="yxyx"
    )
    gt_boxes = bounding_box.convert_format(
        gt_boxes, source=bounding_box_format, target="yxyx"
    )

    total_images = gt_boxes.shape[0]

    source_ids = np.char.mod("%d", np.linspace(1, total_images, total_images))

    ground_truth = {}
    ground_truth["source_id"] = [source_ids]

    ground_truth["num_detections"] = [
        ops.sum(ops.cast(y_true["classes"] >= 0, "int32"), axis=-1)
    ]
    ground_truth["boxes"] = [ops.convert_to_numpy(gt_boxes)]
    ground_truth["classes"] = [ops.convert_to_numpy(gt_classes)]

    predictions = {}
    predictions["source_id"] = [source_ids]
    predictions["detection_boxes"] = [ops.convert_to_numpy(box_pred)]
    predictions["detection_classes"] = [ops.convert_to_numpy(cls_pred)]
    predictions["detection_scores"] = [ops.convert_to_numpy(confidence_pred)]
    predictions["num_detections"] = [
        ops.sum(ops.cast(confidence_pred > 0, "int32"), axis=-1)
    ]
    return coco.compute_pycoco_metrics(ground_truth, predictions)

# @keras_cv_export 是一个装饰器，用于注册这个类，使得可以在 Keras CV 模块中通过名称访问。
# 定义了一个名为 BoxCOCOMetrics 的类，继承自 keras.metrics.Metric。
@keras_cv_export("keras_cv.metrics.BoxCOCOMetrics")
class BoxCOCOMetrics(keras.metrics.Metric):
    # __init__ 是构造函数，用于初始化类的实例。
    def __init__(self, bounding_box_format, evaluate_freq, name=None, **kwargs):
        # 检查 kwargs 是否包含 dtype，如果没有，则默认设置为 "float32"。
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        super().__init__(name=name, **kwargs) # 调用基类的构造函数进行初始化。
        self.ground_truths = [] # 初始化一个列表来存储 ground truth 数据。
        self.predictions = [] # 初始化一个列表来存储预测数据。
        # 设置边界框格式。
        self.bounding_box_format = bounding_box_format
        self.evaluate_freq = evaluate_freq # 设置评估频率。
        self._eval_step_count = 0  # 初始化评估步数计数器。
        # 初始化一个缓存结果列表，用于存储指标结果。
        self._cached_result = [0] * len(METRIC_NAMES)
    # __new__ 是一个特殊方法，用于创建类的新实例。在这里，它调用了基类的 __new__ 方法来创建实例。
    def __new__(cls, *args, **kwargs):
        obj = super(keras.metrics.Metric, cls).__new__(cls)
        # Wrap the update_state function in a py_function and scope it to /cpu:0
        obj_update_state = obj.update_state # 保存原始的 update_state 方法。
        # 定义一个新的函数，用于在 CPU 上执行 update_state。
        def update_state_on_cpu(
            y_true_boxes,
            y_true_classes,
            y_pred_boxes,
            y_pred_classes,
            y_pred_confidence,
            sample_weight=None,
        ):
            y_true = {"boxes": y_true_boxes, "classes": y_true_classes}
            y_pred = {
                "boxes": y_pred_boxes,
                "classes": y_pred_classes,
                "confidence": y_pred_confidence,
            }
            # 确保 update_state 在 CPU 上执行。
            with tf.device("/cpu:0"):
                return obj_update_state(y_true, y_pred, sample_weight)
        obj.update_state_on_cpu = update_state_on_cpu
        # 定义一个新的函数，用于封装 update_state_on_cpu，以便在 TensorFlow 图中使用。
        def update_state_fn(self, y_true, y_pred, sample_weight=None):
            y_true_boxes = y_true["boxes"]
            y_true_classes = y_true["classes"]
            y_pred_boxes = y_pred["boxes"]
            y_pred_classes = y_pred["classes"]
            y_pred_confidence = y_pred["confidence"]
            # 收集输入参数
            eager_inputs = [
                y_true_boxes,
                y_true_classes,
                y_pred_boxes,
                y_pred_classes,
                y_pred_confidence,
            ]
            
            if sample_weight is not None:
                eager_inputs.append(sample_weight)
            # 使用 tf.py_function 在图中执行 Python 函数。
            return tf.py_function(
                func=self.update_state_on_cpu, inp=eager_inputs, Tout=[]
            )
        # 将新定义的函数作为方法绑定到对象 obj 上。
        obj.update_state = types.MethodType(update_state_fn, obj)
        # Wrap the result function in a py_function and scope it to /cpu:0
        # 保存原始的 result 方法。
        obj_result = obj.result
        # 定义一个新的函数，用于在 CPU 上执行 result。
        def result_on_host_cpu(force):
            # 确保 result 在 CPU 上执行。
            with tf.device("/cpu:0"):
                # Without the call to `constant` `tf.py_function` selects the
                # first index automatically and just returns obj_result()[0]
                return tf.constant(obj_result(force), obj.dtype)
        obj.result_on_host_cpu = result_on_host_cpu
        # 定义一个新的函数，用于封装 result_on_host_cpu，以便在 TensorFlow 图中使用。
        def result_fn(self, force=False):
            # 使用 tf.py_function 在图中执行 Python 函数。
            py_func_result = tf.py_function(
                self.result_on_host_cpu, inp=[force], Tout=obj.dtype
            )
            # 创建一个空字典来存储结果。
            result = {}
            # 遍历指标名称并将结果存储到字典中。
            for i, key in enumerate(METRIC_NAMES):
                result[self.name_prefix() + METRIC_MAPPING[key]] = (
                    py_func_result[i]
                )
            return result
        # 将新定义的函数作为方法绑定到对象 obj 上。
        obj.result = types.MethodType(result_fn, obj)
        return obj # 返回创建的对象实例。
    # 方法返回一个前缀字符串，用于标识指标名称。
    def name_prefix(self):
        if self.name.startswith("box_coco_metrics"):
            return ""
        return self.name + "_"
    # update_state 方法用于更新状态。
    def update_state(self, y_true, y_pred, sample_weight=None):
        # 增加评估步数计数器。
        self._eval_step_count += 1
        # 检查 y_true 和 y_pred 是否具有相同的 RaggedTensor 或 Dense Tensor 状态。
        if isinstance(y_true["boxes"], tf.RaggedTensor) != isinstance(
            y_pred["boxes"], tf.RaggedTensor
        ):
            # Make sure we have same ragged/dense status for y_true and y_pred
            # 将 RaggedTensor 转换为 Dense Tensor。
            y_true = bounding_box.to_dense(y_true)
            y_pred = bounding_box.to_dense(y_pred)
        # 将 ground truth 添加到列表中
        self.ground_truths.append(y_true)
        # 将预测结果添加到列表中。
        self.predictions.append(y_pred) 
        # 检查是否达到评估频率，如果是，则计算结果。
        # Compute on first step, so we don't have an inconsistent list of
        # metrics in our train_step() results. This will just populate the
        # metrics with `0.0` until we get to `evaluate_freq`.
        if self._eval_step_count % self.evaluate_freq == 0:
            self._cached_result = self._compute_result()
    # reset_state 方法用于重置状态。
    def reset_state(self):
        self.ground_truths = []
        self.predictions = []
        self._eval_step_count = 0
        self._cached_result = [0] * len(METRIC_NAMES)
    # result 方法用于获取结果。
    def result(self, force=False):
        # 如果强制计算结果，则重新计算。
        if force:
            self._cached_result = self._compute_result()
        return self._cached_result
    # compute_result 方法用于计算指标结果。
    def _compute_result(self):
        # if len(...): 检查是否有数据。
        if len(self.predictions) == 0 or len(self.ground_truths) == 0:
            return dict([(key, 0) for key in METRIC_NAMES])
        # with HidePrints(): 忽略打印输出。
        with HidePrints():  # 计算指标。
            metrics = compute_pycocotools_metric(
                _box_concat(self.ground_truths),
                _box_concat(self.predictions),
                self.bounding_box_format,
            )
        # 创建一个空列表来存储结果。
        results = []
        for key in METRIC_NAMES:
            # 存储结果，并确保结果非负。
            # Workaround for the state where there are 0 boxes in a category.
            results.append(max(metrics[key], 0.0))
        return results
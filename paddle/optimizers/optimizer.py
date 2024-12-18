class Optimizer(object): # 用于实现各种优化算法的基础类
    # 类定义及构造函数
    @imperative_base.no_grad
    def __init__(
        self,
        learning_rate,
        parameters=None,
        weight_decay=None,
        grad_clip=None,
        name=None,
    ):
        # 检查 parameters 是否为有效的输入类型
        if parameters is not None:
            # 如果 parameters 是单个张量，则抛出 TypeError。
            if isinstance(parameters, (paddle.Tensor, core.eager.Tensor)):
                raise TypeError(
                    "`parameters` argument given to the optimizer should be "
                    "an iterable of paddle Tensors, but got argument type is `{}`.".format(
                        type(parameters)
                    )
                )
            # 如果 parameters 是字典，则抛出 TypeError 并建议使用列表形式。
            if isinstance(parameters, dict):
                raise TypeError(
                    "`parameters` argument should not get dict type, "
                    "if parameter groups is needed, please set `parameters`"
                    " as list of dict"
                )
            self._parameter_list = list(parameters) # 保存对象的参数列表
        else:
            self._parameter_list = None
        
        self._name = name # 设置优化器的名称
        # 检查动态图模式下的参数列表：
        if framework._non_static_mode():
            # 如果参数列表为空，则抛出 AttributeError。
            if self._parameter_list is None:
                raise AttributeError(
                    "parameters argument given to the Optimizer should not be None in dygraph mode."
                )
            # 如果设置了 weight_decay 并且存在自定义正则化项，则发出警告信息。
            if weight_decay is not None:
                if not isinstance(self._parameter_list[0], dict):
                    for param in self._parameter_list:
                        if (
                            hasattr(param, 'regularizer')
                            and param.regularizer is not None
                        ):
                            logging.info(
                                "If regularizer of a Parameter has been set by 'paddle.ParamAttr' or 'static.WeightNormParamAttr' already. "
                                "The weight_decay[%s] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                                % weight_decay.__str__()
                            )
                            break
        # 检查学习率是否为有效的类型。
        if not isinstance(learning_rate, (float, LRScheduler)):
            raise TypeError(
                "learning rate should be float or LRScheduler, got %s here"
                % type(learning_rate)
            )
        # 检查梯度裁剪是否为有效的类型。
        if grad_clip is not None:
            if not isinstance(grad_clip, GradientClipBase):
                raise TypeError(
                    "'grad_clip' should be an instance of GradientClipBase's derived class"
                )
        # 设置正则化项。
        if isinstance(weight_decay, float): 
            from paddle.fluid.regularizer import L2Decay
            self.regularization = L2Decay(weight_decay)
        else:
            self.regularization = weight_decay
        # 设置梯度裁剪和学习率。
        self._grad_clip = grad_clip
        self._learning_rate = learning_rate

        self._dtype = None
        # Infer the dtype form parameter
        if self._parameter_list:
            # 如果参数列表包含字典，则检查 params 键是否存在
            if isinstance(self._parameter_list[0], dict):
                for param_group in self._parameter_list:
                    assert (
                        'params' in param_group
                    ), 'params should be set in parameters if parameter groups are optimized in different options'
                self._dtype = self._parameter_list[0]['params'][0].dtype
            else: # 设置self._dtype为第一个参数的类型
                self._dtype = self._parameter_list[0].dtype

        # 初始化学习率映射、累加器、辅助对象等。
        self._learning_rate_map = dict()
        # Dictionary of accumulators. Some optimizer subclasses need to
        # allocate and manage extra tensors associated with the parameters
        # to train. These tensors are called accumulators.
        # {accum_name : { paramter_name : accumulator_for_parameter, ...}, ...}
        self._accumulators = defaultdict(lambda: dict())
        self.helper = None
        self._opti_name_list = []
        self._accumulators_holder = {}
        self._param_device_map = dict()
        self.clear_gradients = self.clear_grad
        self._default_dict = {
            'weight_decay': self.regularization,
            'grad_clip': self._grad_clip,
        }
        
        self._param_groups = [] # 初始化参数组。
        if self._parameter_list and isinstance(self._parameter_list[0], dict):
            for param_group in self._parameter_list:
                self._add_param_group(param_group.copy())
        else:
            self._param_groups = self._parameter_list
        
        # NOTE: Multi Tensor: Pass in all parameters and gradients to the op kernel of the Optimizer at one time for updating for dygraph mode.
        # Optimizer support list: [ paddle.optimizer.Momentum, paddle.optimizer.Adam].
        self._use_multi_tensor = None # 初始化多张量支持和辅助变量。
        self._param_dict = self._create_multi_tensor_dict()
        self._auxiliary_vars = {}
    # 设置和获取辅助变量。
    def _set_auxiliary_var(self, key, val):
        self._auxiliary_vars[key] = val
    
    def _create_multi_tensor_dict(self):
        n = len(self._param_groups) if self._param_groups is not None else 1
        return {
            'FP32_LODTensor': [[] for _ in range(n)],
            'FP16_LODTensor': [[] for _ in range(n)],
        }

    def _get_auxiliary_var(self, key):
        return self._auxiliary_vars.get(key, None)
    # 保存优化器的状态字典： 包括累积器、主权重（如果有）和学习率调度器的状态
    @framework.dygraph_only
    def state_dict(self):
        state_dict = {}
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                state_dict[var_tmp.name] = var_tmp
        # if has master weight and then save master weight
        if hasattr(self, "_master_weights"):
            if len(self._master_weights) != 0:
                state_dict["master_weights"] = self._master_weights
        # global step if use lr decay
        if isinstance(self._learning_rate, LRScheduler):
            state_dict["LR_Scheduler"] = self._learning_rate.state_dict()
        return state_dict
    # 加载优化器的状态字典：加载学习率调度器的状态。加载主权重（如果有）。验证并加载累积器的状态。
    # @framework.dygraph_only 装饰器的作用是：限定方法适用范围：标记该方法或函数仅在动态图模
    # 式下可用。如果试图在静态图模式下调用带有此装饰器的方法，将会抛出异常。通过显式地标记方法，
    # 使代码更加清晰，便于维护和理解。
    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        if isinstance(self._learning_rate, LRScheduler):
            self._learning_rate.set_state_dict(state_dict["LR_Scheduler"])

        # 在 state_dict = state_dict.copy() 中，实际上是将 state_dict 的浅拷贝赋值给了同一
        # 个变量名 state_dict。这样做的目的是为了后续可以安全地修改 state_dict，而不会影响到传入
        # 的原始对象。如果原始对象没有其他引用，那么原始的字典可能会被垃圾回收机制清理。
        state_dict = state_dict.copy()
        if "LR_Scheduler" in state_dict:
            state_dict.pop("LR_Scheduler")
        if "master_weights" in state_dict:
            if hasattr(self, "_master_weights"):
                self._master_weights = state_dict["master_weights"]
            state_dict.pop("master_weights")
        self._accumulators_holder = state_dict
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                assert (
                    var_tmp.name in state_dict
                ), "optimizer Tensor {} not found".format(var_tmp.name)
                var = var_tmp.value()
                tensor = var.get_tensor()
                model_np = np.array(tensor)

                load_para = state_dict[var_tmp.name]

                if isinstance(load_para, Variable):
                    load_para_np = load_para.numpy()
                elif isinstance(load_para, core.VarBase):
                    load_para_np = load_para.numpy()
                elif isinstance(load_para, np.ndarray):
                    load_para_np = load_para
                else:
                    raise RuntimeError(
                        "State dict type {} not supprt".format(
                            str(type(load_para))
                        )
                    )

                assert (
                    model_np.shape == load_para_np.shape
                ), "Parameter shape not match, Dygraph Parameter [ {} ] need tensor with shape {} but load tensor with shape {}".format(
                    model_np.name, model_np.shape, load_para_np.shape
                )

                assert (
                    model_np.dtype == load_para_np.dtype
                ), "Parameter dtype not match, Dygraph Parameter [ {} ] need tensor with dtype {}  but load tensor with dtype {}".format(
                    model_np.name, model_np.dtype, load_para_np.dtype
                )

                tensor.set(load_para_np, framework._current_expected_place())

    def get_opti_var_name_list(self):
        return self._opti_name_list

    def _create_global_learning_rate(self):
        # lr var can't be float16, for pure fp16 training, should extra handle the dtype for lr
        _lr_dtype = (
            paddle.get_default_dtype() if self._dtype is None else self._dtype
        )
        _lr_dtype = (
            paddle.float32
            if (
                paddle.get_default_dtype() != "float16"
                and _lr_dtype == paddle.float16
            )
            else _lr_dtype
        )
        if isinstance(self._learning_rate, LRScheduler):
            lr_var = self._global_learning_rate()
            # only create global lr_var once
            if not isinstance(lr_var, framework.Variable):
                lr_name = unique_name.generate('learning_rate')
                self._learning_rate._var_name = lr_name
                lr_var = self.helper.create_global_variable(
                    name=lr_name,
                    shape=[1],
                    persistable=True,
                    stop_gradient=True,
                    dtype=_lr_dtype,
                )
                main_prog = framework.default_main_program()
                main_prog.lr_sheduler = self._learning_rate
                main_prog.lr_var = lr_var

                self._learning_rate_map[
                    framework.default_main_program()
                ] = lr_var

            lr_value = float(self._learning_rate())
            self.helper.set_variable_initializer(
                lr_var, initializer=Constant(value=lr_value)
            )
        elif isinstance(self._learning_rate, float):
            # only create global lr_var once
            lr = self._global_learning_rate()
            if isinstance(lr, framework.Variable):
                return
            else:
                self._learning_rate_map[
                    framework.default_main_program()
                ] = layers.create_global_var(
                    name=unique_name.generate("learning_rate"),
                    shape=[1],
                    value=float(self._learning_rate),
                    dtype=_lr_dtype,
                    persistable=True,
                )

    @framework.dygraph_only
    def set_lr(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(
                "The type of 'value' in optimizer.set_lr must be float, but received %s."
                % (type(value))
            )
        if isinstance(self._learning_rate, LRScheduler):
            raise RuntimeError(
                "optimizer's learning rate can't be LRScheduler when invoke this API, because this will lead to conflict."
            )
        self._learning_rate = float(value)
        current_lr = self._global_learning_rate()
        if current_lr is not None:
            if in_dygraph_mode():
                place = _current_expected_place()
                _C_ops.full_(
                    current_lr,
                    list(current_lr.shape),
                    float(value),
                    current_lr.dtype,
                    place,
                )

            elif _in_legacy_dygraph():
                _legacy_C_ops.fill_constant(
                    current_lr,
                    'value',
                    float(value),
                    'dtype',
                    current_lr.dtype,
                    'shape',
                    list(current_lr.shape),
                )
            else:
                global_block = framework.default_main_program().global_block()
                global_block.append_op(
                    type='fill_constant',
                    outputs={'Out': [current_lr]},
                    attrs={
                        'dtype': current_lr.dtype,
                        'shape': list(current_lr.shape),
                        'value': float(value),
                    },
                    stop_gradient=True,
                )

    def get_lr(self):
        if isinstance(self._learning_rate, float):
            return self._learning_rate
        else:
            return self._learning_rate()

    def _global_learning_rate(self, program=None):
        """
        get global decayed learning rate
        :return:
        """
        if program is None:
            program = framework.default_main_program()
        return self._learning_rate_map.get(program, None)

    def _append_optimize_op(self, block, param_and_grad):
        """append optimize operator to block and return all the added optimize_op"""
        raise NotImplementedError(
            "Class \"Optimizer\" connot be used directly as an optimizer, please use its subclasses such as \"Adam\""
        )

    def _create_param_lr(self, param_and_grad):
        # create learning rate tensor for every parameter
        param = param_and_grad[0]
        if hasattr(param, 'optimize_attr'):
            param_lr = param.optimize_attr['learning_rate']
            if type(param_lr) == Variable:
                return param_lr
            else:
                if param_lr == 1.0:
                    return self._global_learning_rate()
                else:
                    with default_main_program()._lr_schedule_guard(
                        is_with_opt=True
                    ), framework.name_scope('scale_with_param_lr'):
                        return self._global_learning_rate() * param_lr
        else:
            return self._global_learning_rate()

    def _create_accumulators(self, block, parameters):
       
        pass

    def _finish_update(self, block, parameters_and_grads):
        
        pass

    def _add_accumulator(
        self,
        name,
        param,
        dtype=None,
        fill_value=0.0,
        shape=None,
        type=None,
        device=None,
    ):
       
        if self._name is not None:
            name = self._name + "_" + name
        if (
            name in self._accumulators
            and param.name in self._accumulators[name]
        ):
            if framework._non_static_mode():
                return self._accumulators[name][param.name]
            raise Exception(
                "Accumulator {} already exists for parameter {}".format(
                    name, param.name
                )
            )
        if shape == None:
            shape = param.shape
        assert isinstance(self.helper, LayerHelper)

        var_name = param.name + "_" + name
        var_name = unique_name.generate(var_name)
        self._opti_name_list.append(var_name)

        var = self.helper.create_global_variable(
            name=var_name,
            persistable=True,
            dtype=dtype or param.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR
            if framework._in_eager_without_dygraph_check()
            else (param.type if type is None else type),
            shape=shape,
            belong_to_optimizer=True,
        )
        if device is None:
            device = self._get_device_for_param(param.name)
        with device_guard(device):
            self.helper.set_variable_initializer(
                var, initializer=Constant(value=float(fill_value))
            )

        if framework._non_static_mode():
            if len(self._accumulators_holder) > 0:
                assert (
                    var_name in self._accumulators_holder
                ), "Optimizer set error, {} should in state dict".format(
                    var_name
                )
                var.set_value(self._accumulators_holder[var_name])

        self._accumulators[name][param.name] = var
        return var

    def _get_accumulator(self, name, param):
        
        if self._name is not None:
            name = self._name + "_" + name
        if (
            name not in self._accumulators
            or param.name not in self._accumulators[name]
        ):
            raise Exception(
                "Accumulator {} does not exist for parameter {}".format(
                    name, param.name
                )
            )
        return self._accumulators[name][param.name]

    def _update_param_device_map(self, parameters_and_grads, target_block):
        for param_and_grad in parameters_and_grads:
            if param_and_grad[0].stop_gradient is False:
                param_name = param_and_grad[0].name
                ops = target_block.ops
                device_attr_name = (
                    core.op_proto_and_checker_maker.kOpDeviceAttrName()
                )
                for op in ops:
                    input_arg_names = op.input_arg_names
                    if param_name in input_arg_names:
                        self._param_device_map[param_name] = op.attr(
                            device_attr_name
                        )
                        break

    def _get_device_for_param(self, param_name):
        device = None
        if param_name in self._param_device_map:
            device = self._param_device_map[param_name]
        return device

    def _create_optimization_pass(
        self, parameters_and_grads, param_group_idx=0
    ):
       
        # This is a default implementation of create_optimization_pass that
        # can be shared by most optimizers. This implementation assumes that
        # the subclass will implement the _append_optimize_op method and the
        #  _initialize_tensors method. The subclass can extend the
        # _create_accumulators method if it needs to create accumulators
        # for parameters and extend _finish_update method to add custom ops.

        # Allways called under program_guard use global block as loss block
        # But if current block is in control flow, append optimize op in the
        # grad block of current block

        global_block = framework.default_main_program().global_block()
        target_block = global_block
        current_block = framework.default_main_program().current_block()
        if current_block.idx != global_block.idx:
            assert (
                current_block.backward_block_idx != -1
            ), "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx
            ]

        start = len(target_block.ops)
        self.helper = LayerHelper(self.__class__.__name__)

        self._create_global_learning_rate()

        # NOTE: Multi Tensor support [ Momentum, Adam ] for dygraph mode
        if self._use_multi_tensor and self.__class__.__name__ in [
            'Momentum',
            'Adam',
        ]:
            if (
                len(self._param_dict['FP32_LODTensor'][param_group_idx]) == 0
                and len(self._param_dict['FP16_LODTensor'][param_group_idx])
                == 0
            ):
                if isinstance(parameters_and_grads, list):
                    assert param_group_idx == 0
                    self._multi_tensor_init(
                        target_block,
                        [
                            p[0]
                            for p in parameters_and_grads
                            if not p[0].stop_gradient
                        ],
                        param_group_idx,
                    )
                else:
                    self._update_param_group(parameters_and_grads)
                    self._multi_tensor_init(
                        target_block,
                        [
                            p[0]
                            for p in parameters_and_grads['params']
                            if not p[0].stop_gradient
                        ],
                        param_group_idx,
                    )
            if framework._non_static_mode():
                self._append_optimize_multi_tensor_op(
                    target_block,
                    parameters_and_grads,
                    param_group_idx=param_group_idx,
                )
            else:
                self._update_param_device_map(
                    parameters_and_grads, target_block
                )
                # NOTE: Multi Tensor requires all parameters to be in the same device and program.
                # param_grad_list = [p_0,g_0,p_1,g_1,....]
                param_grad_list = []
                for param_and_grad in parameters_and_grads:
                    if (
                        not param_and_grad[0].stop_gradient
                        and param_and_grad[1] is not None
                    ):
                        param_grad_list.append(param_and_grad[0])
                        param_grad_list.append(param_and_grad[1])
                with param_grad_list[0].block.program._optimized_guard(
                    param_grad_list
                ), name_scope("optimizer"):
                    device = self._get_device_for_param(param_grad_list[0].name)
                    with device_guard(device):
                        self._append_optimize_multi_tensor_op(
                            target_block,
                            parameters_and_grads,
                            param_group_idx=param_group_idx,
                        )
        else:
            if not framework._non_static_mode():
                params_grads_device_map = (
                    parameters_and_grads['params']
                    if isinstance(parameters_and_grads, dict)
                    else parameters_and_grads
                )
                self._update_param_device_map(
                    params_grads_device_map, target_block
                )

            if isinstance(parameters_and_grads, list):
                self._create_accumulators(
                    target_block,
                    [
                        p[0]
                        for p in parameters_and_grads
                        if not p[0].stop_gradient
                    ],
                )
            else:
                params_acc_dict = parameters_and_grads.copy()
                params_acc_dict['params'] = [
                    p[0]
                    for p in params_acc_dict['params']
                    if not p[0].stop_gradient
                ]
                self._create_accumulators(target_block, params_acc_dict)

            if framework._non_static_mode():
                if isinstance(parameters_and_grads, list):
                    for param_and_grad in parameters_and_grads:
                        if param_and_grad[1] is None:
                            continue
                        if param_and_grad[0].stop_gradient is False:
                            self._append_optimize_op(
                                target_block, param_and_grad
                            )
                else:
                    for param_and_grad in parameters_and_grads['params']:
                        if param_and_grad[1] is None:
                            continue
                        if param_and_grad[0].stop_gradient is False:
                            param_grad_dict = dict()
                            param_grad_dict['params'] = param_and_grad
                            param_grad_dict.update(
                                {
                                    k: v
                                    for k, v in parameters_and_grads.items()
                                    if k != 'params'
                                }
                            )
                            self._append_optimize_op(
                                target_block, param_grad_dict
                            )
            else:
                for param_and_grad in parameters_and_grads:
                    if param_and_grad[1] is None:
                        continue
                    with param_and_grad[0].block.program._optimized_guard(
                        param_and_grad
                    ), name_scope("optimizer"):
                        if param_and_grad[0].stop_gradient is False:
                            device = self._get_device_for_param(
                                param_and_grad[0].name
                            )
                            with device_guard(device):
                                optimize_op = self._append_optimize_op(
                                    target_block, param_and_grad
                                )

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self._finish_update(target_block, parameters_and_grads)

        end = len(target_block.ops)
        return target_block._slice_ops(start, end)

    def _append_dgc_ops(self, param_and_grad):
        pass

    def backward(
        self,
        loss,
        startup_program=None,
        parameters=None,
        no_grad_set=None,
        callbacks=None,
    ):
        act_no_grad_set = None
        if framework._non_static_mode():
            pass
        else:
            act_no_grad_set = self._get_no_grad_set(loss, no_grad_set)

        # Infer dtype by loss if None
        if self._dtype is None:
            self._dtype = loss.dtype

        if framework._non_static_mode():
            parameter_list = parameters if parameters else self._parameter_list

            params_grads = []
            for param in parameter_list:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    # create gradient tensor
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))
        else:
            if callbacks is None:
                callbacks = [error_clip_callback]
            else:
                assert isinstance(callbacks, list)
            program = loss.block.program
            assert len(loss.shape) == 1 and loss.shape[0] == 1, (
                "The loss.shape should be (1L,), but the current loss.shape is {}. "
                "Maybe that you should call paddle.mean to process the current loss.".format(
                    loss.shape
                )
            )
            parameter_list = parameters if parameters else self._parameter_list
            with program_guard(program, startup_program):
                from paddle.incubate.autograd.utils import prim_enabled

                if prim_enabled():
                    params_grads = append_backward_new(
                        [loss], parameter_list, act_no_grad_set, callbacks
                    )
                else:
                    params_grads = append_backward(
                        loss, parameter_list, act_no_grad_set, callbacks
                    )
                # Note: since we can't use all_reduce_op now,
                #  dgc_op should be the last op of one grad.
                self._append_dgc_ops(params_grads)
        return params_grads

    def apply_gradients(self, params_grads):
        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        # 'optimizer(grad_clip)' or 'set_gradient_clip'
        if self._grad_clip is not None:
            params_grads = self._grad_clip(params_grads)
        else:

            params_grads = append_gradient_clip_ops(params_grads)

        # Add regularization if any
        params_grads = self.append_regularization_ops(
            params_grads, self.regularization
        )

        optimize_ops = self._create_optimization_pass(params_grads)
        return optimize_ops

    def _apply_optimize(
        self, loss, startup_program, params_grads, param_group_idx=0
    ):
       
        if framework._non_static_mode():
            with program_guard(
                framework.default_main_program(),
                framework.default_startup_program(),
            ):
                if isinstance(params_grads, list):
                    if self._grad_clip is not None:
                        params_grads = self._grad_clip(params_grads)
                    params_grads = self.append_regularization_ops(
                        params_grads, self.regularization
                    )
                else:
                    grad_clip = params_grads['grad_clip']
                    if grad_clip is not None:
                        params_grads['params'] = grad_clip(
                            params_grads['params']
                        )

                    params_grads['params'] = self.append_regularization_ops(
                        params_grads['params'], self.regularization
                    )
                optimize_ops = self._create_optimization_pass(
                    params_grads, param_group_idx=param_group_idx
                )
        else:
            assert param_group_idx == 0
            program = loss.block.program
            with program_guard(program, startup_program):
                optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def _create_regularization_of_grad(self, param, grad, regularization=None):
       
        # If no gradient or no regularization is specified,  then we don't need to do anything
        if grad is None or (
            (
                not hasattr(param, 'regularizer')
                or (hasattr(param, 'regularizer') and param.regularizer is None)
            )
            and regularization is None
        ):
            return grad
        regularization_term = None
        if hasattr(param, 'regularizer') and param.regularizer is not None:
            # Add variable for regularization term in grad block
            regularization_term = param.regularizer(param, grad, grad.block)
        elif regularization is not None:
            regularization_term = regularization(param, grad, grad.block)

        assert regularization_term is not None

        if framework.in_dygraph_mode():
            return _C_ops.add_n([grad, regularization_term])
        elif framework._in_legacy_dygraph():
            return _legacy_C_ops.sum([grad, regularization_term])

        new_grad = grad
        if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
            # FIXME(zcd): If the grad is SELECTED_ROWS, after regularization,
            # the grad's type and name will be changed. But the gradient's name
            # is used in ParallelExecutor Reduce mode, so I add a flag for
            # the new_grad here.
            new_grad = grad.block.create_var(
                name=grad.name + core.kNewGradSuffix(),
                dtype=param.dtype,
                shape=param.shape,
                lod_level=param.lod_level,
                type=core.VarDesc.VarType.LOD_TENSOR,
            )

        inputs = {"X": [grad, regularization_term]}
        outputs = {"Out": [new_grad]}
        grad.block.append_op(type='sum', inputs=inputs, outputs=outputs)

        return new_grad

    def append_regularization_ops(
        self, parameters_and_grads, regularization=None
    ):
        params_and_grads = []
        if framework._non_static_mode():
            for param, grad in parameters_and_grads:
                new_grad = self._create_regularization_of_grad(
                    param, grad, regularization
                )
                params_and_grads.append((param, new_grad))
        else:
            repeate_regularizer = False
            with framework.name_scope('regularization'):
                for param, grad in parameters_and_grads:
                    if (
                        not repeate_regularizer
                        and param.regularizer is not None
                        and regularization is not None
                    ):
                        repeate_regularizer = True
                        logging.info(
                            "If regularizer of a Parameter has been set by 'fluid.ParamAttr' or 'fluid.WeightNormParamAttr' already. "
                            "The Regularization[%s] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                            % regularization.__str__()
                        )
                    with param.block.program._optimized_guard([param, grad]):
                        new_grad = self._create_regularization_of_grad(
                            param, grad, regularization
                        )
                        params_and_grads.append((param, new_grad))
        return params_and_grads

    def _get_no_grad_set(self, loss, no_grad_set=None):
        no_grad_set = _get_no_grad_set_name(no_grad_set)
        parameters = loss.block.program.global_block().all_parameters()
        param_no_trainable = set(
            [param.name for param in parameters if param.stop_gradient is True]
        )
        # If the parameter is no trainable, it should not have a gradient.
        no_grad_set.update(param_no_trainable)

        return no_grad_set

    @framework.dygraph_only
    def clear_grad(self, set_to_zero=True):
        param_list = []
        if self._parameter_list is None or not isinstance(
            self._parameter_list[0], dict
        ):
            for p in self._parameter_list:
                if not p.stop_gradient:
                    param_list.append(p)
        else:
            for param_group in self._param_groups:
                for p in param_group['params']:
                    if not p.stop_gradient:
                        param_list.append(p)

        if _in_eager_without_dygraph_check():
            for p in param_list:
                p.clear_gradient(set_to_zero)
        else:
            core.clear_gradients(param_list, set_to_zero)

    @imperative_base.no_grad
    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):
        assert isinstance(loss, Variable), "The loss should be an Tensor."

        parameter_list = parameters if parameters else self._parameter_list

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameters=parameter_list,
            no_grad_set=no_grad_set,
        )

        optimize_ops = self._apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads
        )

        return optimize_ops, params_grads

    @imperative_base.no_grad
    @framework.dygraph_only
    def step(self):
        if not isinstance(self._param_groups[0], dict):
            params_grads = []
            for param in self._param_groups:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))

            self._apply_optimize(
                loss=None,
                startup_program=None,
                params_grads=params_grads,
                param_group_idx=0,
            )

        else:
            # optimize parameters in groups
            for idx, param_group in enumerate(self._param_groups):
                params_grads = defaultdict(lambda: list())
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        grad_var = param._grad_ivar()
                        params_grads['params'].append((param, grad_var))
                params_grads.update(
                    {k: v for k, v in param_group.items() if k != 'params'}
                )
                self._apply_optimize(
                    loss=None,
                    startup_program=None,
                    params_grads=params_grads,
                    param_group_idx=idx,
                )

    def _add_param_group(self, param_group):
        
        params = param_group['params']
        if isinstance(params, Parameter):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters should be in ordered collections,"
                "but received set, please use list instead."
            )
        else:
            param_group['params'] = list(params)

        # Update optimization options for each groups
        for k, v in self._default_dict.items():
            param_group.setdefault(k, v)

        param_set = set()
        for group in self._param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group"
            )

        for param in param_group['params']:
            weight_decay = param_group['weight_decay']
            if isinstance(weight_decay, float):
                from ..fluid.regularizer import L2Decay

                regularization = L2Decay(weight_decay)
            else:
                regularization = weight_decay
            param.regularizer = regularization
            param.optimize_attr['learning_rate'] = param_group.get(
                'learning_rate', 1.0
            )

        self._param_groups.append(param_group)

    def _update_param_group(self, parameters):
        
        pass

    @framework.dygraph_only
    def _multi_tensor_init(self, target_block, parameters, param_group_idx):
       
        pass

    @framework.dygraph_only
    def _append_optimize_multi_tensor_op(
        self, target_block, parameters_and_grads, param_group_idx
    ):
       
        pass

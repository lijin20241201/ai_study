class Program(object):
    def __init__(self):
        # 成员变量 self.desc：core.ProgramDesc 是一个 C++ 类，它表示程序的底层描述符。self.desc
        # 是 Python 层面向用户的描述符，它存储了整个程序的底层信息。
        self.desc = core.ProgramDesc()
        # self.blocks 是一个列表，用于存储程序中的所有块（Block）。每个程序至少有一个块，这里初始化了一
        # 个空的 Block。
        self.blocks = [Block(self, 0)]
        self.current_block_idx = 0# 当前激活的块的索引，默认为 0，即第一个块。
        global global_prog_seed
        self._seed = global_prog_seed # self._seed 是一个全局种子，用于随机数生成的一致性。
        # self._current_role 表示当前操作的角色，默认为前向传播（OpRole.Forward）
        self._current_role = core.op_proto_and_checker_maker.OpRole.Forward
        # self.__op_role_var 是一个列表，用于存储当前操作角色相关的变量。
        self.__op_role_var = []
        # 接下来是一些与分布式训练相关的变量：
        # _is_distributed = True if under distributed training
        self._is_distributed = False # 标记程序是否处于分布式训练模式。
        # _is_chief = True 
        self._is_chief = False # 标记当前训练者是否为主训练者（通常是编号为 0 的训练者）。
        # _parameters_on_pservers records all the parameters distributed on parameter servers.
        # 存储分布在参数服务器上的所有参数。
        self._parameters_on_pservers = None
        # _endpoints is a list about parameter servers ip:port, such as ["ip:port","ip:port"]
        self._endpoints = []  # 参数服务器的 IP 和端口列表。
        # if current role is parameter server, the _ps_endpoint is its "ip:port"
        self._ps_endpoint = None # 如果当前角色是参数服务器，则存储其 IP 和端口。
        # trainers_endpoints, it is used for distribution.
        self._trainers_endpoints = [] # 训练者的 IP 和端口列表，用于分布式训练。
        # 分布式的查找表名称。
        self._distributed_lookup_table = None 

        # 接着是一些与优化相关的变量：
        # 是否启用深梯度压缩（Deep Gradient Compression）或 LAMB 优化器。
        self._enable_dgc = False
        self._use_lamb = False
        # NCCL 通信组的数量，默认为 1。
        self._nccl_comm_num = 1
        # 是否使用分层 AllReduce，以及分层 AllReduce 的内部节点数量。
        self._use_hierarchical_allreduce = False
        self._hierarchical_allreduce_inter_nranks = 0

        # if this program has been optimized by distributed optimizer
        # fleet_opt will be given a value
        # 如果程序已经被分布式优化器优化过，则会赋值。
        self._fleet_opt = None
        self._program_config = None
        # 如果程序已经被管道优化器解析过，则会赋值。
        # assigned if this program has been parsed by a pipeline optimizer
        self._pipeline_opt = None
        # 如果程序已经被异构管道参数服务器优化器解析过，则会赋值。
        # assigned if this program has been parsed by a heter pipeline parameter server optimizer
        self._heter_pipeline_opt = None
        # 添加梯度的次数。
        # appending gradients times
        self._appending_grad_times = 0
        # 自动生成的用于自动检查点的标识符。
        # identifier for auto checkpoint
        self._auto_checkpoint_name = unique_name.generate(
            "__auto_checkpoint_program__"
        )

        # compiled program, i.e. Graph
        self._graph = None # 编译后的程序图。
        # 标记是否为启动程序。
        self._is_start_up_program_ = False
    # _find_var_class_kwargs 方法的主要作用是从新的描述符中提取变量的元数据，并准备好创建新的变量实例所
    # 需的参数。这使得在重建程序时能够保留变量的所有属性，并确保新创建的变量与旧变量保持一致
    def _find_var_class_kwargs(self, new_desc):
        # NOTE: not all variables support shape/dtype/lod_level methods.
        # For example: RAW, STEP_SCOPES, etc.
        # 内部函数 get_var_desc_attr_or_none
        def get_var_desc_attr_or_none(var_desc, attr_name, allowed_types):
            # 该函数用于从 var_desc 中获取指定的属性，如果 var_desc 的类型在 allowed_types 
            # 列表中，则返回该属性的值；否则返回 None。
            # 主要用于处理不同类型变量的兼容性问题，因为并非所有类型的变量都支持 shape、dtype 和 l
            # od_level 等方法。
            if var_desc.type() in allowed_types:
                return getattr(var_desc, attr_name)()
            else:
                return None
        # 保存当前程序的描述符，以备后续使用。
        old_desc = self.desc
        # 初始化一个列表 all_new_vars 用于存储所有新变量的元数据。
        all_new_vars = []
        # 获取新描述符 new_desc 中的块数量。
        block_num = new_desc.num_blocks()
        # 遍历新描述符中的每个块。
        for idx in range(block_num):
            # 如果索引 idx 大于当前程序中的块数量，则创建新的块。
            if idx > (len(self.blocks) - 1):
                self._create_block()
            # 获取新描述符中的当前块。
            new_block_desc = new_desc.block(idx)
            all_new_vars.append([])
            # 初始化一个列表 block_new_vars 用于存储当前块的新变量元数据。
            block_new_vars = all_new_vars[-1]
            # 遍历每个变量描述符
            for new_var_desc in new_block_desc.all_vars():
                if self.blocks[idx].has_var(new_var_desc.name()):
                    old_var = self.blocks[idx].var(new_var_desc.name())
                else:
                    old_var = None
                # 遍历当前块中的每个变量描述符
                # 检查当前块是否已经存在具有相同名称的变量，如果存在，则获取该变量；
                # 否则设置为 None。
                # 构造一个字典 kwargs，包含新变量的基本属性，如类型、名称、形状、数据类型、LOD 级别等。
                kwargs = {
                    'type': new_var_desc.type(),
                    'name': new_var_desc.name(),
                    'shape': get_var_desc_attr_or_none(
                        new_var_desc,
                        "shape",
                        [
                            core.VarDesc.VarType.LOD_TENSOR,
                            core.VarDesc.VarType.SELECTED_ROWS,
                            core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                        ],
                    ),
                    'dtype': get_var_desc_attr_or_none(
                        new_var_desc,
                        "dtype",
                        [
                            core.VarDesc.VarType.LOD_TENSOR,
                            core.VarDesc.VarType.SELECTED_ROWS,
                            core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                        ],
                    ),
                    'lod_level': get_var_desc_attr_or_none(
                        new_var_desc,
                        "lod_level",
                        [
                            core.VarDesc.VarType.LOD_TENSOR,
                            core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                        ],
                    ),
                    # 对于 error_clip、stop_gradient、is_data 和 belong_to_optimizer，只有当旧变量存
                    # 在时才从旧变量中获取这些属性，否则设置默认值。
                    'error_clip': old_var.error_clip
                    if old_var is not None
                    else None,
                    'stop_gradient': old_var.stop_gradient
                    if old_var is not None
                    else False,
                    'is_data': old_var.is_data
                    if old_var is not None
                    else False,
                    'need_check_feed': new_var_desc.need_check_feed(),
                    'belong_to_optimizer': old_var.belong_to_optimizer
                    if old_var is not None
                    else False,
                }
                # 如果旧变量是一个 Parameter 类型，则更新 kwargs 字典以包含更多属性。
                if isinstance(old_var, Parameter):
                    kwargs.update(
                        {
                            'trainable': old_var.trainable,
                            'optimize_attr': old_var.optimize_attr,
                            'regularizer': old_var.regularizer,
                            'do_model_average': old_var.do_model_average,
                            'need_clip': old_var.need_clip,
                            'is_distributed': old_var.is_distributed,
                            'is_parameter': old_var.is_parameter,
                        }
                    )
                    # 将构造的变量元数据添加到 block_new_vars 列表中。
                    block_new_vars.append(
                        {
                            'class': Parameter,
                            'kwargs': copy.deepcopy(kwargs),
                        }
                    )
                else:
                    # 对于其他类型的变量，添加 persistable 属性。
                    kwargs['persistable'] = new_var_desc.persistable()
                    # 将构造的变量元数据添加到 block_new_vars 列表中
                    block_new_vars.append(
                        {
                            'class': Variable,
                            'kwargs': copy.deepcopy(kwargs),
                        }
                    )
        # 返回一个包含所有新变量元数据的列表。
        return all_new_vars

    def _rebuild_from_desc(self, desc):
        # 调用 _find_var_class_kwargs 方法，根据给定的描述符 desc 获取所有新变量的类和初始化参数。
        all_new_vars = self._find_var_class_kwargs(desc)
        # 获取新描述符中的块数量。断言新描述符中的块数量等于 _find_var_class_kwargs 
        # 方法返回的变量列表长度,断言新描述符中的块数量等于当前程序描述符中的块数量。
        block_num = desc.num_blocks()
        assert block_num == len(all_new_vars)
        assert block_num == self.desc.num_blocks()
        # 清空每个块中的变量和操作。
        # clear old blocks and desc
        for idx in range(block_num):
            block = self.blocks[idx]
            block.vars.clear()
            block.ops.clear()
        # 将新描述符中的块移动到当前块的描述符中。
        for idx in range(block_num):
            block_desc = self.blocks[idx].desc
            new_block_desc = desc.block(idx)
            block_desc._move_from(new_block_desc)
        # 删除不再需要的描述符对象 desc。
        del desc
        # 遍历每个块，并为每个块添加新变量。
        # add new vars first
        for idx in range(block_num):
            block = self.blocks[idx]
            for new_var in all_new_vars[idx]:
                clazz = new_var['class']
                kwargs = new_var['kwargs']
                kwargs['block'] = block
                # 使用类 clazz 和参数 kwargs 创建新变量实例，并将
                # block 作为参数传递。
                clazz(**kwargs)
        # 遍历每个块，并为每个块附加操作。
        # then append op
        for idx in range(block_num):
            block = self.blocks[idx]
            block_desc = self.desc.block(idx)
            for op_idx in range(block_desc.op_size()):
                op_desc = block_desc.op(op_idx)
                # 使用 Operator 类创建新操作实例，并将其添加到块的操作列表中。
                op = Operator(block=block, desc=op_desc)
                block.ops.append(op)
    # 设置全局种子 global_prog_seed，并将其赋值给当前程序的种子 self._seed。
    def global_seed(self, seed=0):
        global global_prog_seed
        global_prog_seed = seed
        self._seed = global_prog_seed

    @property
    def _op_role(self):
     
        return self._current_role
    # 提供了对 _current_role 属性的访问和修改方法。
    @_op_role.setter
    def _op_role(self, role):
        self._current_role = role
    # 提供了对 _op_role_var 属性的访问方法。
    @property
    def _op_role_var(self):
        return self.__op_role_var
    # 一个上下文管理器，用于临时改变 _current_role 为 OpRole.Backward。
    # 假设在一个训练循环中，我们需要先执行前向传播，然后进行反向传播，最后进行优化步骤。我们可以这样使用这些
    # 上下文管理器,例如，在 _optimized_guard 方法中，我们设置当前角色为 Optimize，并将相关的参数和梯度变量
    # 存入 __op_role_var 列表中。当优化过程完成之后，我们恢复之前的 current_role 和 __op_role_var，这样就可
    # 以确保不会影响到其他阶段的操作
    # 在使用生成器函数作为上下文管理器时，yield 之前的任何异常都会导致生成器提前终止，而不会执行 finally 块中的清理
    # 工作。这意味着如果在 yield 之前发生异常，那么 finally 块中的代码（通常是恢复状态的代码）将不会被执行。
    # @signature_safe_contextmanager 装饰器确保了即使在 yield 之前发生异常，上下文管理器仍然能够按照预期的方式进行
    # 清理工作。它通过捕获 yield 之前的任何异常，并确保 finally 块中的代码被执行，从而保证了上下文管理器的安全性。
    @signature_safe_contextmanager
    def _backward_role_guard(self):
        # 不同的操作可能有不同的作用，例如前向传播、反向传播、优化步骤等。通过设置不同的角色，
        # 可以在执行时区分这些操作，并且在某些情况下可以根据角色的不同来调整行为或者进行特殊的处理。
        # self.__op_role_var 是一个与当前操作角色相关的变量列表。在执行特定的角色操作时（比如优化步骤），这个
        # 列表可能会被用来标识哪些变量是当前操作所关心的。例如，在优化过程中，我们通常关心的是参数及其对应的梯度，
        # 因此这个列表可能会包含这些变量的名字。
        tmp_role = self._current_role
        # 在进入上下文时将当前角色设置为 Backward，并在退出上下文时恢复原始角色。
        OpRole = core.op_proto_and_checker_maker.OpRole
        self._current_role = OpRole.Backward
        try:
            yield
        finally:
            self._current_role = tmp_role

    @signature_safe_contextmanager
    def _optimized_guard(self, param_and_grads):
        # 保存当前的角色和操作角色变量。
        tmp_role = self._current_role
        tmp_var = self.__op_role_var
        # 设置当前角色为 Optimize。
        OpRole = core.op_proto_and_checker_maker.OpRole
        self._current_role = OpRole.Optimize
        # 更新操作角色变量列表，将传入的参数和梯度转换为名称形式。
        self.__op_role_var = [
            var.name if isinstance(var, Variable) else var
            for var in param_and_grads
        ]
        # 执行上下文内的操作，并在退出上下文时恢复原来的角色和变量
        try:
            yield
        finally:
            self.__op_role_var = tmp_var
            self._current_role = tmp_role

    @signature_safe_contextmanager
    def _lr_schedule_guard(self, is_with_opt=False):
        # 保存当前的角色和操作角色变量。
        tmp_role = self._current_role
        tmp_var = self.__op_role_var
        # 设置当前角色为 LRSched。
        OpRole = core.op_proto_and_checker_maker.OpRole
        self._current_role = OpRole.LRSched
        # 如果 is_with_opt 为 True，则将角色设置为 LRSched 和 Optimize 的组合。
        if is_with_opt:
            self._current_role = int(OpRole.LRSched) | int(OpRole.Optimize)
        # TODO(typhoonzero): how to set target learning rate var
        self.__op_role_var = [] # 清空操作角色变量列表。
        try: # 执行上下文内的操作，并在退出上下文时恢复原来的角色和变量。
            yield
        finally:
            self.__op_role_var = tmp_var
            self._current_role = tmp_role

    def __str__(self):
        # 返回可读代码字符串：
        return self._to_readable_code()

    def _to_readable_code(self, skip_op_callstack=True):
        # 确保 skip_op_callstack 参数为布尔类型。
        assert isinstance(
            skip_op_callstack, bool
        ), "skip_op_callstack parameter's type is error, expect bool, received {}".format(
            type(skip_op_callstack)
        )
        program_str = "" # 初始化程序字符串。
        # 遍历每个块，并将每个块的可读代码拼接到 program_str 中。
        for block in self.blocks:
            program_str += block._to_readable_code(skip_op_callstack)
            program_str += '\n'
        return program_str # 返回拼接后的程序字符串。

    def to_string(self, throw_on_error, with_details=False):
        # 确保 throw_on_error 和 with_details 参数为布尔类型。
        assert isinstance(
            throw_on_error, bool
        ), "The type of throw_on_error parameter is wrong, expected bool, but received {}.".format(
            type(throw_on_error)
        )
        assert isinstance(
            with_details, bool
        ), "The type of with_details parameter is wrong, expected bool, but received {}.".format(
            type(with_details)
        )
        # 如果需要详细的字符串表示，则拼接每个块的详细字符串表示。
        if with_details:
            res_str = ""
            for block in self.blocks:
                res_str += block.to_string(throw_on_error, with_details)
        else: # 否则，将描述符序列化为字符串，并使用 _debug_string_ 函数生成简化字符串表示。
            protostr = self.desc.serialize_to_string()
            proto = framework_pb2.ProgramDesc.FromString(
                six.binary_type(protostr)
            )
            res_str = _debug_string_(proto, throw_on_error)
        return res_str
    def _get_desc(self):
        return self.desc
    def _version(self):
        return self.desc._version()

    def clone(self, for_test=False):
        # NOTE(zhiqiu): we sync the original program first, since its program may diff with
        # its desc due to modifying desc in c++ space. E.g. save op will add kLookupTablePath in desc.
        self._sync_with_cpp()

        pruned_origin_block_id_map = None
        if for_test:
            forward_prog = Program()
            forward_prog.desc, pruned_origin_block_id_map = core.prune_backward(
                self.desc
            )
            forward_prog.blocks = [
                Block(forward_prog, i)
                for i in six.moves.range(forward_prog.desc.num_blocks())
            ]
            forward_prog._sync_with_cpp()
            p = forward_prog._inference_optimize(prune_read_op=False)
        else:
            p = Program()
            p.current_block_idx = self.current_block_idx
            p._seed = self._seed
            p.desc = core.ProgramDesc(self.desc)
            p.blocks = [
                Block(p, i) for i in six.moves.range(self.desc.num_blocks())
            ]

            p._current_role = self._current_role
            p.__op_role_var = self.__op_role_var
            p._appending_grad_times = self._appending_grad_times
            if hasattr(self, 'lr_sheduler'):
                p.lr_sheduler = self.lr_sheduler

            # NOTE(zhiqiu): we sync the cloned program, to update its program by
            # its desc.
            p._sync_with_cpp()

        p._copy_param_info_from(self)
        p._copy_data_info_from(self, pruned_origin_block_id_map)
        p._copy_dist_param_info_from(self)
        return p

    def _prune(self, targets):
        
        return self._prune_with_input([], targets)

    def _prune_with_input(self, feeded_var_names, targets):
        
        # NOTE(zhiqiu): we sync the original program first, since its program may diff with
        # its desc due to modifying desc in c++ space. E.g. save op will add kLookupTablePath in desc.
        self._sync_with_cpp()

        if not isinstance(feeded_var_names, list):
            feeded_var_names = [feeded_var_names]
        if not isinstance(targets, list):
            targets = [targets]

        for var in feeded_var_names:
            if not isinstance(var, six.string_types):
                raise ValueError(
                    "All feeded_var_names of Program._prune_with_input() can only be "
                    "str, but received %s." % type(var)
                )

        # find out all variables that can be generated or updated with given feed
        generatable_vars = set()

        for idx, op in enumerate(self.global_block().ops):
            runnable_op = True
            for name in op.input_arg_names:
                if not self.global_block().has_var(name):
                    continue
                if self.global_block().var(name).persistable:
                    continue
                if name not in generatable_vars.union(feeded_var_names):
                    runnable_op = False
                    break
            if runnable_op:
                generatable_vars = generatable_vars.union(op.output_arg_names)

        targets_idx = []
        for t in targets:
            if not isinstance(t, Operator):
                if isinstance(t, Variable):
                    name = t.name
                elif isinstance(t, six.string_types):
                    name = str(t)
                else:
                    raise ValueError(
                        "All targets of Program._prune_with_input() can only be "
                        "Variable or Operator, but received %s." % type(t)
                    )

                # NOTEZ(zhiqiu): For variable to be fed in fetch_list, there two cases:
                # (1) the variable is leaf, it has no op that generates it;
                # (2) the variable is not leaf, and we need to prune the op that generates it.
                # In both cases, wo can just skip target_op of that it.
                if name in feeded_var_names:
                    # however if the var is also updated by a runnable op, will shall keep it
                    if name not in generatable_vars:
                        continue

                # After transpiler processing, the op that output this
                # variable maybe has been changed, so t.op is not reliable
                # and we need to find the current op that generate this
                # variable here.
                target_op = None
                global_block = self.global_block()
                for idx, op in enumerate(global_block.ops):
                    if name in op.output_arg_names:
                        # NOTE(zhiqiu): Find op that generate target name.
                        # Skip optimize op except for optimize op in targets,
                        # since optimize op generates parameters.
                        if op._is_optimize_op() and op not in targets:
                            continue
                        else:
                            target_op = op

                if target_op is not None:
                    targets_idx.append([target_op.block.idx, target_op.idx])
            else:
                targets_idx.append([t.block.idx, t.idx])

        res = Program()
        res.desc, pruned_origin_block_id_map = core.prune(
            self.desc, set(feeded_var_names), targets_idx
        )
        res.blocks = [
            Block(res, i) for i in six.moves.range(res.desc.num_blocks())
        ]
        res._sync_with_cpp()

        res._copy_param_info_from(self)
        res._copy_data_info_from(self, pruned_origin_block_id_map)
        res._copy_dist_param_info_from(self)

        return res

    def _inference_optimize(self, prune_read_op=True):
        res = Program()
        res.desc = core.ProgramDesc(self.desc)

        # remove all readers and the read_op if exist
        read_op_idx = 0
        root_block = res.desc.block(0)
        if prune_read_op:
            while True:
                if (
                    read_op_idx >= root_block.op_size()
                    or root_block.op(read_op_idx).type() == 'read'
                ):
                    break
                read_op_idx += 1
            if read_op_idx < root_block.op_size():
                root_block._remove_op(0, read_op_idx + 1)
            for var in root_block.all_vars():
                if var.type() == core.VarDesc.VarType.READER:
                    root_block._remove_var(cpt.to_bytes(var.name()))

        # change all `is_test` attributes to True
        for i in six.moves.range(res.desc.num_blocks()):
            block = res.desc.block(i)
            for j in six.moves.range(block.op_size()):
                op = block.op(j)
                if op.has_attr('is_test'):
                    op._set_bool_attr('is_test', True)
                if op.type() == "batch_norm":
                    # Remove the output ReserveSpace of batch_norm if exists.
                    op.remove_output("ReserveSpace")
        res.blocks = [
            Block(res, i) for i in six.moves.range(res.desc.num_blocks())
        ]
        res._sync_with_cpp()
        return res

    def _remove_training_info(self, clip_extra=True):
        
        res = Program()
        res.desc = core.ProgramDesc(self.desc)

        res.blocks = [
            Block(res, i) for i in six.moves.range(res.desc.num_blocks())
        ]
        res._sync_with_cpp()

        # Note: The op_role and op_role_var cann't be deleted currently,
        # and we will try to remove them in the future.
        common_clipped_attrs_list = ['op_callstack', 'with_quant_attr']

        for i in six.moves.range(res.desc.num_blocks()):
            block = res.desc.block(i)
            for var in block.all_vars():
                var.clear_is_parameter()
                var.clear_stop_gradient()
            if not clip_extra:
                continue
            for op_idx in range(0, block.op_size()):
                op = block.op(op_idx)
                if op.type() not in OpProtoHolder.instance().op_proto_map:
                    continue

                extra_attrs_map = core.get_op_extra_attrs(op.type())

                proto = OpProtoHolder.instance().get_op_proto(op.type())
                remove_input_list = []
                for name in op.input_names():
                    find = False
                    for input_proto in proto.inputs:
                        if input_proto.name != name:
                            continue
                        if input_proto.extra:
                            remove_input_list.append(name)
                        find = True
                        break
                    if not find:
                        remove_input_list.append(name)
                # The extra input of op will be removed in the future
                # for name in remove_input_list:
                #     op.remove_input(name)

                remove_output_list = []
                for name in op.output_names():
                    find = False
                    for output_proto in proto.outputs:
                        if output_proto.name != name:
                            continue
                        if output_proto.extra:
                            remove_output_list.append(name)
                        find = True
                        break
                    if not find:
                        remove_output_list.append(name)
                # The extra output of op will be removed in the future
                # for name in remove_output_list:
                #     op.remove_output(name)

                op_quant_name = (
                    core.op_proto_and_checker_maker.kOpWithQuantAttrName()
                )
                quant = (
                    bool(op.attr(op_quant_name))
                    if op_quant_name in op.attr_names()
                    else False
                )
                quant_attrs = [
                    op_quant_name,
                    "quantization_type",
                    "skip_quant",
                    "activation_bits",
                    "bit_length",
                    "quantize_weight_bits",
                    "weight_quant_scale",
                ]
                for extra_attr_name in extra_attrs_map.keys():
                    op.remove_attr(extra_attr_name)
                remove_attr_list = []
                for name in op.attr_names():
                    if quant:
                        if name in quant_attrs:
                            continue
                        if name.endswith("_threshold"):
                            continue
                    if len(extra_attrs_map) > 0:
                        if name in common_clipped_attrs_list:
                            op.remove_attr(name)
                        continue
                    find = False
                    for attr_proto in proto.attrs:
                        if attr_proto.name != name:
                            continue
                        find = True
                        break
                    if not find:
                        remove_attr_list.append(name)
                for name in remove_attr_list:
                    op.remove_attr(name)
        return res

    @staticmethod
    def parse_from_string(binary_str):
        p = Program()
        p.desc = core.ProgramDesc(binary_str)
        p.blocks = [Block(p, i) for i in six.moves.range(p.desc.num_blocks())]
        p._sync_with_cpp()
        return p

    @staticmethod
    def _construct_from_desc(desc):
        p = Program()
        p.desc = desc
        p.blocks = [Block(p, i) for i in six.moves.range(p.desc.num_blocks())]
        p._sync_with_cpp()
        return p

    @property
    def random_seed(self):
        return self._seed

    @property
    def num_blocks(self):
        return self.desc.num_blocks()

    @random_seed.setter
    def random_seed(self, seed):
        if not isinstance(seed, int):
            raise ValueError(
                "Program.random_seed's input seed must be an integer, but received %s."
                % type(seed)
            )
        self._seed = seed

    def __repr__(self):
        return self.__str__()

    def global_block(self):
        return self.blocks[0]

    def block(self, index):
        return self.blocks[index]

    def current_block(self):
        return self.blocks[self.current_block_idx]

    def _create_block(self, parent_idx=None):
        new_block_idx = len(self.blocks)
        parent = (
            self.current_block()
            if parent_idx is None
            else self.block(parent_idx)
        )
        self.desc.append_block(parent.desc)
        self.current_block_idx = new_block_idx
        self.blocks.append(Block(self, self.current_block_idx))
        return self.current_block()

    def _rollback(self):
        self.current_block_idx = self.current_block().parent_idx

    def _sync_with_cpp(self):
       
        for block_idx in range(len(self.blocks), self.desc.num_blocks()):
            self.blocks.append(Block(self, block_idx))
        for block in self.blocks:
            block._sync_with_cpp()

    def _copy_param_info_from(self, other):
       
        if not isinstance(other, Program):
            raise TypeError(
                "Function Program._copy_param_info_from() needs to pass in a source Program, but received %s"
                % type(other)
            )

        self.global_block()._copy_param_info_from(other.global_block())

    def _copy_dist_param_info_from(self, other):
       
        if not isinstance(other, Program):
            raise TypeError(
                "Function Program._copy_param_info_from() needs to pass in a source Program, but received %s"
                % type(other)
            )
        self._is_distributed = other._is_distributed
        self._is_chief = other._is_chief
        self._parameters_on_pservers = other._parameters_on_pservers
        self._endpoints = other._endpoints
        self._ps_endpoint = other._ps_endpoint
        self._distributed_lookup_table = other._distributed_lookup_table

    def _copy_data_info_from(self, other, pruned_origin_block_id_map=None):
       
        if not isinstance(other, Program):
            raise TypeError(
                "Function Program._copy_param_info_from() needs to pass in a source Program, but received %s"
                % type(other)
            )

        if not pruned_origin_block_id_map:
            pruned_origin_block_id_map = {
                i: i for i in six.moves.range(self.desc.num_blocks())
            }

        # NOTE(zhiqiu): All vars in cloned program exist in original program.
        # The reverse is not true, due to backward pruning.
        for i, block in enumerate(self.blocks):
            other_block = other.blocks[pruned_origin_block_id_map[i]]
            for var in list(block.vars.values()):
                other_var = other_block.var(var.name)
                if other_var.is_data:
                    var.is_data = True
                if other_var.desc.need_check_feed():
                    var.desc.set_need_check_feed(True)
                if other_var.stop_gradient:
                    var.stop_gradient = True

    def list_vars(self):
       
        for each_block in self.blocks:
            for each_var in list(each_block.vars.values()):
                yield each_var

    def all_parameters(self):
        
        parameters = []
        for each_block in self.blocks:
            parameters.extend(each_block.all_parameters())
        return parameters

    def state_dict(self, mode='all', scope=None):
       
        # The 'framework' is a low-level module, and 'executor'
        # can not be imported at the begainning of this file.
        # Therefore, the above two modules are dynamically imported.
        from .executor import global_scope

        if scope is not None and not isinstance(scope, core._Scope):
            raise TypeError(
                "`scope` should be None or `paddle.static.Scope'` type, but received {}.".format(
                    type(scope)
                )
            )

        if scope is None:
            scope = global_scope()

        if not isinstance(mode, str):
            raise TypeError(
                "Type of `mode` should be string, but received {}.".format(
                    type(mode)
                )
            )

        def is_parameter(var):
            return isinstance(var, Parameter)

        def is_persistable(var):
            if (
                var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH
                or var.desc.type() == core.VarDesc.VarType.FETCH_LIST
                or var.desc.type() == core.VarDesc.VarType.READER
            ):
                return False
            return var.persistable

        def is_belong_to_optimizer(var):
            if not (isinstance(var, Parameter) or var.desc.need_check_feed()):
                return is_persistable(var)
            return False

        def condition(var):

            if mode == 'param':
                return is_parameter(var)
            elif mode == 'opt':
                return is_belong_to_optimizer(var)
            elif mode == 'all':
                return is_parameter(var) or is_belong_to_optimizer(var)
            else:
                raise ValueError(
                    "`mode` string should be 'param', 'opt' or 'all', but received {}.".format(
                        mode
                    )
                )

        var_list = filter(condition, self.list_vars())

        state_dict = dict()
        for var in var_list:
            var_temp = scope.find_var(var.name)
            if var_temp is None:
                raise ValueError(
                    "Can not find Variable '{}' in the scope. Make sure it is initialized".format(
                        var.name
                    )
                )
            state_dict[var.name] = var_temp.get_tensor()

        return state_dict

    def set_state_dict(self, state_dict, scope=None):
        
        if not isinstance(state_dict, dict):
            raise TypeError(
                "Type of `state_dict` should be dict, but received {}.".format(
                    type(state_dict)
                )
            )

        vars_dict = {var.name: var for var in self.list_vars()}
        condition = (
            True if 'StructuredToParameterName@@' in state_dict else False
        )
        for name, value in state_dict.items():
            if condition:
                if name == "StructuredToParameterName@@":
                    continue
                if name in state_dict['StructuredToParameterName@@']:
                    name = state_dict['StructuredToParameterName@@'][name]
            if name in vars_dict:
                try:
                    vars_dict[name].set_value(value, scope)
                except ValueError as err:
                    warnings.warn(
                        ("Skip loading for '{}'. ".format(name) + str(err))
                    )
                except TypeError as err:
                    warnings.warn(
                        ("Skip loading for '{}'. ".format(name) + str(err))
                    )
            else:
                warnings.warn(
                    (
                        "Skip loading for '{0}'. Because '{0}' not in the program.".format(
                            name
                        )
                    )
                )

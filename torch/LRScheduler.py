# 所有学习率调度器的基类。LRScheduler 的主要职责是在训练过程中根据一定的规则调整优化器的学习率
class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        # 检查优化器类型
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        # optimizer: 一个 Optimizer 对象，用于存储模型的参数和更新规则。
        self.optimizer = optimizer  # 初始化优化器
        # Initialize epoch and base learning rates
        # last_epoch: 上一次调用 step 方法的 epoch 数。默认为 -1，表示从头开始。
        # 如果 last_epoch == -1，则设置 initial_lr 为当前的学习率。
        if last_epoch == -1: # 初始化 epoch 和基础学习率
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:# 否则，检查 initial_lr 是否存在于 param_groups 中，如果不存在则抛出异常。
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   f"in param_groups[{i}] when resuming an optimizer")
        # 获取基础学习率
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch # 初始化最后的 epoch
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()` 
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method
            # 这里修改了 optimizer.step 方法，增加了一个计数器 _step_count，确保 
            # optimizer.step() 在 lr_scheduler.step() 之前被调用
            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step) # 修改 optimizer.step 方法
        self.verbose = verbose # 设置是否打印学习率变化信息：

        self._initial_step() # 初始化步数

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        return self._last_lr
    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        if is_verbose:
            if epoch is None:
                print(f'Adjusting learning rate of group {group} to {lr:.4e}.')
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print(f'Epoch {epoch_str}: adjusting learning rate of group {group} to {lr:.4e}.')
    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1
        # 检查是否在 optimizer.step() 之前调用了 lr_scheduler.step()。
        # 更新 last_epoch 并计算新的学习率。
        # 更新 optimizer.param_groups 中的学习率。
        # 打印学习率信息（如果 verbose 为 True）。
        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
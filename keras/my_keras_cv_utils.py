def get_tensor_input_name(tensor):
    if keras_3():
        return tensor._keras_history.operation.name
    else:
        return tensor.node.layer.name
def parse_model_inputs(input_shape, input_tensor, **kwargs):
    if input_tensor is None:
        return keras.layers.Input(shape=input_shape, **kwargs)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return keras.layers.Input(
                tensor=input_tensor, shape=input_shape, **kwargs
            )
        else:
            return input_tensor
def correct_pad_downsample(inputs, kernel_size):
    img_dim = 1
    input_size = inputs.shape[img_dim : (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )
# 这里重写了 property 类的 __get__ 方法。__get__ 是一个特殊方法，用于在属性访问时调
# 用。在这个方法中，self 是 classproperty 的实例，_ 通常用于接收实例的引用（但在这里
# 我们不需要它，因为我们处理的是类属性，所以使用下划线 _ 作为约定俗成的弃用参数），
# owner_cls 是拥有这个属性的类本身。
class classproperty(property):
    def __get__(self, _, owner_cls):
        return self.fget(owner_cls)
def format_docstring(**replacements):
    """在这个format_docstring装饰器的上下文中，文档注释（docstring）中的容易变动的变量可以使用
    双括号{{variable_name}}作为占位符。当您使用装饰器并传递相应的**kwargs参数时，这些占位符会被
    动态地替换为实际的值，从而实现了文档注释的灵活配置。
    这种机制特别有用，当您需要在多个地方重复使用相同的文档注释模板，但某些部分（如函数名、参数名、版本
    号等）需要根据实际情况进行更改时。通过使用占位符和装饰器，您可以轻松地维护这些文档注释，而无需在每
    个函数、类或方法中手动更改它们。
    """
    def decorate(obj):
        doc = obj.__doc__
        # We use `str.format()` to replace variables in the docstring, but use
        # double brackets, e.g. {{var}}, to mark format strings. So we need to
        # to swap all double and single brackets in the source docstring.
        doc = "{".join(part.replace("{", "{{") for part in doc.split("{{"))
        doc = "}".join(part.replace("}", "}}") for part in doc.split("}}"))
        obj.__doc__ = doc.format(**replacements)
        return obj

    return decorate
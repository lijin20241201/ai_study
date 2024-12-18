from transformers import __version__
from transformers.dynamic_module_utils import custom_object_save
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from transformers.utils import (
    CONFIG_NAME,
    PushToHubMixin,
    add_model_info_to_auto_map,
    add_model_info_to_custom_pipelines,
    cached_file,
    copy_func,
    download_url,
    extract_commit_hash,
    is_remote_url,
    is_torch_available,
    logging,
)
import copy
import json
import re
import warnings
from typing import Any
from packaging import version
from collections import OrderedDict
from typing import Mapping
import huggingface_hub
from huggingface_hub import(
    _CACHED_NO_EXIST,
    CommitOperationAdd,
    ModelCard,
    ModelCardData,
    constants,
    create_branch,
    create_commit,
    create_repo,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    try_to_load_from_cache,
)
import os
from transformers.utils.generic import working_or_temp_dir
from typing import Dict, List, Optional, Tuple, Union
import torch
logger = logging.get_logger(__name__)
# 正则匹配配置文件
_re_configuration_file = re.compile(r"config\.(.*)\.json")
# PushToHubMixin 是一个 混入类（Mixin Class）。混入类是一个特定的设计模式，在面向对象编程中，尤其是在 Python 
# 中，常用于为其他类添加额外的功能，而不需要通过继承来创建复杂的类层次结构。
# 不包含状态：混入类一般不管理类的状态，而只是为类提供方法或额外的功能。例如，PushToHubMixin 类可能仅提供将模
# 型推送到 Hugging Face Hub 的方法，而不关心该类本身的状态。
# 在面向对象编程中，类的状态通常指的是类的 实例变量（也叫做 属性）的值。类的状态表示的是类在某一时刻的内部数据或信息。
class PushToHubMixin:
    # 用于创建仓库。Python 中的命名约定：单个下划线 (_)：表示这是一个 受保护的 属性或方法。虽然没有强制执行私有性，
    # 但这是告诉开发者，“这个方法/属性是内部的，应该在类的外部避免直接访问”。
    def _create_repo(
        self,
        repo_id: str,
        # Optional 是一个特殊的类型提示（type hint），它通常用于表明某个变量或参数可以是特定类型的值，也可以是
        # None。它实际上是 Union 类型的一个简化形式。Optional[X] 等价于 Union[X, None]
        private: Optional[bool] = None, 
        # token: Optional[Union[bool, str]] = None 是一种 类型注解，它的意思是 token 参数可以是 布尔值（
        # bool） 或 字符串（str） 类型，也可以是 None。
        token: Optional[Union[bool, str]] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> str:
        # 检查 repo_url 是否存在，如果存在，则发出警告，表示 repo_url 参数已弃用，并推荐使用 repo_id。
        if repo_url is not None:
            warnings.warn(
                "The `repo_url` argument is deprecated and will be removed in v5 of Transformers. Use `repo_id` "
                "instead."
            )
            # 如果 repo_id 和 repo_url 都设置了，则抛出 ValueError。
            if repo_id is not None:
                raise ValueError(
                    "`repo_id` and `repo_url` are both specified. Please set only the argument `repo_id`."
                )
            # 如果没有设置 repo_id，则从 repo_url 中提取 repo_id。
            repo_id = repo_url.replace(f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/", "")
        # 如果 organization 存在，则发出警告，提示使用新的参数格式。
        if organization is not None:
            warnings.warn(
                "The `organization` argument is deprecated and will be removed in v5 of Transformers. Set your "
                "organization directly in the `repo_id` passed instead (`repo_id={organization}/{model_id}`)."
            )
            # 如果 repo_id 不是以 organization 开头，则修正 repo_id。
            if not repo_id.startswith(organization):
                if "/" in repo_id:
                    repo_id = repo_id.split("/")[-1] # 列表最后一个
                repo_id = f"{organization}/{repo_id}"
        # 调用 create_repo 函数创建仓库，并返回仓库的 ID。
        url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        return url.repo_id
    # 定义一个 _get_files_timestamps 方法，用于获取指定目录下所有文件的最后修改时间戳。
    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        # 代码通过字典推导式构建一个字典。每个文件的名称 f 会作为字典的键，文件的最后修改时间戳会作为字典的值。
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}
    # 定义一个 _upload_modified_files 方法，用于上传修改过的文件到仓库。
    def _upload_modified_files(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        files_timestamps: Dict[str, float],
        commit_message: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        revision: str = None,
        commit_description: str = None,
    ):
        # 如果 commit_message 为空，则根据类名自动生成提交信息。
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            elif "Tokenizer" in self.__class__.__name__:
                commit_message = "Upload tokenizer"
            elif "FeatureExtractor" in self.__class__.__name__:
                commit_message = "Upload feature extractor"
            elif "Processor" in self.__class__.__name__:
                commit_message = "Upload processor"
            else:
                commit_message = f"Upload {self.__class__.__name__}"
        # f not in files_timestamps：如果 f 文件名没有在 files_timestamps 字典中（即该文件是新添加的，之前没有
        # 记录其时间戳），则它被认为是已修改的。
        # os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]：如果文件 f 在 files_timestamps 
        # 字典中，但其修改时间（通过 os.path.getmtime() 获取）比之前记录的时间戳（files_timestamps[f]）晚，说明该文件已经被修改了。
        modified_files = [
            f
            for f in os.listdir(working_dir)
            if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
        ]
        # 过滤掉非文件和非目录的条目。
        modified_files = [
            f
            for f in modified_files
            if os.path.isfile(os.path.join(working_dir, f)) or os.path.isdir(os.path.join(working_dir, f))
        ]
        # 构建上传操作列表。
        operations = []
        # upload standalone files
        for file in modified_files:
            if os.path.isdir(os.path.join(working_dir, file)): # 文件夹的情况
                # go over individual files of folder
                for f in os.listdir(os.path.join(working_dir, file)):
                    operations.append(
                        CommitOperationAdd(
                            path_or_fileobj=os.path.join(working_dir, file, f), path_in_repo=os.path.join(file, f)
                        )
                    )
            else: # 文件的情况
                operations.append(
                    CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file)
                )
        # 如果指定了 revision，则创建一个新的分支。
        if revision is not None:
            create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)
        # 记录日志信息，并执行提交操作。
        logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
        return create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            create_pr=create_pr,
            revision=revision,
        )
    # 定义一个 push_to_hub 方法，用于将模型推送到 HuggingFace Hub。
    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "5GB",
        create_pr: bool = False,
        safe_serialization: bool = True,
        revision: str = None,
        commit_description: str = None,
        tags: Optional[List[str]] = None,
        **deprecated_kwargs,
    ) -> str:
        use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
        ignore_metadata_errors = deprecated_kwargs.pop("ignore_metadata_errors", False)
        # 处理 use_auth_token 参数的弃用情况，并将其转换为 token。
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            
            token = use_auth_token

        repo_path_or_name = deprecated_kwargs.pop("repo_path_or_name", None)
        # 处理 repo_path_or_name 参数的弃用情况，并从中推断 repo_id 和 working_dir。
        if repo_path_or_name is not None:
            # Should use `repo_id` instead of `repo_path_or_name`. When using `repo_path_or_name`, we try to infer
            # repo_id from the folder path, if it exists.
            warnings.warn(
                "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
                "`repo_id` instead.",
                FutureWarning,
            )
            if repo_id is not None:
                raise ValueError(
                    "`repo_id` and `repo_path_or_name` are both specified. Please set only the argument `repo_id`."
                )
            if os.path.isdir(repo_path_or_name):
                # repo_path: infer repo_id from the path
                repo_id = repo_id.split(os.path.sep)[-1]
                working_dir = repo_id
            else:
                # repo_name: use it as repo_id
                repo_id = repo_path_or_name
                working_dir = repo_id.split("/")[-1]
        else:
            # Repo_id is passed correctly: infer working_dir from it
            working_dir = repo_id.split("/")[-1]

        # Deprecation warning will be sent after for repo_url and organization
        # 移除弃用的参数。
        repo_url = deprecated_kwargs.pop("repo_url", None)
        organization = deprecated_kwargs.pop("organization", None)
        # 创建仓库。
        repo_id = self._create_repo(
            repo_id, private=private, token=token, repo_url=repo_url, organization=organization
        )

        # Create a new empty model card and eventually tag it
        model_card = create_and_tag_model_card(
            repo_id, tags, token=token, ignore_metadata_errors=ignore_metadata_errors
        )
        # 决定是否使用临时目录。
        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)
        # 使用上下文管理器创建工作目录，并获取文件时间戳。
        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # 保存所有文件。self.save_pretrained 时，实际上是调用了继承自 PreTrainedModel 或其他具有相同方法
            # 的类中的 save_pretrained 方法。
            self.save_pretrained(work_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)

            # 更新模型卡
            model_card.save(os.path.join(work_dir, "README.md"))
            # 上传修改后的文件，并返回上传结果。
            return self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                revision=revision,
                commit_description=commit_description,
            )

# 预训练模型配置类,继承自PushToHubMixin
class PretrainedConfig(PushToHubMixin):
    model_type: str = "" # 标识模型的类型,类属性
    is_composition: bool = False # 指示模型是否由多个组件组成,类属性
    # 用于将属性名映射到实际存储的属性名，实现属性名的别名功能。类属性,具体的就是在构造方法内设置
    # 属性时,实际上是在调用object的__setattr__方法,这时如果子类重写了父类的这个方法,就会走这个
    # 重写方法的逻辑,从而真正存储的是attribute_map中key对应的value,在调用某个key对应的属性时
    # 也是走子类的__getattribute__的逻辑
    attribute_map: Dict[str, str] = {} 
    # 当用户没有明确指定应该使用哪个模型类时，框架可以根据 _auto_class 的值来自动选择合适的模型类。例如，
    # 如果 _auto_class 被设置为 "AutoConfig"，那么框架可能会使用 AutoConfig 类来根据模型类型自动选
    # 择并加载正确的配置类。
    _auto_class: Optional[str] = None
    # 如果该属性名在 attribute_map的键里面，则实际设置的是key在attribute_map中对应的value
    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)
    # 调用__getattribute__时,首先判断key是否是attribute_map中的键,如果是,返回的是它在attribute_map
    # 中对应的value的属性
    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)
    def __init__(self, **kwargs):
        # 首先从传入的kwargs中提取键对应的值
        # 是否返回字典形式的输出,默认True
        self.return_dict = kwargs.pop("return_dict", True)
        # 是否把经过最后一个解码器的dec_hidden加入all_hiddens,默认False
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        # 是否输出注意力权重
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
        self.torch_dtype = kwargs.pop("torch_dtype", None)  # Only used by PyTorch models
        # 是否使用float16数据类型
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.tf_legacy_loss = kwargs.pop("tf_legacy_loss", False)  # Only used by TensorFlow models
        # 要修剪的头索引集合,默认空集合
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        # 是否共享输入和输出的词嵌入（word embeddings）,默认是True
        self.tie_word_embeddings = kwargs.pop(
            "tie_word_embeddings", True
        )  
        # 设置在将样本数据送入前馈层前在特征维度进行拆分的大小,0表示不进行拆分
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        # 用来设置一个标记，指示当前模型是否为编码器-解码器（encoder-decoder）架构。
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        # 该标志用于指示当前模型是否为解码器部分。
        self.is_decoder = kwargs.pop("is_decoder", False)
        # 该参数指定跨注意力（cross-attention）机制中的隐藏层大小。跨注意力机制通常用于在编码器和解码器之间传递
        # 信息。这个参数可以用来控制跨注意力层的大小。
        self.cross_attention_hidden_size = kwargs.pop("cross_attention_hidden_size", None)
        # 该标志指示是否在解码器中添加跨注意力机制。如果为 True，则解码器会在生成输出时考虑编码器产生的上下文信息。
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        # 该标志指示是否共享编码器和解码器的权重。如果为 True，则编码器和解码器使用相同的权重,
        # 这在某些场景下可以减少模型参数的数量。
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)
        # 向后兼容性:序列生成的参数。虽然我们将保留加载这些的功能,参数，不建议保存它们。在遥远的未来
        # ，我们不需要装载它们。
        for parameter_name, default_value in self._get_generation_defaults().items():
            # 设置属性,如果kwargs没有当前参数键,就取默认值
            setattr(self, parameter_name, kwargs.pop(parameter_name, default_value))
        # 微调任务参数
        # 模型架构的名称。标识模型的具体实现。
        # 例如，["BertForSequenceClassification"] 表示这是一个基于 BERT 的序列分类模型。
        self.architectures = kwargs.pop("architectures", None)
        # 微调任务的名称。标识模型将用于哪个特定的微调任务。例如，"sequence-classification" 
        # 表示这是一个序列分类任务。
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        # id2label 是一个从 ID 到标签的映射，用于将数字 ID 映射到实际的标签。
        # label2id 是一个从标签到 ID 的映射，用于将实际的标签映射到数字 ID。
        self.id2label = kwargs.pop("id2label", None) 
        self.label2id = kwargs.pop("label2id", None)
        # 确保传入的 id2label 和 label2id 是字典类型，并且 id2label 的键为整数。
        if self.label2id is not None and not isinstance(self.label2id, dict):
            raise ValueError("Argument label2id should be a dictionary.")
        if self.id2label is not None:
            if not isinstance(self.id2label, dict):
                raise ValueError("Argument id2label should be a dictionary.")
            # token分类中要预测的标签数
            num_labels = kwargs.pop("num_labels", None)
            # 如果两者不一致,发出警告
            if num_labels is not None and len(self.id2label) != num_labels:
                logger.warning(
                    f"You passed along `num_labels={num_labels}` with an incompatible id to label map: "
                    f"{self.id2label}. The number of labels wil be overwritten to {self.num_labels}."
                )
            # key转换成整数形式
            self.id2label = {int(key): value for key, value in self.id2label.items()}
        # 这里是self.id2label未设定的情况
        else:
            # 使用num_labels作为几分类,这个是在序列分类时的情况
            self.num_labels = kwargs.pop("num_labels", 2)
        # 如果设置了torch_dtype
        if self.torch_dtype is not None and isinstance(self.torch_dtype, str):
            # 如果torch被导入
            if is_torch_available():
                import torch
                # 获取torch.dtype,torch.float32之类的
                self.torch_dtype = getattr(torch, self.torch_dtype)
        # 使用的分词器类的名称。
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None) # 模型的前缀。
        self.bos_token_id = kwargs.pop("bos_token_id", None) # 用于标记序列的开始。
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None) # 用于标记序列的结束。
        self.sep_token_id = kwargs.pop("sep_token_id", None) # 用于分割输入序列的不同部分
        # 用于解码器开始时的起始符号。
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)
        # 特定任务所需的额外参数,不同的任务可能需要不同的配置选项，例如特定的超参数或预处理步骤。
        self.task_specific_params = kwargs.pop("task_specific_params", None)
        # 问题类型。标识模型解决的问题类型，如回归、单标签分类或多标签分类。
        self.problem_type = kwargs.pop("problem_type", None)
        # 回归,单标签分类,多标签分类
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        # 确保传入的 problem_type 是有效的类型之一
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {self.problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )
        # TPU 设备标志。已废弃，用于指示是否使用 TPU 进行加速。
        if kwargs.pop("xla_device", None) is not None:
            logger.warning(
                "The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can "
                "safely remove it from your `config.json` file."
            )
        # 这是加载预训练模型或配置文件所需的名称或路径
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        # 用于记录预训练模型或配置文件的Git提交哈希值
        self._commit_hash = kwargs.pop("_commit_hash", None)
        # 注意力机制的实现方式，如果有相关的配置需求，会用到这个参数
        self._attn_implementation_internal = kwargs.pop("attn_implementation", None)
        # 用于记录当前Transformers库的版本信息
        self.transformers_version = kwargs.pop("transformers_version", None)
        # 如果传入了梯度检查点参数（并且值为True），会触发一个警告，说明这个参数在将来的版本
        # 中将不再被支持，并建议用户使用model.gradient_checkpointing_enable()方法
        if kwargs.get("gradient_checkpointing", False):
            warnings.warn(
                "Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5 "
                "Transformers. Using `model.gradient_checkpointing_enable()` instead, or if you are using the "
                "`Trainer` API, pass `gradient_checkpointing=True` in your `TrainingArguments`."
            )
        
        # 遍历kwargs中剩余的未处理参数，尝试将它们设置为类的属性。如果设置过程中出现AttributeError，则记录错误并重新
        # 抛出异常。这允许用户向配置类中添加自定义的参数。
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err
    @property
    def name_or_path(self) -> str: # 提供了获取和设置_name_or_path属性的方法
        return getattr(self, "_name_or_path", None)
    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)
    @property # 一个属性访问器，用于判断模型是否应该返回字典类型的结果而不是元组。
    def use_return_dict(self) -> bool:
        # 如果设置了torchscript属性,强制return_dict=False,避免jit错误
        return self.return_dict and not self.torchscript
    @property
    def num_labels(self) -> int: # 几分类
        return len(self.id2label)
    @num_labels.setter # id2label 是一个字典，它将每个标签的整数ID映射到其对应的标签名称
    def num_labels(self, num_labels: int):
        # 如果 self.id2label 不存在、为 None 或者其长度与新的 num_labels 不匹配，这个方法会重新初始化这两个字典。
        # 这样做是为了确保标签ID和标签名称之间有一一对应的关系，且数量正确。
        if not hasattr(self, "id2label") or self.id2label is None or len(self.id2label) != num_labels:
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            # label2id 是 id2label 的逆字典，它将标签名称映射回其整数ID
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))
    # 约定上的“受保护”成员变量或方法,这里提到的“私有”，更多的是从逻辑意义上讲，而不是从语言特性上来定义的。
    # 当访问 self._attn_implementation 属性时，实际上调用的是 getter 方法。
    @property
    def _attn_implementation(self):
        # 如果 _attn_implementation_internal 已经被设置，那么返回它的值；如果没有设置或者为 None，则返回默认值 "eager"。
        if hasattr(self, "_attn_implementation_internal"):
            if self._attn_implementation_internal is None:
                # `config.attn_implementation` should never be None, for backward compatibility.
                return "eager"
            else:
                return self._attn_implementation_internal
        else:
            return "eager"
    # 当尝试设置 _attn_implementation 属性时，setter 方法会被调用。它简单地将传入的 value 赋值
    # 给内部属性 _attn_implementation_internal。
    # _attn_implementation 既不是真正的私有属性（因为它没有使用双下划线前缀），也不是一个方法（因为它被定
    # 义为一个属性，尽管它有一个setter）。它是一个带有getter和setter的特殊属性。
    # getter 方法允许读取 _attn_implementation_internal 的值（如果它存在且不为None)，否则返回"eager"
    # setter 方法允许将新值赋给 _attn_implementation_internal。
     # 这里使用了 _attn_implementation_internal 作为实际存储属性值的地方，而 _attn_implementation 作为一个访问接口
    # ，这样做的目的是为了更好地控制属性的访问和修改行为。通过使用 property，可以在访问或设置属性时进行必要的检查或处理。
    @_attn_implementation.setter
    def _attn_implementation(self, value):
        self._attn_implementation_internal = value
    # 保存模型,参数:保存到的文件夹,Union[str, os.PathLike] 作为类型注解，表示它可以接受列表中任意一种类型的值。在这个例子中，
    # 它表示可以接受 str 类型或 os.PathLike 类型的对象。
    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        self._set_token_in_kwargs(kwargs) # 设置token到kwargs
        if os.path.isfile(save_directory): # 不能是文件类型
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
        non_default_generation_parameters = {}
        for parameter_name, default_value in self._get_generation_defaults().items():
            if hasattr(self, parameter_name) and getattr(self, parameter_name) != default_value:
                non_default_generation_parameters[parameter_name] = getattr(self, parameter_name)
        # 检查是否有任何非默认的生成参数被设置在模型配置中。
        # 如果存在非默认的生成参数，将会发出警告，并告知用户应该将这些参数保存在一个单独的 GenerationConfig 文件中。
        # 这一警告将在未来的版本中升级为错误
        if len(non_default_generation_parameters) > 0:
            logger.warning(
                "Some non-default generation parameters are set in the model config. These should go into a "
                "GenerationConfig file (https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model) "
                "instead. This warning will be raised to an exception in v4.41.\n"
                f"Non-default generation parameters: {str(non_default_generation_parameters)}"
            )
        # 如果指定了目录不存在，则创建该目录，允许目录已经存在 (exist_ok=True)。
        os.makedirs(save_directory, exist_ok=True)
        # 推送至远程仓库的相关处理：
        # 如果push_to_hub为True，则会从kwargs中弹出commit_message和repo_id
        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            # 从kwargs中弹出repo_id,如果没有提供，则使用保存目录的最后一部分作为repo_id
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            # 创建一个仓库
            repo_id = self._create_repo(repo_id, **kwargs)
            # 使用_get_files_timestamps方法获取保存目录中文件的时间戳，以便后续判断哪些文件被修改过
            files_timestamps = self._get_files_timestamps(save_directory)
        # 如果存在自定义配置，则保存定义该配置的文件到指定目录，并设置相应的属性，以便可以从远程仓库加载。
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)
        # 如果我们使用预定义的名称保存，我们可以使用' from_pretrained '加载
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        # 将配置保存为JSON文件（使用to_json_file方法），文件名为CONFIG_NAME
        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")
        # 如果需要推送到远程仓库，则上传修改后的文件，并提交更改。
        if push_to_hub: 
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )
    # _set_token_in_kwargs是一个静态方法，用于在kwargs中设置认证token。如果token为None，则从kwargs中
    # 尝试获取token或use_auth_token
    @staticmethod 
    def _set_token_in_kwargs(kwargs, token=None):
        # 如果传入的参数token是None,尝试从kwargs中获取,否则设定为None
        if token is None: 
            token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None) # 使用认证token
        # 如果获取到了这个属性,发出警告
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            # 如果同时指定了token和use_auth_token，则抛出ValueError。
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token
        # token 代表的是一个认证令牌（authentication token），它通常用于验证和授权对受保护资源的访问。
        # 在这个上下文中，它很可能是用于访问某个API(如Hugging Face的Model Hub API）的认证令牌，以便能
        # 够执行如上传模型、下载预训练模型等操作。
        if token is not None:# 如果token不为None，则将其添加到kwargs中
            kwargs["token"] = token
    # from_pretrained是一个类方法，用于从预训练模型加载配置。它接受多个参数，包括模型名称或路径、缓
    # 存目录、是否强制下载、是否仅使用本地文件、认证token、版本等。并返回一个 PretrainedConfig 对象。
    @classmethod # 类方法
    def from_pretrained(
        cls, # 当前类对象
        # 这里是预训练模型的保存路径:可选,默认是字符串或者PathLike类型
        pretrained_model_name_or_path: Union[str, os.PathLike],
        # 缓存目录的路径，用于缓存下载的模型文件。
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False, # 是否强制下载模型文件，即使本地已有缓存。
        local_files_only: bool = False,# 是否只从本地加载文件，而不从网络下载。
        token: Optional[Union[str, bool]] = None,# 访问模型仓库的访问令牌，可以是字符串或布尔值。
        revision: str = "main", # 指定从仓库加载的分支或标签。
        **kwargs, # 其他关键字参数
    ) -> "PretrainedConfig":
        # 将 cache_dir、force_download、local_files_only 和 revision 等参数添加到 kwargs 中。
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision
        # 使用 cls._set_token_in_kwargs 方法设置访问令牌。
        cls._set_token_in_kwargs(kwargs, token)
        # 调用 cls.get_config_dict 方法获取配置字典 config_dict 和更新后的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        # 如果配置字典中的 model_type 与当前类的 model_type 不匹配，发出警告。
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
        # 使用 cls.from_dict 方法从配置字典创建配置对象，并返回。
        return cls.from_dict(config_dict, **kwargs)
    # 这个方法用于从预训练模型的路径或名称中加载配置字典。类方法
    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # 使用 cls._set_token_in_kwargs 方法设置访问令牌。
        cls._set_token_in_kwargs(kwargs)
        original_kwargs = copy.deepcopy(kwargs) # 深拷贝
        # 调用 cls._get_config_dict 方法获取基本配置字典 config_dict 和更新后的 kwargs
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
        # 如果配置字典中包含 _commit_hash 字段，则将其记录到 original_kwargs 中。
        if "_commit_hash" in config_dict:
            original_kwargs["_commit_hash"] = config_dict["_commit_hash"]
        # 如果配置字典中包含 configuration_files 字段，则进一步处理指向的配置文件。
        if "configuration_files" in config_dict:
            configuration_file = get_configuration_file(config_dict["configuration_files"])
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )
        # 返回最终的配置字典 config_dict 和关键字参数 kwargs。
        return config_dict, kwargs
    # 用于从预训练模型的路径或名称中加载配置字典。该方法负责处理多种情况，包括本地文件、远程URL以及从缓存中
    # 加载配置文件，并且处理了多种参数，以确保配置文件的正确加载和解析。model_name_or_path:预训练模型的名称或路径。
    # kwargs包含多个关键字参数，如 cache_dir、force_download、resume_download 等，用于控制文件的下载和缓存行为。
    @classmethod # 类方法
    def _get_config_dict( 
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # 从 kwargs 中提取各种控制参数，如 cache_dir、force_download、resume_download 等
        # 指定下载的文件应缓存到哪个目录。如果未指定，则可能会使用默认缓存目录。
        cache_dir = kwargs.pop("cache_dir", None)
        # 如果设置为 True，则即使缓存目录中已存在相应文件，也会重新下载。
        force_download = kwargs.pop("force_download", False)
        # 如果设置为 True，并且已部分下载的文件存在于缓存目录中，则继续下载剩余部分。
        resume_download = kwargs.pop("resume_download", None)
        #  提供一个字典，用于设置 HTTP 和 HTTPS 代理。
        proxies = kwargs.pop("proxies", None)
        # 提供一个访问令牌，用于访问私有模型或需要身份验证的资源。
        token = kwargs.pop("token", None)
        # 如果设置为 True，则不会从网络下载文件，只会从本地文件系统加载文件。
        local_files_only = kwargs.pop("local_files_only", False)
        # 指定从模型存储库中加载哪个版本（通常是 Git 分支或标签）。
        revision = kwargs.pop("revision", None)
        # 如果设置为 True，则允许从远程位置加载自定义代码。此选项主要适用于自动类（Auto classes），
        # 在此处被忽略。
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        # 指定模型文件所在的子文件夹路径。
        subfolder = kwargs.pop("subfolder", "")
        # 当提到“内部使用”时，这通常指的是某些参数或变量在软件开发中的用途，它们主要用于内部逻辑处
        # 理，并非直接面向最终用户的接口部分。这些参数可能涉及到框架或库的内部机制，而不是公开API的一部分。
        # _from_pipeline 这个参数表明配置文件是从某个管道（pipeline）中获取的。例如，在Hugging Face的
        # Transformers库中，可能会有多个不同的管道（如文本分类、命名实体识别等），每个管道可能会有不同的配置
        # 要求或者处理方式。通过 _from_pipeline 参数，可以告诉内部逻辑是从哪个管道获取的配置，从而可以进行适当的处理。
        # 内部使用，表示配置是从哪个管道（pipeline）加载的。
        from_pipeline = kwargs.pop("_from_pipeline", None)
        # 内部使用，表示配置是从自动类（Auto class）加载的。
        from_auto_class = kwargs.pop("_from_auto", False)
        # 记录模型的特定版本，通常是一个 Git 提交的哈希值。
        commit_hash = kwargs.pop("_commit_hash", None)
        # 指定 GGUF 格式的文件名。GGUF 是一种轻量级的二进制格式，用于存储神经网络权重。
        gguf_file = kwargs.get("gguf_file", None)
        # 如果 trust_remote_code 为 True，则打印一条警告信息，因为该标志仅适用于自动类，此处忽略。
        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )
        # 设置 user_agent 字典，用于在下载文件时提供元数据信息。
        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        # 检查 pretrained_model_name_or_path 是否为本地目录或文件。
        is_local = os.path.isdir(pretrained_model_name_or_path)
        # 如果是本地文件，直接设置 resolved_config_file
        if os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            # Special case when pretrained_model_name_or_path is a local file
            resolved_config_file = pretrained_model_name_or_path
            is_local = True
        # 如果是远程URL，下载并设置 resolved_config_file
        elif is_remote_url(pretrained_model_name_or_path):
            configuration_file = pretrained_model_name_or_path if gguf_file is None else gguf_file
            resolved_config_file = download_url(pretrained_model_name_or_path)
        else:# 如果不是本地文件或远程URL，则尝试从本地文件夹或缓存中加载配置文件。
            configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME) if gguf_file is None else gguf_file
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            # 如果在加载配置文件时遇到任何环境错误或其他异常，抛出相应的错误信息。
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory"
                    f" containing a {configuration_file} file"
                )
        # 成功加载配置文件后，解析为字典，并记录 commit_hash
        try:
            if gguf_file:
                config_dict = load_gguf_checkpoint(resolved_config_file, return_tensors=False)["config"]
            else:
                # Load config dict
                config_dict = cls._dict_from_json_file(resolved_config_file)
            config_dict["_commit_hash"] = commit_hash
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
            )
        # 记录加载配置文件的日志信息。
        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")
        # 如果配置字典中包含 auto_map 或 custom_pipelines，则进行相应的处理。
        if "auto_map" in config_dict and not is_local:
            config_dict["auto_map"] = add_model_info_to_auto_map(
                config_dict["auto_map"], pretrained_model_name_or_path
            )
        if "custom_pipelines" in config_dict and not is_local:
            config_dict["custom_pipelines"] = add_model_info_to_custom_pipelines(
                config_dict["custom_pipelines"], pretrained_model_name_or_path
            )
        # 返回最终的配置字典 config_dict 和关键字参数 kwargs。
        return config_dict, kwargs

    @classmethod # 类方法,此方法用于从字典创建 PretrainedConfig 对象。
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # 移除 _from_auto 和 _from_pipeline 参数，因为它们是内部使用的，不应影响 return_unused_kwargs。
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # 如果 kwargs 中存在 _commit_hash 并且 config_dict 中也存在，则使用 config_dict 中的值。
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]
        # 将 config_dict 中的 attn_implementation 设置为 kwargs 的值，如果存在的话。
        # 从 kwargs 中移除 attn_implementation
        config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)
        # 使用 config_dict 创建 PretrainedConfig 对象。
        config = cls(**config_dict)
        # 如果配置中有 pruned_heads 属性，将其键转换为整数。
        if hasattr(config, "pruned_heads"):
            config.pruned_heads = {int(key): value for key, value in config.pruned_heads.items()}
        # 如果 kwargs 中存在 num_labels 和 id2label，并且两者长度不一致，则抛出异常。
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels }` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )
        to_remove = []
        # 使用 kwargs 更新配置对象的属性。
        for key, value in kwargs.items():
            if hasattr(config, key):
                current_attr = getattr(config, key)
                # To authorize passing a custom subconfig as kwarg in models that have nested configs.
                if isinstance(current_attr, PretrainedConfig) and isinstance(value, dict):
                    value = current_attr.__class__(**value)
                setattr(config, key, value)
                if key != "torch_dtype":
                    to_remove.append(key)
        # 移除已经用于更新配置对象的 kwargs。
        for key in to_remove:
            kwargs.pop(key, None)
        logger.info(f"Model config {config}")
        # 如果 return_unused_kwargs 为 True，则返回配置对象和未使用的 kwargs；否则只返回配置对象。
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config
    # 此方法用于从JSON文件加载配置对象。json_file：包含配置信息的JSON文件路径。
    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PretrainedConfig":
        # 使用 _dict_from_json_file 方法将JSON文件内容读取为字典。
        config_dict = cls._dict_from_json_file(json_file)
        # cls(**config_dict) 是构造函数（__init__ 方法）的调用，它使用传递的字典 config_dict 
        # 来初始化一个新的 PretrainedConfig 对象。
        return cls(**config_dict) 
    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        # 读取jsion文件,并转换成json格式字典返回
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
    # 判断两个配置是否相同
    def __eq__(self, other):
        return isinstance(other, PretrainedConfig) and (self.__dict__ == other.__dict__)
    # 打印字符串
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"
    # 此方法用于获取当前配置对象与默认配置不同的属性。
    # 假设我们有两个配置对象：一个是默认配置对象 default_config，另一个是我们当前正在使用的配置对象 current_config
    # default_config = PretrainedConfig(hidden_size=768, num_hidden_layers=12)
    # current_config = PretrainedConfig(hidden_size=768, num_hidden_layers=18, activation_function="gelu")
    # num_hidden_layers：current_config 中的值为 18，而 default_config 中的值为 12。
    # activation_function：current_config 中新增了一个属性 activation_function，其值为 "gelu"。
    # 当我们调用 current_config.to_diff_dict() 时，它将返回一个字典，仅包含那些与默认配置不同的配置项：
    # 记录哪些配置项与默认配置不同。
    # 在需要恢复默认配置时，可以使用这个差异字典来快速调整配置。
    # 在存储配置信息时，仅需存储这些不同的配置项，而不是整个配置对象。
    def to_diff_dict(self) -> Dict[str, Any]:
        # 使用 to_dict 方法获取当前配置的字典表示。
        config_dict = self.to_dict()
        # 获取默认配置字典。
        default_config_dict = PretrainedConfig().to_dict()
        # 如果当前配置是组合配置，则获取类特定配置字典。
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}
        serializable_config_dict = {}
        # 遍历当前配置字典，筛选出与默认配置不同的属性。
        for key, value in config_dict.items():
            # 检查当前属性是否为 PretrainedConfig 类型的实例。如果是，则表示这是一个嵌套的配置对象。
            # 检查当前键是否存在于类特定的配置字典中。进一步确认类特定配置字典中的对应键的值是否为字典类型。
            # 这是因为我们需要将嵌套配置的差异与类特定配置进行对比。
            if (
                isinstance(getattr(self, key, None), PretrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
            ):
                # 如果以上条件都满足，则使用 recursive_diff_dict 函数来递归地处理嵌套配置。这个函数将会找出嵌套配置与类特定
                # 配置之间的差异。
                # 这里 value 是当前属性的值，class_config_dict[key] 是类特定配置中的对应值，config_obj 是当前嵌套配置对象本身。
                diff = recursive_diff_dict(value, class_config_dict[key], config_obj=getattr(self, key, None))
                # 如果嵌套配置对象中有 model_type 属性，即使这个属性没有差异，也需要将其包含在差异字典中。这是因为它是一个
                # 重要的标识符，即使没有变化也需要保留。
                if "model_type" in value:
                    # Needs to be set even if it's not in the diff
                    diff["model_type"] = value["model_type"]
                # 如果差异字典 diff 的长度大于 0，则表示有差异，需要将其加入到 serializable_config_dict 中。
                if len(diff) > 0:
                    # serializable_config_dict[key] = diff：将差异字典加入到最终的可序列化配置字典中。
                    serializable_config_dict[key] = diff
            elif (
                key not in default_config_dict
                or key == "transformers_version"
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value
        # 如果有量化配置，将其转换为可序列化的形式。
        if hasattr(self, "quantization_config"):
            serializable_config_dict["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )
            # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            # 移除不可序列化的字段
            _ = serializable_config_dict.pop("_pre_quantization_dtype", None)
        self.dict_torch_dtype_to_str(serializable_config_dict)
        if "_attn_implementation_internal" in serializable_config_dict:
            del serializable_config_dict["_attn_implementation_internal"]
        # 返回差异配置字典。
        return serializable_config_dict
    # to_dict 方法的主要目的是将 PretrainedConfig 对象转换为其字典表示形式，以便于序列化和存储。这个方法会
    # 处理一些特殊的属性，比如嵌套的配置对象和量化配置等，确保它们能够被正确地转换为字典格式。
    def to_dict(self) -> Dict[str, Any]:
        # 复制对象属性：使用 deepcopy 复制对象的 __dict__ 属性。
        output = copy.deepcopy(self.__dict__)
        # 添加 model_type：如果类具有 model_type 属性，则添加到输出字典中。
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        # 移除内部使用属性：移除 _auto_class、_commit_hash 和 _attn_implementation_internal 这些内
        # 部使用的属性。    
        if "_auto_class" in output:
            del output["_auto_class"]
        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_attn_implementation_internal" in output:
            del output["_attn_implementation_internal"]
        # 设置 transformers_version：设置当前版本号。
        output["transformers_version"] = __version__
        # 处理嵌套配置：如果值是另一个 PretrainedConfig 对象，则递归调用 to_dict 方法
        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, PretrainedConfig):
                value = value.to_dict()
                del value["transformers_version"]
            output[key] = value
        # 处理量化配置：如果对象有 quantization_config 属性，则将其转换为字典格式，并移除不可序列化的
        # _pre_quantization_dtype。
        # 在深度学习和机器学习中，“量化”是指将模型中的权重或激活值从高精度数据类型（如浮点数）转换为较低精度的数据
        # 类型（如整数）。量化技术主要用于减少模型的内存占用和计算成本，同时还可以加速推理过程。量化配置通常包含有
        # 关如何执行量化的细节，例如量化的方法、精度、范围等。
        # 量化配置是用来存储与模型量化相关的信息，这些信息对于执行量化操作非常重要。在 to_dict 方法中，量化配置被转换
        # 为字典格式，并且任何不可序列化的字段（如 _pre_quantization_dtype）都会被移除，以确保最终的输出可以被序列化并存储。
        if hasattr(self, "quantization_config"):
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )
            # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            _ = output.pop("_pre_quantization_dtype", None)
        # 转换 torch.dtype 类型：调用 dict_torch_dtype_to_str 方法将 torch.dtype 类型转换为字
        # 符串表示
        self.dict_torch_dtype_to_str(output)

        return output
    # to_json_string 方法用于将配置对象转换为 JSON 格式的字符串。该方法接受一个布尔参数 use_diff，用于指
    # 定是否返回与默认配置不同的部分。如果 use_diff 为 True，则返回差异配置的 JSON 字符串；如果为 False，
    # 则返回完整配置的 JSON 字符串。
    # 当 use_diff 为 True 时，to_json_string 方法返回的是与默认配置不同的部分，这通常用于记录变更或差异。
    # 当 use_diff 为 False 时，返回的是完整的配置信息，这通常用于记录完整的配置状态。
    def to_json_string(self, use_diff: bool = True) -> str:
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        # config_dict：这是要序列化为 JSON 格式的字典对象,indent=2 表示每级缩进两个空格。
        # 输出的 JSON 字符串将以给定数量的空格缩进每一级，使得输出更加易读。sort_keys：如果
        # 为 True，则输出的 JSON 字符串中的键将按字母顺序排序。
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
    # 写出配置到文件
    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))
    # 更新当前对象的属性
    # update 方法接收一个字典 config_dict，然后遍历这个字典中的键值对，并将这些键值对设置为当
    # 前对象的属性。
    def update(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)
    # update_from_string 方法允许你通过一个字符串来更新对象的属性。这个方法首先将字符串解析成字典，然后根
    # 据字典中的键值对更新对象的属性。
    # 通过使用 update_from_string 方法，可以方便地从字符串中解析并更新对象的属性，提高配置管理的灵活性和便捷性。
    def update_from_string(self, update_str: str):
        # update_str：这是一个字符串，其中包含了一系列的键值对，键值对之间使用逗号 , 分隔，键和值之间
        # 使用等号 = 分隔。
        # 使用 update_str.split(",") 将输入字符串按逗号 , 分割成列表,使用列表推导式 [x.split("=")
        # for x in ...] 将每个元素按等号 = 分割成键和值的列表,使用 dict(...) 将分割后的键值对转换成字典 d。
        d = dict(x.split("=") for x in update_str.split(","))
        # 使用 for k, v in d.items(): 遍历字典 d 中的键值对。
        for k, v in d.items():
            # 使用 hasattr(self, k) 检查当前对象是否具有键 k 对应的属性。如果没有，则抛出 ValueError 异常。
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")
            # 使用 getattr(self, k) 获取键 k 对应的当前属性值 old_v。
            old_v = getattr(self, k)
            # 根据旧值 old_v 的类型，将新值 v 转换成相应的类型：
            # 如果 old_v 是布尔类型 (bool)，则根据字符串 v 的内容转换为 True 或 False。
            # 如果 old_v 是整数类型 (int)，则将 v 转换成整数。
            # 如果 old_v 是浮点数类型 (float)，则将 v 转换成浮点数。
            # 如果 old_v 不是字符串 (str)，则抛出 TypeError 异常。
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise TypeError(
                    f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
                )
            # 使用 setattr(self, k, v) 将新值 v 设置为对象的属性 k。
            setattr(self, k, v)
    # 用于将字典中的 torch_dtype 字段从 torch.dtype 类型转换为字符串类型，以便于序列化。
    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            # torch.float32,.split(".")[1]取出float32
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)
    # 这个类方法用于注册当前配置类到 transformers 库的自动配置类中，以便通过自动配置类来实例化当前配置类
    @classmethod # 类方法
    def register_for_auto_class(cls, auto_class="AutoConfig"):
        # 首先检查 auto_class 是否为字符串，如果不是，则使用类名作为自动配置类的名称。
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__
        import transformers.models.auto as auto_module
        # 然后检查 transformers.models.auto 模块中是否存在指定的自动配置类。
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")
        # 如果自动配置类存在，那么将当前配置类的名称存储到 _auto_class 属性中。
        cls._auto_class = auto_class
    # 方法前面加上一个下划线是受保护的方法（protected method），意味着它主要是在类的内部使用，而不是外部直接调用
    # 如果方法前面加上两个下划线,则表示这是一个私有方法,这样的方法名称会被进行名称改编即类外访问时会变成 
    # _ClassName__get_generation_defaults,这样做是为了避免子类无意中覆盖基类的私有方法，但实际上并不阻止外部访问。
    @staticmethod 
    # 静态方法,其键是字符串类型，值可以是任何类型（Any）。Any 是类型提示的一个特殊类型，表示可以接受任何类型的值。
    def _get_generation_defaults() -> Dict[str, Any]:
        return {
            "max_length": 20, # 生成文本的最大长度。
            "min_length": 0, # 生成文本的最小长度。
            # 是否使用随机采样来进行生成。如果设为 True，则在生成过程中使用随机采样；如果设为 False，则使用贪婪搜索
            "do_sample": False,
             # 是否启用提前终止。如果设为 True，则在满足一定条件（如达到 num_beams）时提前结束生成。
            "early_stopping": False,
            "num_beams": 1, # 增加 num_beams 可以提高生成质量，但会增加计算成本。
            "num_beam_groups": 1, # 在多样性生成中使用，确保生成结果具有多样性。
            "diversity_penalty": 0.0, # 增加多样性惩罚可以使得生成的结果更加多样化。
            "temperature": 1.0, # 温度越高，生成结果越随机；温度越低，生成结果越集中。
            "top_k": 50, # 只从概率最高的 k 个候选词中选择下一个词。
            "top_p": 1.0, # 只从累积概率达到 p 的候选词中选择下一个词。
            "typical_p": 1.0, # 只从累积概率达到 p 的候选词中选择下一个词，类似于 top_p 但有所不同。
            "repetition_penalty": 1.0, # 增加重复惩罚可以减少生成结果中的重复现象。
            "length_penalty": 1.0, # 影响生成序列的长度，通常与 num_beams 一起使用。
            "no_repeat_ngram_size": 0, # 确保生成的序列中不出现连续的 n-gram
            "encoder_no_repeat_ngram_size": 0, # 在编码器端应用 n-gram 不重复限制。
            "bad_words_ids": None, # 确保生成结果中不包含某些特定词汇。
            "num_return_sequences": 1, # 指定生成多少个序列
            "output_scores": False, # 是否输出每个生成步骤的得分。
            "return_dict_in_generate": False, # 生成时是否返回字典形式的结果。
            # 强制使用的开始符号 ID。确保生成序列以特定符号开始。
            "forced_bos_token_id": None, 
            "forced_eos_token_id": None, # 强制使用的结束符号 ID。
            #  是否移除无效的生成值。确保生成结果中不含无效值。
            "remove_invalid_values": False,
            # 根据长度对生成结果进行惩罚，使得较短的序列更有优势。
            "exponential_decay_length_penalty": None,
            "suppress_tokens": None, # 确保生成结果中不包含某些特定词汇。
            "begin_suppress_tokens": None, # 确保生成序列的开头不包含某些特定词汇。
        }
    # 判断是否有非默认的生成参数,模型的属性不应该有这些参数
    def _has_non_default_generation_parameters(self) -> bool:
    # 这段代码定义了一个名为 _has_non_default_generation_parameters 的方法，其目的是判断当前对象（self）是否有任何生成
    # （generation）参数被设置成了非默认值。这个方法通过以下步骤实现：
    # 获取默认参数：首先，它调用 self._get_generation_defaults() 方法来获取一个字典，这个字典包含了所有生成参数的名称（
    # parameter_name）和它们对应的默认值（default_value）
    # 遍历参数：然后，它遍历这个字典中的每一项（即每一个生成参数及其默认值）
    # 检查属性值：对于每一个参数，它首先检查当前对象（self）是否拥有这个参数名作为属性（使用 hasattr(self, parameter_name)）
    # 。如果拥有，它接着获取这个属性的值（使用 getattr(self, parameter_name)），并将其与默认值进行比较。
    # 返回结果：如果在遍历过程中发现任何一个参数的值不等于其默认值，方法立即返回 True，表示存在非默认的生成参数。如果遍历完
    # 所有参数后都没有发现非默认值，方法最终返回 False
    # 这个方法的设计非常适用于需要动态检查对象状态，特别是当对象的某些属性（在这里是生成参数）有预设的默认值，并且你可
    # 能需要知道这些属性是否被修改过的场景。
        for parameter_name, default_value in self._get_generation_defaults().items():
            if hasattr(self, parameter_name) and getattr(self, parameter_name) != default_value:
                return True
        return False
# 这个函数 get_configuration_file 的目的是从给定的一组配置文件中找到最适合当前 transformers 
# 版本的配置文件。它通过以下步骤实现这一目标：
# configuration_files: 一个字符串列表，包含多个配置文件的路径。
def get_configuration_file(configuration_files: List[str]) -> str:
    # 初始化映射表：创建一个空字典 configuration_files_map 用于存储匹配到的配置文件及其版本号。
    configuration_files_map = {} 
    # 正则表达式匹配：使用正则表达式 _re_configuration_file 匹配每个文件名，提取出版本号，并将
    # 匹配到的文件路径保存到 configuration_files_map 中。
    for file_name in configuration_files:
        search = _re_configuration_file.search(file_name) # 正则匹配
        if search is not None:
            v = search.groups()[0] 
            configuration_files_map[v] = file_name
    # 排序版本号：将所有匹配到的版本号进行排序，得到 available_versions 列表。
    available_versions = sorted(configuration_files_map.keys())
    # 选择配置文件：默认选择 CONFIG_NAME 作为配置文件名。
    configuration_file = CONFIG_NAME
    transformers_version = version.parse(__version__)
    # 遍历 available_versions 列表，查找与当前 transformers 版本兼容的最新配置文件，并返回该文件的路径。
    for v in available_versions:
        if version.parse(v) <= transformers_version:
            # 因为available_versions排序了,所以后边新的会覆盖旧的
            configuration_file = configuration_files_map[v]
        else:
            # No point going further since the versions are sorted.
            break
    return configuration_file
def recursive_diff_dict(dict_a, dict_b, config_obj=None):
    diff = {} # 用来存储差异的字典
    # 默认配置处理：如果提供了 config_obj，则尝试获取其默认配置的字典形式
    default = config_obj.__class__().to_dict() if config_obj is not None else {}
    # 遍历 dict_a：对于 dict_a 中的每个键值对
    # 特殊值处理：如果 config_obj中str(key)对应的属性是一个 PretrainedConfig 类的实例，并且该键也存在
    # 于 dict_b 中，且 dict_b 中的值也是一个字典，则递归调用 recursive_diff_dict 来比较这两个嵌套字典。
    # 如果递归调用返回值不为空，则将其添加到 diff 字典中。
    # 一般值比较：如果当前键不存在于 dict_b 中，或者 dict_a 和 dict_b 中对应键的值不相等，或者当前键不存在于
    # 默认配置中，或者 dict_a 中的值与默认配置中的值不相等，则将 dict_a 中的值添加到 diff 字典中。
    for key, value in dict_a.items():
        # 从 config_obj 中获取对应属性的值，并将其存储在 obj_value中,因为config_obj是对象
        obj_value = getattr(config_obj, str(key), None) 
        if isinstance(obj_value, PretrainedConfig) and key in dict_b and isinstance(dict_b[key], dict):
            diff_value = recursive_diff_dict(value, dict_b[key], config_obj=obj_value)
            if len(diff_value) > 0:
                diff[key] = diff_value
        elif key not in dict_b or value != dict_b[key] or key not in default or value != default[key]:
            diff[key] = value
    return diff
from abc import ABC, abstractmethod
# 通过这些定义和构造函数的逻辑，这个类提供了一种机制来管理不同任务类型的模型导出过程，包括配置信息、
# 任务类型以及可能需要打补丁的操作。这有助于确保导出的ONNX模型符合预期的形状和格式要求。
# 在软件开发和机器学习领域，“打补丁”（Patching）通常指的是对现有代码或行为进行局部修改，以修复错误、
# 增强功能或适应特定的需求。在机器学习模型的上下文中，“打补丁”通常涉及到对模型的行为进行微调
# ，使其能够在特定的环境下更好地工作，尤其是在导出模型为ONNX格式时，可能需要对模型的某些部分进行
# 特殊的处理。
# 在将深度学习模型导出为ONNX格式的过程中，“打补丁”的概念主要体现在以下几个方面：
# 操作符不兼容：
# 某些深度学习框架中的操作符在ONNX中没有直接对应的实现，或者实现方式不同。这时就需要对模型进行一些
# 调整，使其能够正确地转换为ONIX格式。
# 例如，某些框架特有的操作符需要被替换为ONNX支持的基本操作符组合。
# 静态图优化：
# 在导出过程中，可能需要对模型的计算图进行优化，以确保在ONNX中能够高效执行。这可能涉及到重写某些层
# 或操作的实现。
# 比如，将某些动态操作改为静态操作，或者合并某些计算步骤。
# 内存管理和性能优化：
# 在导出过程中，可能需要对模型的内存使用模式进行调整，以提高性能或减少内存占用。
# 例如，调整张量的布局或尺寸，以适应ONNX的内存管理策略。
# 解决框架差异：
# 不同的深度学习框架在实现上可能存在差异，当需要将模型导出为ONNX格式时，可能需要针对这些差异进行特殊处理。
# 例如，解决框架之间的API差异，使模型能够正确地在ONNX环境中运行。
class OnnxConfig(ABC):
    """
    ONNX可导出模型的基类，描述了如何通过ONNX格式导出模型的元数据。这个基类是为了支持模型以ONNX格式导出
    而设计的，并且包含了关于如何进行导出的相关元数据信息。ONNX（Open Neural Network Exchange）
    是一种开放格式，用于表示机器学习模型，使得模型可以在不同的框架之间共享。这个基类提供了一种标准化的方
    式来描述和处理模型导出的过程。
    """
    # 这些变量定义了一些默认的固定值，用于在导出ONNX模型时使用。它们可能是为了简化导出过程而设定的一些默认
    # 批量大小、序列长度和其他参数。
    default_fixed_batch = 2
    default_fixed_sequence = 8
    default_fixed_num_choices = 4
    # ONNX支持的PyTorch最低版本
    torch_onnx_minimum_version = version.parse("1.8")
    # _tasks_to_common_outputs 是一个字典，用于映射不同的任务类型到其常见的输出形状。
    _tasks_to_common_outputs = {
        # 预测下个token
        "causal-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "default": OrderedDict({"last_hidden_state": {0: "batch", 1: "sequence"}}),
        # 图片分类
        "image-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        # 图像分割和目标检测
        "image-segmentation": OrderedDict( 
            {
                "logits": {0: "batch", 1: "sequence"},
                "pred_boxes": {0: "batch", 1: "sequence"},
                "pred_masks": {0: "batch", 1: "sequence"},
            }
        ),
        "masked-im": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "masked-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "multiple-choice": OrderedDict({"logits": {0: "batch"}}),
        # 目标检测
        "object-detection": OrderedDict(
            {
                "logits": {0: "batch", 1: "sequence"},
                "pred_boxes": {0: "batch", 1: "sequence"},
            }
        ),
        # 问答
        "question-answering": OrderedDict(
            {
                "start_logits": {0: "batch", 1: "sequence"},
                "end_logits": {0: "batch", 1: "sequence"},
            }
        ),
        "semantic-segmentation": OrderedDict({"logits": {0: "batch", 1: "num_labels", 2: "height", 3: "width"}}),
        "seq2seq-lm": OrderedDict({"logits": {0: "batch", 1: "decoder_sequence"}}),
        # 序列分类
        "sequence-classification": OrderedDict({"logits": {0: "batch"}}),
        # token 分类
        "token-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "vision2seq-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "speech2seq-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
    }
    # config：这是传入的一个配置对象，通常是一个继承自 PretrainedConfig 的类的实例，包含了模型的配置信息。
    # task：这是一个字符串，表示当前模型的任务类型，默认为 "default"。atching_specs：这是一个 
    # PatchingSpec 类型的列表，用于描述在导出过程中需要打补丁的操作。如果未提供，则默认为空列表。
    def __init__(self, config: "PretrainedConfig", task: str = "default", patching_specs: List[PatchingSpec] = None):
        # 将传入的配置对象赋值给类的 _config 属性。
        self._config = config
        # 如果传入的任务类型不在 _tasks_to_common_outputs 字典的键中，则抛出一个 ValueError 异常，指出不支持的任务类型。
        if task not in self._tasks_to_common_outputs:
            raise ValueError(
                f"{task} is not a supported task, supported tasks: {self._tasks_to_common_outputs.keys()}"
            )
        # self.task = task：将传入的任务类型赋值给类的 task 属性。
        self.task = task
        # patching_specs 是一个列表，包含一系列的 PatchingSpec 对象，这些对象描述了需要在导出过程中应用的补丁
        self._patching_specs = []
        # 处理补丁规格：遍历列表中的每个 PatchingSpec 对象。
        # 如果 patching_specs 不为空，则遍历列表中的每个 PatchingSpec 对象。
        for spec in patching_specs if patching_specs is not None else []:
            final_spec = spec
        # 如果 spec.orig_op 为空，则使用 spec.o 的 spec.name 属性来获取原始操作，并替换 spec.orig_op。
        # 这样做可能是为了确保 orig_op 字段不为空，以便后续处理可以正确进行。orig_op：原始的操作或模块。
        # name：操作或模块的名称。o：可能是指向原始操作或模块的对象。
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            # 将处理后的 PatchingSpec 对象添加到 _patching_specs 列表中
            # 通过这种方式，可以在模型导出过程中应用必要的补丁，以确保模型能够正确地转换为ONNX格式，并
            # 在ONNX环境中正常工作。
            self._patching_specs.append(final_spec)
    @classmethod # 类方法
    def from_model_config(cls, config: "PretrainedConfig", task: str = "default") -> "OnnxConfig":
        # 调用构造方法返回实例对象
        return cls(config, task=task)
   # 抽象方法,属性,返回值是一个映射字典,需要子类重写
    @property
    @abstractmethod
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        raise NotImplementedError()
    # 返回当前任务类型对应的常见输出形状。
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = self._tasks_to_common_outputs[self.task]
        return copy.deepcopy(common_outputs)
    # 返回一个字典，用于覆盖配置中的某些值,在导出ONNX模型时，可能需要临时修改
    # 某些配置值，例如禁用缓存。
    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        if hasattr(self._config, "use_cache"):
            return {"use_cache": False}
        return None
    # 返回默认的批量大小。
    @property
    def default_batch_size(self) -> int:
        # Using 2 avoid ONNX making assumption about single sample batch
        return OnnxConfig.default_fixed_batch
    # 返回默认的序列长度
    @property
    def default_sequence_length(self) -> int:
        return OnnxConfig.default_fixed_sequence
    # 提供默认的选择数量，适用于多选题等任务。
    @property
    def default_num_choices(self) -> int:
        return OnnxConfig.default_fixed_num_choices
    # 指定ONNX模型导出时使用的基本操作集版本
    @property
    def default_onnx_opset(self) -> int:
        return DEFAULT_ONNX_OPSET
    # 在验证导出的ONNX模型时使用的绝对容差值。
    @property
    def atol_for_validation(self) -> float:
        return 1e-5
    # 判断当前环境是否支持ONNX导出所需的PyTorch版本。
    @property
    def is_torch_support_available(self) -> bool:
        if is_torch_available():
            from transformers.utils import get_torch_version
            return version.parse(get_torch_version()) >= self.torch_onnx_minimum_version
        else:
            return False
    # 静态方法，判断模型参数是否需要使用外部数据格式。在ONNX模型文件过大时，决定是否使用外部数
    # 据格式存储模型参数。
    @staticmethod
    def use_external_data_format(num_parameters: int) -> bool:
        return (
            compute_serialized_parameters_size(num_parameters, ParameterFormat.Float)
            >= EXTERNAL_DATA_FORMAT_SIZE_LIMIT
        )
    # 在模型导出或测试时，提供一批模拟的图像数据作为输入。
    def _generate_dummy_images(
        self, batch_size: int = 2, num_channels: int = 3, image_height: int = 40, image_width: int = 40
    ):
        images = []
        for _ in range(batch_size):
            data = np.random.rand(image_height, image_width, num_channels) * 255
            images.append(Image.fromarray(data.astype("uint8")).convert("RGB"))
        return images
    # 该方法用于生成虚拟的音频数据，通常用于音频处理模型的测试或导出。
    # batch_size：生成音频数据的批量大小，默认为2。sampling_rate：采样率，默认为22050Hz。
    # time_duration：音频的持续时间，默认为5秒。frequency：生成的纯音波频率，默认为220Hz。
    # 该方法生成一个批量的纯音波信号，每个信号是一个正弦波形。对于每个样本，它首先创建一个时间数组 t
    # ，然后基于这个时间数组生成一个纯正弦波形，并将其添加到 audio_data 列表中。
    # 在模型训练、测试或导出过程中，当没有实际音频数据可用时，可以使用这种方法生成虚拟的音频数据作为输入。
    def _generate_dummy_audio(
        self, batch_size: int = 2, sampling_rate: int = 22050, time_duration: float = 5.0, frequency: int = 220
    ):
        audio_data = []
        for _ in range(batch_size):
            # time variable
            t = np.linspace(0, time_duration, int(time_duration * sampling_rate), endpoint=False)
            # generate pure sine wave at `frequency` Hz
            audio_data.append(0.5 * np.sin(2 * np.pi * frequency * t))
        return audio_data
    # 该方法用于生成虚拟的输入数据，根据传入的不同预处理器（如文本、图像或音频处理工具），生成
    # 相应的虚拟输入数据。在模型训练、测试或导出过程中，当没有实际输入数据可用时，可以使用这种
    # 方法生成虚拟的输入数据作为占位符，以便测试模型的处理流程或导出模型为ONNX格式。
    # 这段代码主要用于生成虚拟输入数据，以供模型导出或测试使用。通过不同的预处理器类型（如文本、图像或音频处理工具）
    # ，它可以灵活地生成相应的虚拟输入数据，帮助开发者在缺少实际数据的情况下测试模型的完整性和导出流程。
    def generate_dummy_inputs(
        self,
        # preprocessor：预处理器对象，可以是 PreTrainedTokenizerBase（文本）、FeatureExtractionMixin
        # （音频或图像）或 ImageProcessingMixin（图像）。
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin", "ImageProcessingMixin"],
        batch_size: int = -1, # 批量大小，默认为-1（表示动态轴）
        seq_length: int = -1, # 序列长度，默认为-1（表示动态轴）。
        num_choices: int = -1,# 选项数量，默认为-1（表示动态轴）
        is_pair: bool = False,# 是否为配对输入，默认为False。
        framework: Optional[TensorType] = None, # 使用的框架类型，如TensorFlow或PyTorch。
        num_channels: int = 3,# 图像的通道数，默认为3（RGB）。
        image_width: int = 40, # 图像宽度，默认为40像素。
        image_height: int = 40, # 图像高度，默认为40像素。
        sampling_rate: int = 22050, # 采样率，默认为22050Hz。
        time_duration: float = 5.0, # 音频持续时间，默认为5秒。
        frequency: int = 220, # 音频频率，默认为220Hz。
        tokenizer: "PreTrainedTokenizerBase" = None, # 标记器对象，已弃用。
    ) -> Mapping[str, Any]:
        from transformers.feature_extraction_utils import FeatureExtractionMixin
        from transformers.image_processing_utils import ImageProcessingMixin
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        # 该方法首先检查传入的参数，并根据预处理器类型生成相应的虚拟输入数据。具体实现如下：
        # 如果同时传入了 tokenizer 和 preprocessor，则抛出错误。
        if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
            raise ValueError("You cannot provide both a tokenizer and a preprocessor to generate dummy inputs.")
        # 如果传入了 tokenizer，则警告并将其作为 preprocessor 使用。
        if tokenizer is not None:
            warnings.warn(
                "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
                " `preprocessor` instead.",
                FutureWarning,
            )
            logger.warning("Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.")
            preprocessor = tokenizer
        # 生成虚拟文本输入：
        # 计算有效的批量大小和序列长度。
        # 生成虚拟的文本输入，并使用预处理器进行编码。
        if isinstance(preprocessor, PreTrainedTokenizerBase):
            # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
            batch_size = compute_effective_axis_dimension(
                batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
            )
            # If dynamic axis (-1) we forward with a fixed dimension of 8 tokens to avoid optimizations made by ONNX
            token_to_add = preprocessor.num_special_tokens_to_add(is_pair)
            seq_length = compute_effective_axis_dimension(
                seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
            )
            # Generate dummy inputs according to compute batch and sequence
            input_token = (
                preprocessor.unk_token
                if (preprocessor.unk_token is not None and len(preprocessor.unk_token) > 0)
                else "0"
            )
            dummy_input = [" ".join([input_token]) * seq_length] * batch_size
            if self.task == "multiple-choice":
                # If dynamic axis (-1) we forward with a fixed dimension of 4 candidate answers to avoid optimizations
                # made by ONNX
                num_choices = compute_effective_axis_dimension(
                    num_choices, fixed_dimension=OnnxConfig.default_fixed_num_choices, num_token_to_add=0
                )
                dummy_input = dummy_input * num_choices
                # The shape of the tokenized inputs values is [batch_size * num_choices, seq_length]
                tokenized_input = preprocessor(dummy_input, text_pair=dummy_input)
                # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
                for k, v in tokenized_input.items():
                    tokenized_input[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]
                return dict(tokenized_input.convert_to_tensors(tensor_type=framework))
            return dict(preprocessor(dummy_input, return_tensors=framework))
        # 调用 _generate_dummy_images 方法生成虚拟图像，并使用预处理器进行处理。
        elif isinstance(preprocessor, ImageProcessingMixin):
            if preprocessor.model_input_names[0] != "pixel_values":
                raise ValueError(
                    f"The `preprocessor` is an image processor ({preprocessor.__class__.__name__}) and expects"
                    f' `model_input_names[0]` to be "pixel_values", but got {preprocessor.model_input_names[0]}'
                )
            # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            dummy_input = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
            return dict(preprocessor(images=dummy_input, return_tensors=framework))
        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == "pixel_values":
            # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            dummy_input = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
            return dict(preprocessor(images=dummy_input, return_tensors=framework))
        # 生成虚拟音频输入：调用 _generate_dummy_audio 方法生成虚拟音频，并使用预处理器进行处理。
        elif (
            isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == "input_features"
        ):
            # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            dummy_input = self._generate_dummy_audio(batch_size, sampling_rate, time_duration, frequency)
            # 根据不同的预处理器类型，返回相应的虚拟输入数据。
            return dict(preprocessor(dummy_input, return_tensors=framework))
        else:
            raise ValueError(
                "Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor."
            )
    # 此方法可能用于在导出模型到ONNX格式或者测试模型时提供一致性的输入数据。它直接返回传入的参考输入，这
    # 意味着它假定传入的输入已经适合作为虚拟输入。
    def generate_dummy_inputs_onnxruntime(self, reference_model_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return reference_model_inputs
    # 此方法用于替换ONNX模型中的特定操作符（ops）。
    def patch_ops(self):
        for spec in self._patching_specs:
            custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
            setattr(spec.o, spec.name, custom_op)
    # 此方法用于恢复之前由 patch_ops 替换掉的操作符。此方法允许用户在不需要自定义行为时恢复模型的原始操作符，从而确
    # 保模型的一致性和正确性。
    def restore_ops(self):
        # 遍历 _patching_specs 中的所有规范，并根据规范中的信息恢复原来的操作符。如果指定了 op_wrapper，
        # 则使用 op_wrapper 包装原始操作符。
        for spec in self._patching_specs:
            orig_op = spec.orig_op if spec.op_wrapper is None else spec.op_wrapper(spec.orig_op)
            setattr(spec.o, spec.name, orig_op)
    # 此方法用于将模型输出的集合属性扁平化处理。使用 itertools.chain.from_iterable 将 field 中的每一项展开成
    # 单一序列，然后构造一个新的字典，其中键是 name 后跟点和索引号，值是对应的项。
    # 此方法可能用于处理ONNX模型输出的复杂结构，例如当输出是一个嵌套的列表或元组时，将其转换为一个简单的字典结构，
    # 方便进一步处理。
    @classmethod
    def flatten_output_collection_property(cls, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        from itertools import chain
        return {f"{name}.{idx}": item for idx, item in enumerate(chain.from_iterable(field))}
@dataclasses.dataclass
class PatchingSpec:
    o: Any # 表示要打补丁的目标对象
    name: str # 表示要替换的方法或属性的名称。
    custom_op: Callable # 表示用于替换原有方法的新方法或操作。
    orig_op: Optional[Callable] = None # 表示原始的方法或操作，如果需要保留原始的操作，可以在这里存储
    # 示一个包装器函数，用于包装 custom_op，可以用来添加额外的逻辑或处理。
    op_wrapper: Optional[Callable] = None
# 这段代码定义了一个名为 OnnxConfigWithPast 的类，它是从 OnnxConfig 继承而来，并且实现了对
# ONNX导出的支持，特别关注于处理包含过去键值对（past key-values）的情况。这类配置主要
# 用于支持序列生成任务中的增量解码，即在每次生成新的token时，利用之前生成的序列信息来加速解码过程。
class OnnxConfigWithPast(OnnxConfig, ABC):
    # 初始化方法接收配置对象、任务名称、补丁规范列表和是否使用过去键值对作为参数，并设置 use_past 属性。
    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        super().__init__(config, task=task, patching_specs=patching_specs)
        self.use_past = use_past
    # 提供一个方便的方法来创建使用过去键值对的配置实例。
    @classmethod
    def with_past(cls, config: "PretrainedConfig", task: str = "default") -> "OnnxConfigWithPast":
        
        return cls(config, task=task, use_past=True)
    # 根据是否使用过去键值对来填充输出映射。
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super().outputs
        if self.use_past:
            self.fill_with_past_key_values_(common_outputs, direction="outputs")
        return common_outputs
    # 根据配置中的 use_cache 属性来决定是否使用过去键值对。
    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        if hasattr(self._config, "use_cache"):
            return {"use_cache": self.use_past}
        return None
    # 获取模型配置中的层数
    @property
    def num_layers(self) -> int:
        # 如果_config中没这个属性,抛出错误
        if not hasattr(self._config, "num_layers"):
            raise AttributeError(
                "could not find the number of layers attribute in the model configuration, override the num_layers"
                " property of the model OnnxConfig to solve this"
            )
        # 正常情况返回层数
        return self._config.num_layers
    # 取模型配置中的注意力头数。
    @property
    def num_attention_heads(self) -> int:
        if not hasattr(self._config, "num_attention_heads"):
            raise AttributeError(
                "could not find the number of attention heads attribute in the model configuration, override the"
                " num_attention_heads property of the model OnnxConfig to solve this"
            )
        return self._config.num_attention_heads
    # 生成用于ONNX导出的虚拟输入数据，特别处理了使用过去键值对的情况
    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # TODO: should we set seq_length = 1 when self.use_past = True?
        common_inputs = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

            batch, seqlen = common_inputs["input_ids"].shape
            # Not using the same length for past_key_values
            past_key_values_length = seqlen + 2
            shape = (
                batch,
                self.num_attention_heads,
                past_key_values_length,
                self._config.hidden_size // self.num_attention_heads,
            )

            if "attention_mask" in common_inputs:
                mask_dtype = common_inputs["attention_mask"].dtype
                common_inputs["attention_mask"] = torch.cat(
                    [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)],
                    dim=1,
                )

            common_inputs["past_key_values"] = []
            for _ in range(self.num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))

        return common_inputs
    # 填充输入或输出映射中的过去键值对动态轴。
    def fill_with_past_key_values_(
        self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str, inverted_values_shape: bool = False
    ):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        name = "past_key_values" if direction == "inputs" else "present"
        for i in range(self.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
            if inverted_values_shape:
                inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch", 1: "past_sequence + sequence"}
            else:
                inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
    # 将嵌套的过去键值对扁平化为易于处理的形式。
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        flattened_output[f"{name}.{idx}.key"] = t[0]
        flattened_output[f"{name}.{idx}.value"] = t[1]
    # 根据名字和字段来扁平化输出集合属性，特别处理了 present 和 past_key_values。
    def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        flattened_output = {}
        if name in ["present", "past_key_values"]:
            for idx, t in enumerate(field):
                self._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            flattened_output = super().flatten_output_collection_property(name, field)
        return flattened_output
# 专门用于处理序列到序列（seq2seq）模型的ONNX导出，尤其是那些支持使用过去键值对
# （past key-values）来进行增量解码的模型,此类主要用于在导出模型时正确处理编
# 码器和解码器的输出轴名称，以及生成虚拟输入数据时考虑编码器和解码器的不同需求
class OnnxSeq2SeqConfigWithPast(OnnxConfigWithPast):
    # 此方法调整了输出轴的名称，确保对于编码器和解码器有不同的轴名称，
    # 并且在使用过去键值对时，调用父类的 fill_with_past_key_values_ 方法来处理输出。
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super(OnnxConfigWithPast, self).outputs
        # Renaming the outputs axes properly.
        for name, axes_names in common_outputs.items():
            sequence_name = "encoder_sequence" if "encoder" in name else "decoder_sequence"
            for axis_idx, name in axes_names.items():
                if "sequence" in name:
                    axes_names[axis_idx] = sequence_name
                # 在上下文中，这意味着在对 common_outputs 中的条目进行修改之后，为了保持条目的顺序
                # 不发生变化，需要将修改过的条目重新赋值回去。如果不这样做，可能会导致有序字典的顺序发
                # 生改变，进而影响到后续依赖于这些顺序的操作
                else:
                    axes_names[axis_idx] = name
        if self.use_past:
            self.fill_with_past_key_values_(common_outputs, direction="outputs")

        return common_outputs

    @property
    def num_layers(self) -> Tuple[int]:
        try:
            # 获取父类的层数
            num_layers = super().num_layers
            # 这是编码器和解码器层数相同的情况
            num_layers = (num_layers, num_layers)
        # 当出错时,捕获
        except AttributeError:
            # 判断配置中是否有encoder_layers和decoder_layers
            if hasattr(self._config, "encoder_layers") and hasattr(self._config, "decoder_layers"):
                # 如果有,设置,这种情况两者不同
                num_layers = (self._config.encoder_layers, self._config.decoder_layers)
            else: # 其它情况,抛出错误
                raise AttributeError(
                    "could not find the number of encoder and decoder layers attributes in the model configuration,"
                    " override the num_layers property of the model OnnxConfig to solve this"
                )
        return num_layers
    @property
    def num_attention_heads(self) -> Tuple[int]:
        try:
            num_attention_heads = super().num_attention_heads
            num_attention_heads = (num_attention_heads, num_attention_heads)
        except AttributeError:
            if hasattr(self._config, "encoder_attention_heads") and hasattr(self._config, "decoder_attention_heads"):
                num_attention_heads = (self._config.encoder_attention_heads, self._config.decoder_attention_heads)
            else:
                raise AttributeError(
                    "could not find the number of attention heads for the encoder and the decoder attributes in the"
                    " model configuration, override the num_attention_heads property of the model OnnxConfig to solve"
                    " this"
                )
        return num_attention_heads
    # 这个方法通常在准备导出一个支持增量解码的编码器-解码器模型到ONNX格式时使用。
    # 生成的虚拟输入数据可以用来测试ONNX模型的行为是否与原始模型一致，也可以直接用于ONNX模型的导出过程中。
    # is_pair:如果为 True，则表示输入包含一对文本（如问题和答案）
    # framework: Optional[TensorType] = None 指定使用的框架类型，如 TensorType.TENSORFLOW 
    # 或 TensorType.PYTORCH
    # seq_length: int 序列长度。如果设置为 -1，则表示序列长度是动态的
    # batch_size: int 指定批量大小。如果设置为 -1，则表示批量大小是动态的。
    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 此步骤调用了父类的 generate_dummy_inputs 方法来生成编码器所需的虚拟输入数据。
        encoder_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )
        # 这里根据 use_past 属性来决定解码器的序列长度。如果不使用过去键值对，则解码器的序列长度与
        # 编码器相同；如果使用，则设置为 1，因为每次解码只生成一个新标记。
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=decoder_seq_length, is_pair=is_pair, framework=framework
        )
         # 然后再次调用父类的 generate_dummy_inputs 方法来生成解码器的虚拟输入数据，并且将解码器输
        # 入的键名前加上 decoder_ 来区分它们。
        # 这一步将解码器输入合并到编码器输入中，并确保键名不会冲突。
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        common_inputs = dict(**encoder_inputs, **decoder_inputs)
        # 如果模型需要使用过去键值对进行增量解码，则需要生成虚拟的过去键值对。这里首先检查是否
        # 安装了PyTorch，因为生成张量需要用到PyTorch。
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            batch = common_inputs["input_ids"].shape[0]
            encoder_seq_length = common_inputs["input_ids"].shape[1]
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            # 计算了编码器和解码器的张量形状，以便于后续生成虚拟的过去键值对。
            encoder_shape = (
                batch,
                num_encoder_attention_heads,
                encoder_seq_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            decoder_shape = (
                batch,
                num_decoder_attention_heads,
                # Not using the same length for past_key_values
                decoder_seq_length + 3,
                self._config.hidden_size // num_decoder_attention_heads,
            )
            # 这部分代码生成了过去键值对的虚拟数据。对于编码器和解码器共有的层，生成四个张量（
            # 解码器的键和值，编码器的键和值），对于只有编码器或解码器的层，则只生成两个张量。
            common_inputs["past_key_values"] = []
            # If the number of encoder and decoder layers are present in the model configuration, both are considered
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"
            for _ in range(min_num_layers):
                # For encoder-decoder models, past_key_values contains pre-computed values for both the encoder and the
                # decoder layers, hence a tuple of 4 tensors instead of 2
                common_inputs["past_key_values"].append(
                    (
                        torch.zeros(decoder_shape),
                        torch.zeros(decoder_shape),
                        torch.zeros(encoder_shape),
                        torch.zeros(encoder_shape),
                    )
                )
            # TODO: test this.
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
        # 最后返回包含所有虚拟输入数据的字典。
        return common_inputs
    # 该方法用于填充 inputs_or_outputs 映射中的过去键值对（past key-values）信息。这种方法主要用于在导出
    # ONNX模型时,确保输入或输出的形状和轴名称正确无误，尤其是在处理具有编码器-解码器结构的模型时。
    # inputs_or_outputs:这是一个映射，包含了输入或输出的名字及其对应的轴索引和轴名称。例如，
    # {"output.0": {0: "batch", 1: "sequence"}} 表示输出 output.0 的第0轴名为 batch，第1轴名为 sequence。
    # 指定是填充输入还是输出的映射，只能是 "inputs" 或 "outputs"。
    def fill_with_past_key_values_(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        # 此段代码确保 direction 参数只可能是 "inputs" 或 "outputs"，否则抛出 ValueError。
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')
        # 此段代码根据 direction 参数确定使用 past_key_values（输入方向）还是 present（输出方向）作为键值对的名字前缀。
        name = "past_key_values" if direction == "inputs" else "present"
        # If the number of encoder and decoder layers are present in the model configuration, both are considered
        # 这里获取了编码器和解码器的层数，并计算了两者的最小层数和最大层数之间的差值。还确定了哪
        # 一侧（编码器或解码器）有更多的层。
        num_encoder_layers, num_decoder_layers = self.num_layers
        # 两者间的较小者
        min_num_layers = min(num_encoder_layers, num_decoder_layers)
        # 两者间的较大者-较小者=层差
        max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
        # 如果encoder层较多,remaining_side_name= "encoder",否者是"decoder"
        remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"
        # 这里定义了编码器和解码器的序列轴名称，对于输入方向，解码器的序列轴名称是 past_decoder_sequence
        # ，而对于输出方向，则加上了当前的序列长度 + sequence。
        encoder_sequence = "past_encoder_sequence"
        decoder_sequence = "past_decoder_sequence" if direction == "inputs" else "past_decoder_sequence + sequence"
        # 此段代码填充了 inputs_or_outputs 映射，分别为编码器和解码器的键和值填充轴信息。对于较小层数内的层，
        # 同时填充编码器和解码器的信息fill_with_past_key_values_ 方法确保了在导出ONNX模型时，输入或输出映射
        # 中的轴信息正确反映了过去键值对的状态，这对于正确地导出并使用支持增量解码的编码器-解码器模型非常重要。
        for i in range(min_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.key"] = {0: "batch", 2: encoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.value"] = {0: "batch", 2: encoder_sequence}
        # 对于两者之间的层,只填充剩余侧（更多层的那一侧）的信息。
        for i in range(min_num_layers, max_num_layers):
            if remaining_side_name == "encoder":
                axes_info = {0: "batch", 2: encoder_sequence}
            else:
                axes_info = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.{remaining_side_name}.key"] = axes_info

    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        flattened_output[f"{name}.{idx}.decoder.key"] = t[0]
        flattened_output[f"{name}.{idx}.decoder.value"] = t[1]
        flattened_output[f"{name}.{idx}.encoder.key"] = t[2]
        flattened_output[f"{name}.{idx}.encoder.value"] = t[3]
# 这个类主要用于定义 T5 模型在导出为 ONNX 格式时的输入定义和默认的操作集版本。通过这种方式，可以确保在导出模型时
# ，输入的形状和轴信息被正确地定义，从而帮助生成正确的 ONNX 模型。
class T5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    # 这部分代码定义了模型的基本输入，包括 input_ids 和 attention_mask。这些输入的形状信息
    # 被定义为字典，其中键是轴索引（通常是0或1），值是轴名称（如 "batch" 或 "encoder_sequence"）。
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        # 这部分代码根据 use_past 属性的不同，处理了解码器的输入。如果模型支持使用过去键值对进
        # 行增量解码 (use_past=True)，则解码器的输入和注意力掩码的形状会有所不同，以适应增量解码的需求。
        if self.use_past:
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            common_inputs["decoder_input_ids"] = {0: "batch"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}
        # 如果模型支持使用过去键值对 (use_past=True)，则调用 fill_with_past_key_values_
        # 方法来填充过去键值对的信息。
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
        return common_inputs
    @property # 这部分代码指定了默认使用的 ONNX 操作集版本为 13。
    def default_onnx_opset(self) -> int:
        return 13


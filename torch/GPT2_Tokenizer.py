# 在Python的try...except...else...结构中，else块是可选的，并且它仅在try块中的代码成功执行
# （即没有引发任何被except子句捕获的异常）时才会执行。如果try块中的代码引发了异常，并且这个异常
# 被某个except子句捕获了，那么else块中的代码将不会被执行。
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_gpt2_fast"] = ["GPT2TokenizerFast"]
# 这个装饰器的应用意味着该类的某个方法（很可能是构造函数）将包含 INIT_TOKENIZER_DOCSTRING 中定义的
# 文档字符串。这有助于保持文档的一致性和标准化，特别是在处理预训练模型或分词器时，这些文档字符串可能包含
# 关于如何初始化分词器、需要的参数、支持的模型等重要信息。
@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizer(PreTrainedTokenizerBase):
    def __init__(self, **kwargs):
        # 1.创建了一个 Trie 对象，这个用于存储分词过程中需要快速查找的词汇。
        self.tokens_trie = Trie()
        # 2. 初始化 _added_tokens_decoder
        if not hasattr(self, "_added_tokens_decoder"):
            self._added_tokens_decoder: Dict[int, AddedToken] = {}

        # 3. 如果传入了 added_tokens_decoder，更新这个字典。
        self._added_tokens_decoder.update(kwargs.pop("added_tokens_decoder", {}))
        # _added_tokens_encoder
        self._added_tokens_encoder: Dict[str, int] = {k.content: v for v, k in self._added_tokens_decoder.items()}

        # 4 调用父类 __init__：这是关键的一步，但在这里，它应该在所有初始化步骤之后进行，因为你可能想在调用父类初始化
        # 之前完全配置好你的对象。然而，在这个上下文中，如果父类 __init__ 依赖于这些设置，则应该在设置之后调用它。由于这
        # 部分代码看起来像是子类应该先设置一些属性，因此顺序是合理的。
        super().__init__(**kwargs)
        # 4. 检查特殊标记是否已经在词汇表中，如果没有，则添加它们
        self._add_tokens(
            [token for token in self.all_special_tokens_extended if token not in self._added_tokens_encoder],
            special_tokens=True,
        )
        self._decode_use_source_tokenizer = False
    # 返回一个布尔值，表明这个分词器是否是“快速”的。在这里，它总是返回 False，因为这是慢速分词器的基类。
    @property
    def is_fast(self) -> bool:
        return False
    # 抛出了一个 NotImplementedError，这意味着子类必须实现这个方法以返回词汇表的大小
    # vocab_size 应该是一个返回分词器词汇表大小的属性。由于这个属性在基类中被定义为抛出 NotImplementedError，
    # 它实际上是在强制要求任何继承自这个基类的子类都必须实现这个属性。
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError
    # 这两个属性提供了对添加到词汇表中的token的编码器和解码器的访问。使用sorted 来确保返回的字典顺序是确定的
    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        return {k.content: v for v, k in sorted(self._added_tokens_decoder.items(), key=lambda item: item[0])}
    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        return dict(sorted(self._added_tokens_decoder.items(), key=lambda item: item[0]))
    # 这个 setter 方法确保了传入的值是正确的类型，并处理了字符串到 AddedToken 对象的转换。
    @added_tokens_decoder.setter
    def added_tokens_decoder(self, value: Dict[int, Union[AddedToken, str]]) -> Dict[int, AddedToken]:
        # Always raise an error if string because users should define the behavior
        for index, token in value.items():
            if not isinstance(token, (str, AddedToken)) or not isinstance(index, int):
                raise ValueError(
                    f"The provided `added_tokens_decoder` has an element of type {index.__class__, token.__class__}, should be a dict of {int, Union[AddedToken, str]}"
                )
            self._added_tokens_decoder[index] = AddedToken(token) if isinstance(token, str) else token
            self._added_tokens_encoder[str(token)] = index
    # 如果一个属性或方法名以单个下划线_开头，这通常被视为一种约定，表明这个属性或方法是“受保护的”或“内部使用的”，但并不严格阻止外部访问
    # 。它更多地是一种提示给开发者，说明这个属性或方法不应该被类的外部直接使用。
    def get_added_vocab(self) -> Dict[str, int]:
        return self._added_tokens_encoder
    # 这种命名方式会导致Python解释器对属性名进行“名称重整”（name mangling），即在属性名前加上_ClassName__（其
    # 中ClassName是类名，且首字母大写）。这确实提供了一种比单个下划线更强的封装方式，但仍然不是真正的私有，因为仍然
    # 可以通过特定的名称访问它。
    # 以双下划线（__）开头和结尾的方法名（称为“魔法方法”或“特殊方法”）并不是私有方法，而是有特殊用途的方法。
    def __len__(self):
        return len(set(self.get_vocab().keys()))
    # new_tokens（一个包含字符串或AddedToken对象的列表，代表要添加的新标记）和special_tokens（一个布尔值，指示是否
    # 将新标记视为特殊标记）
    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        # 初始化计数器：首先，初始化一个计数器added_tokens为0，用于记录实际添加到词汇表中的新标记数量
        added_tokens = 0
        # 检查输入：如果new_tokens为None，则直接返回0，表示没有添加任何新标记。
        if new_tokens is None:
            return added_tokens
        # 创建当前词汇表的副本：通过self.get_vocab().copy()获取当前词汇表的副本，以避免在迭代过程中修改原始词汇表。
        current_vocab = self.get_vocab().copy()
        new_idx = len(current_vocab)  # only call this once, len gives the last index + 1
        # 遍历new_tokens列表中的每个标记。对于每个标记
        for token in new_tokens:
            # 检查其类型，确保它是字符串或AddedToken对象。如果不是，则抛出TypeError
            if not isinstance(token, (str, AddedToken)):
                raise TypeError(f"Token {token} is not a string but a {type(token)}.")
            if str(token) == "": # 跳过空字符串。
                continue
            if isinstance(token, str):
                # 如果标记是字符串且已经存在于_added_tokens_encoder中，则跳过它。
                if token in self._added_tokens_encoder:
                    continue
                else:
                    # 如果标记是字符串且不在_added_tokens_encoder中，则根据special_tokens参数和all_special_tokens
                    # 列表的内容，将其转换为AddedToken对象，并设置其special和normalized属性。
                    is_special = token in self.all_special_tokens or special_tokens
                    token = AddedToken(
                        token, rstrip=False, lstrip=False, normalized=not is_special, special=is_special
                    )
            # 如果标记是AddedToken对象且special_tokens为True，则更新其special和normalized属性。
            elif special_tokens:
                # doing token.special=True changes the normalization! will fix in rust
                # this is important and the only reason why the AddedTokens in each class are normalized by default
                token.__setstate__({"special": True, "normalized": token.normalized})
            # 如果标记（无论是字符串还是AddedToken对象）已经存在于_added_tokens_decoder中，则跳过它。
            if token in self._added_tokens_decoder:
                continue
            # 如果标记不是特殊标记、需要被标准化，并且分词器被配置为执行小写转换（do_lower_case为True）
            # ，则将标记的内容转换为小写。
            if not token.special and token.normalized and getattr(self, "do_lower_case", False):
                # Normalize if requested
                token.content = token.content.lower()
            # 如果标记的内容不在当前词汇表中，则将其添加到词汇表中
            if token.content not in current_vocab:
                token_index = new_idx + added_tokens # 新token的idx
                current_vocab[token.content] = token_index # word-->idx
                added_tokens += 1 # 索引加1
            else: 
                token_index = current_vocab[token.content]
            # 如果标记是特殊标记且不在all_special_tokens列表中，则将其添加到
            # _additional_special_tokens列表中。
            if token.special and str(token) not in self.all_special_tokens:
                self._additional_special_tokens.append(token)
            # 更新分词器的其他部分
            self._added_tokens_decoder[token_index] = token
            self._added_tokens_encoder[token.content] = token_index
            # 如果启用了详细模式（self.verbose为True），则在添加每个新标记时都会记录一条日志信息
            if self.verbose:
                logger.info(f"Adding {token} to the vocabulary")
        # 调用_update_trie()方法,可能是为了更新分词器内部使用的数据结构（如前缀树/Trie）以反映新添加的标记。
        self._update_trie()
        return added_tokens
    # 更新trie
    def _update_trie(self, unique_no_split_tokens: Optional[str] = []):
        for token in self._added_tokens_decoder.values():
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token.content)
        for token in unique_no_split_tokens:
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token)
    # 这个方法的主要目的是计算并返回在准备模型输入时需要添加的特殊标记的数量
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))
    # 用于将给定的文本 text 转换成一系列标记（tokens）
    # 待处理的文本输入,**kwargs:关键字参数，用于传递额外的配置选项。
    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        # 一个布尔值，用于控制是否拆分特殊标记。这个值首先尝试从 kwargs 中获取，如果未提供，则使用实例变量
        # self.split_special_tokens 的值。
        split_special_tokens = kwargs.pop("split_special_tokens", self.split_special_tokens)
        # 调用 self.prepare_for_tokenization(text, **kwargs) 方法对文本进行预处理，可能包括去除额外的空格
        # 、处理换行符等。同时，这个方法也会更新 kwargs 字典，移除已处理的参数。
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)
        # 如果 kwargs 字典在预处理后仍然不为空，说明有未识别的参数，此时会记录一条警告日志。
        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")
        # 如果实例变量 self.do_lower_case 为 True，则将所有非特殊标记的文本转换为小写
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase. Might be super slow as well?
            # 遍历 self.all_special_tokens 列表，对每个特殊标记使用 re.escape 进行转义，然后将这些转义后的
            # 特殊标记添加到 escaped_special_toks 列表中
            escaped_special_toks = [re.escape(s_tok) for s_tok in (self.all_special_tokens)]
            # 遍历 self._added_tokens_decoder.values()，对于每个对象，如果它不是特殊的（not s_tok.special）
            # 且已规范化（s_tok.normalized），则将其 content 属性进行转义并添加到 escaped_special_toks 列表中。
            escaped_special_toks += [
                re.escape(s_tok.content)
                for s_tok in (self._added_tokens_decoder.values())
                if not s_tok.special and s_tok.normalized
            ]
            # 这个模式匹配任何在 escaped_special_toks 列表中的特殊标记（作为捕获组），或者任意
            # 字符序列（至少一个字符，非贪婪匹配），作为另一个捕获组。
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            # 使用 re.sub 函数和上述模式替换文本中的匹配项
            # lambda 函数用于处理每个匹配项。如果匹配到的是特殊标记（即 m.groups()[0] 不为 None）
            # ，则直接返回该特殊标记（因为 re.escape 已经确保了这些标记在正则表达式中不会被错误解释）。
            # 如果匹配到的是任意字符序列（即 m.groups()[0] 为 None），则将这部分文本转换为小写并返回。
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)
        # 如果 split_special_tokens 为 True
        if split_special_tokens:
            # no_split_token 被设置为空列表，因为在这个模式下我们不关心哪些标记不应该被拆分
            no_split_token = []
            # tokens 被设置为包含整个文本 text 的单一元素列表，即 [text]。这意味着整个文本被
            # 视为一个单独的“标记”，无论它包含什么特殊标记还是普通文本，都不会被拆分
            tokens = [text]
        else:# 如果 split_special_tokens 为 False
            # no_split_token 被设置为 self._added_tokens_encoder.keys()，这是一个包含所有不应
            # 被拆分的特殊标记的键的集合。
            no_split_token = self._added_tokens_encoder.keys()  # don't split on any of the added tokens
            # "This is something<special_token_1>  else"
            # 使用 self.tokens_trie.split(text) 来拆分文本
            tokens = self.tokens_trie.split(text)
        # ["This is something", "<special_token_1>", "  else"]
        # 遍历拆分后的标记列表，对于不应拆分的特殊标记，根据 AddedToken 实例的属性（如 rstrip、
        # lstrip、single_word）来调整相邻标记的空格。
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = self._added_tokens_decoder.get(self._added_tokens_encoder[token], None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                    if tok_extended.single_word and left and left[-1] != " ":
                        tokens[i - 1] += token
                        tokens[i] = ""
                    elif tok_extended.single_word and right and right[0] != " ":
                        tokens[i + 1] = token + tokens[i + 1]
                        tokens[i] = ""
                else:
                    raise ValueError(
                        f"{tok_extended} cannot be tokenized because it was not properly added"
                        f" to the tokenizer. This means that it is not an `AddedToken` but a {type(tok_extended)}"
                    )
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids
    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None
        if token in self._added_tokens_encoder:
            return self._added_tokens_encoder[token]
        return self._convert_token_to_id(token)
    def _convert_token_to_id(self, token):
        raise NotImplementedError
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                if is_split_into_words:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string or a list/tuple of strings when"
                        " `is_split_into_words=True`."
                    )
                else:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of"
                        " integers."
                    )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        first_ids = get_input_ids(text) # 获取input_ids
        second_ids = get_input_ids(text_pair) if text_pair is not None else None
        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
            split_special_tokens=split_special_tokens,
        )

        return BatchEncoding(batch_outputs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
    ) -> BatchEncoding:
        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
                split_special_tokens=split_special_tokens,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        return (text, kwargs)

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str: ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]: ...

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            if ids in self._added_tokens_decoder:
                return self._added_tokens_decoder[ids].content
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self._added_tokens_decoder:
                tokens.append(self._added_tokens_decoder[index].content)
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        legacy_added_tokens = set(self._added_tokens_encoder.keys()) - set(self.all_special_tokens) | {
            token for token in self.additional_special_tokens if self.convert_tokens_to_ids(token) >= self.vocab_size
        }
        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        # TODO @ArthurZ in version 5, special tokens should be handled in convert_tokens_to_string, while _convert_tokens_to_string
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in legacy_added_tokens:
                if current_sub_text:
                    string = self.convert_tokens_to_string(current_sub_text)
                    if len(string) > 0:
                        sub_texts.append(string)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

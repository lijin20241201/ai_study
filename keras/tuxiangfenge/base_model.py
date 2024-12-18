class TFPreTrainedModel(keras.Model, TFModelUtilsMixin, TFGenerationMixin, PushToHubMixin):
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    _auto_class = None
    _using_dummy_loss = None
    _label_to_output_map = None

    # a list of re pattern of tensor names to ignore from the model when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_missing = None
    # a list of re pattern of tensor names to ignore from the weights when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_unexpected = None
    _requires_load_weight_prefix = False

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        dummies = {}
        for key, spec in self.input_signature.items():
            # 2 is the most correct arbitrary size. I will not be taking questions
            dummy_shape = [dim if dim is not None else 2 for dim in spec.shape]
            if spec.shape[0] is None:
                # But let's make the batch size 1 to save memory anyway
                dummy_shape[0] = 1
            dummies[key] = tf.ones(shape=dummy_shape, dtype=spec.dtype)
            if key == "token_type_ids":
                # Some models have token_type_ids but with a vocab_size of 1
                dummies[key] = tf.zeros_like(dummies[key])
        if self.config.add_cross_attention and "encoder_hidden_states" in inspect.signature(self.call).parameters:
            if "encoder_hidden_states" not in dummies:
                if self.main_input_name == "input_ids":
                    dummies["encoder_hidden_states"] = tf.ones(
                        shape=(1, 2, self.config.hidden_size), dtype=tf.float32, name="encoder_hidden_states"
                    )
                else:
                    raise NotImplementedError(
                        "Model has cross-attention but we couldn't infer the shape for the encoder hidden states. Please manually override dummy_inputs!"
                    )
        return dummies

    def build_in_name_scope(self):
        with tf.name_scope(self.name):
            self.build(input_shape=None)

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a TensorFlow model.
        """
        return "tf"

    def build(self, input_shape=None):
        pass  # This is just here to make sure we don't call the superclass build()

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Save config and origin of the pretrained weights if given in model
        self.config = config
        self.name_or_path = config.name_or_path
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        self._set_save_spec(self.input_signature)

    def get_config(self):
        return self.config.to_dict()

    @functools.wraps(keras.Model.fit)
    def fit(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().fit(*args, **kwargs)

    @functools.wraps(keras.Model.train_on_batch)
    def train_on_batch(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().train_on_batch(*args, **kwargs)

    @functools.wraps(keras.Model.test_on_batch)
    def test_on_batch(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().test_on_batch(*args, **kwargs)

    @functools.wraps(keras.Model.predict_on_batch)
    def predict_on_batch(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().predict_on_batch(*args, **kwargs)

    @functools.wraps(keras.Model.predict)
    def predict(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().predict(*args, **kwargs)

    @functools.wraps(keras.Model.evaluate)
    def evaluate(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().evaluate(*args, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        if isinstance(config, PretrainedConfig):
            return cls._from_config(config, **kwargs)
        return cls._from_config(cls.config_class.from_dict(config, **kwargs))

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.
        """
        return cls(config, **kwargs)

    def get_head_mask(self, head_mask: tf.Tensor | None, num_hidden_layers: int) -> tf.Tensor:
        
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.shape.rank == 1:
            head_mask = head_mask[None, None, :, None, None]
            head_mask = tf.repeat(head_mask, repeats=num_hidden_layers, axis=0)
        elif head_mask.shape.rank == 2:
            head_mask = head_mask[:, None, :, None, None]
        assert head_mask.shape.rank == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = tf.cast(head_mask, tf.float32)  # switch to float if need + fp16 compatibility
        return head_mask

    @tf.function
    def serving(self, inputs):
        
        output = self.call(inputs)

        return self.serving_output(output)

    @property
    def input_signature(self) -> Dict[str, tf.TensorSpec]:
        
        model_inputs = list(inspect.signature(self.call).parameters)
        sig = {}
        if "input_ids" in model_inputs:
            if self.__class__.__name__.endswith("ForMultipleChoice"):
                text_dims = 3
            else:
                text_dims = 2
            for input_name in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "decoder_input_ids",
                "decoder_attention_mask",
            ):
                if input_name in model_inputs:
                    sig[input_name] = tf.TensorSpec([None] * text_dims, tf.int32, name=input_name)
        if "pixel_values" in model_inputs:
            pixel_values_shape = [None, None, None, None]
            if hasattr(self.config, "vision_config"):
                vision_config = self.config.vision_config
            else:
                vision_config = self.config
            if hasattr(vision_config, "num_channels"):
                pixel_values_shape[1] = vision_config.num_channels
            else:
                raise NotImplementedError(
                    "Could not infer number of channels from config, please override input_signature to specify input shapes."
                )
            if hasattr(vision_config, "image_size"):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.image_size
            elif hasattr(vision_config, "input_size"):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.input_size
            else:
                raise NotImplementedError(
                    "Could not infer input image shape from config, please override input_signature to specify input shapes."
                )
            sig["pixel_values"] = tf.TensorSpec(pixel_values_shape, tf.float32, name="pixel_values")
        if "input_features" in model_inputs:
            raise NotImplementedError("Audio models need a manually defined input_signature")
        return sig

    def serving_output(self, output):
       
        if not isinstance(output, ModelOutput):
            return output
        for key in output:
            if key.endswith("hidden_states") and not getattr(self.config, "output_hidden_states", False):
                output[key] = None
            elif key.endswith("attentions") and not getattr(self.config, "output_attentions", False):
                output[key] = None
            elif key == "past_key_values" and not getattr(self.config, "use_cache", False):
                output[key] = None
            elif key == "cross_attentions" and not (
                getattr(self.config, "output_attentions", False) and getattr(self.config, "add_cross_attention", False)
            ):
                output[key] = None
            if isinstance(output[key], (tuple, list)):
                try:
                    output[key] = tf.convert_to_tensor(output[key])
                except (ValueError, tf.errors.InvalidArgumentError):
                    pass  # Layers may not have the same dimensions
        return output

    @classmethod
    def can_generate(cls) -> bool:
        
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation.
        # Alternativelly, the model can also have a custom `generate` function.
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            return False
        return True

    def get_input_embeddings(self) -> keras.layers.Layer:
       
        main_layer = getattr(self, self.base_model_prefix, self)

        if main_layer is not self:
            return main_layer.get_input_embeddings()
        else:
            raise NotImplementedError

    def _save_checkpoint(self, checkpoint_dir, epoch):
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        # We avoid tf.train.checkpoint or saving weights in TF format, even though that includes optimizer
        # state for us, because it requires special handling for objects like custom losses, which we use
        # internally and which users are likely to use too
        weights_path = os.path.join(checkpoint_dir, "weights.h5")
        self.save_weights(weights_path)
        extra_data = {"epoch": epoch, "optimizer_state": self.optimizer.get_weights()}
        extra_data_path = os.path.join(checkpoint_dir, "extra_data.pickle")
        with open(extra_data_path, "wb") as f:
            pickle.dump(extra_data, f)

    def prepare_tf_dataset(
        self,
        dataset: "datasets.Dataset",  # noqa:F821
        batch_size: int = 8,
        shuffle: bool = True,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        collate_fn: Optional[Callable] = None,
        collate_fn_args: Optional[Dict[str, Any]] = None,
        drop_remainder: Optional[bool] = None,
        prefetch: bool = True,
    ):
       
        requires_backends(self, ["datasets"])
        import datasets

        if collate_fn is None:
            if tokenizer is None:
                collate_fn = DefaultDataCollator(return_tensors="np")
            else:
                collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="np")
        if collate_fn_args is None:
            collate_fn_args = {}

        if not isinstance(dataset, datasets.Dataset):
            raise TypeError("Dataset argument should be a datasets.Dataset!")
        model_inputs = list(inspect.signature(self.call).parameters)
        model_labels = find_labels(self.__class__)
        if "cols_to_retain" in list(inspect.signature(dataset._get_output_signature).parameters.keys()):
            output_signature, _ = dataset._get_output_signature(
                dataset,
                batch_size=None,
                collate_fn=collate_fn,
                collate_fn_args=collate_fn_args,
                cols_to_retain=model_inputs,
            )
        else:
            # TODO Matt: This is a workaround for older versions of datasets that are missing the `cols_to_retain`
            #            argument. We should remove this once the minimum supported version of datasets is > 2.3.2
            unwanted_columns = [
                feature
                for feature in dataset.features
                if feature not in model_inputs and feature not in ("label_ids", "label")
            ]
            dataset = dataset.remove_columns(unwanted_columns)
            output_signature, _ = dataset._get_output_signature(
                dataset, batch_size=None, collate_fn=collate_fn, collate_fn_args=collate_fn_args
            )
        output_columns = list(output_signature.keys())
        feature_cols = [col for col in output_columns if col in model_inputs and col not in model_labels]
        label_cols = [col for col in output_columns if col in model_labels]

        # Backwards compatibility for older versions of datasets. Previously, if `columns` or `label_cols`
        # were a single element list, the returned element spec would be a single element. Now, passing [feature]
        # will return a dict structure {"feature": feature}, and passing a single string will return a single element.
        feature_cols = feature_cols[0] if len(feature_cols) == 1 else feature_cols
        label_cols = label_cols[0] if len(label_cols) == 1 else label_cols

        if drop_remainder is None:
            drop_remainder = shuffle
        tf_dataset = dataset.to_tf_dataset(
            columns=feature_cols,
            label_cols=label_cols,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
            collate_fn=collate_fn,
            collate_fn_args=collate_fn_args,
            prefetch=prefetch,
        )
        return tf_dataset

    def compile(
        self,
        optimizer="rmsprop",
        loss="auto_with_warning",
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs,
    ):
        
        if loss in ("auto_with_warning", "passthrough"):  # "passthrough" for workflow backward compatibility
            logger.info(
                "No loss specified in compile() - the model's internal loss computation will be used as the "
                "loss. Don't panic - this is a common way to train TensorFlow models in Transformers! "
                "To disable this behaviour please pass a loss argument, or explicitly pass "
                "`loss=None` if you do not want your model to compute a loss. You can also specify `loss='auto'` to "
                "get the internal loss without printing this info string."
            )
            loss = "auto"
        if loss == "auto":
            loss = dummy_loss
            self._using_dummy_loss = True
        else:
            self._using_dummy_loss = False
        parent_args = list(inspect.signature(keras.Model.compile).parameters.keys())
        # This argument got renamed, we need to support both versions
        if "steps_per_execution" in parent_args:
            super().compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                weighted_metrics=weighted_metrics,
                run_eagerly=run_eagerly,
                steps_per_execution=steps_per_execution,
                **kwargs,
            )
        else:
            super().compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                weighted_metrics=weighted_metrics,
                run_eagerly=run_eagerly,
                experimental_steps_per_execution=steps_per_execution,
                **kwargs,
            )

    def compute_loss(self, *args, **kwargs):
        if hasattr(keras.Model, "compute_loss"):
            # This will be true in TF 2.8 or greater
            return super().compute_loss(*args, **kwargs)
        else:
            warnings.warn(
                "The old compute_loss method is deprecated as it conflicts with the Keras compute_loss "
                "method added in TF 2.8. If you want the original HF compute_loss, please call "
                "hf_compute_loss() instead. From TF versions >= 2.8, or Transformers versions >= 5, "
                "calling compute_loss() will get the Keras method instead.",
                FutureWarning,
            )
            return self.hf_compute_loss(*args, **kwargs)

    def get_label_to_output_name_mapping(self):
        arg_names = list(inspect.signature(self.call).parameters)
        if self._label_to_output_map is not None:
            return self._label_to_output_map
        elif "start_positions" in arg_names:
            return {"start_positions": "start_logits", "end_positions": "end_logits"}
        elif "sentence_order_label" in arg_names:
            return {"labels": "prediction_logits", "sentence_order_label": "sop_logits"}
        elif "next_sentence_label" in arg_names:
            return {"labels": "prediction_logits", "next_sentence_label": "seq_relationship_logits"}
        elif "mc_labels" in arg_names:
            return {"labels": "logits", "mc_labels": "mc_logits"}
        else:
            return {}

    def train_step(self, data):
        # We hardcode the most common renamings; models with weirder names can set `self._label_to_output_map`
        arg_names = list(inspect.signature(self.call).parameters)
        label_kwargs = find_labels(self.__class__)
        label_to_output = self.get_label_to_output_name_mapping()
        output_to_label = {val: key for key, val in label_to_output.items()}
        if not self._using_dummy_loss and parse(tf.__version__) < parse("2.11.0"):
            # Newer TF train steps leave this out
            data = expand_1d(data)
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        # If the inputs are mutable dictionaries, make a shallow copy of them because we will modify
        # them during input/label pre-processing. This avoids surprising the user by wrecking their data.
        # In addition, modifying mutable Python inputs makes XLA compilation impossible.
        if isinstance(x, dict):
            x = x.copy()
        if isinstance(y, dict):
            y = y.copy()

        # When using a dummy loss, we ensure that separate labels are copied to the correct model arguments,
        # if those keys are not already present in the input dict
        if self._using_dummy_loss and y is not None:
            # If y is a tensor and the model only has one label-like input, map y to that input
            if len(label_kwargs) == 1 and isinstance(y, tf.Tensor):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                label_kwarg = next(iter(label_kwargs))
                if label_kwarg not in x:
                    x[label_kwarg] = y
            # Otherwise, copy keys from y to x as long as they weren't already present in x
            elif isinstance(y, dict):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                for key, val in y.items():
                    if key in arg_names and key not in x:
                        x[key] = val
                    elif output_to_label.get(key, None) in arg_names and key not in x:
                        x[output_to_label[key]] = val
        if y is None:
            y = {key: val for key, val in x.items() if key in label_kwargs}
            if not y and not self._using_dummy_loss:
                raise ValueError("Could not find label column(s) in input dict and no separate labels were provided!")

        if isinstance(y, dict):
            # Rename labels at this point to match output heads
            y = {label_to_output.get(key, key): val for key, val in y.items()}

        # Run forward pass.
        with tf.GradientTape() as tape:
            if self._using_dummy_loss and "return_loss" in arg_names:
                y_pred = self(x, training=True, return_loss=True)
            else:
                y_pred = self(x, training=True)
            if self._using_dummy_loss:
                loss = self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
            else:
                loss = None

            # This next block matches outputs to label keys. Tensorflow's standard method for doing this
            # can get very confused if any of the keys contain nested values (e.g. lists/tuples of Tensors)
            if isinstance(y, dict) and len(y) == 1:
                if list(y.keys())[0] in y_pred.keys():
                    y_pred = y_pred[list(y.keys())[0]]
                elif list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred[1]
                else:
                    y_pred = y_pred[0]
                _, y = y.popitem()
            elif isinstance(y, dict):
                # If the labels are a dict, match keys from the output by name
                y_pred = {key: val for key, val in y_pred.items() if key in y}
            elif isinstance(y, tuple) or isinstance(y, list):
                # If the labels are a tuple/list, match keys to the output by order, skipping the loss.
                if list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred.to_tuple()[1:]
                else:
                    y_pred = y_pred.to_tuple()
                y_pred = y_pred[: len(y)]  # Remove unused fields in case those cause problems
            else:
                # If the labels are a single tensor, match them to the first non-loss tensor in the output
                if list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred[1]
                else:
                    y_pred = y_pred[0]

            if loss is None:
                loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def test_step(self, data):
        # We hardcode the most common renamings; models with weirder names can set `self._label_to_output_map`
        arg_names = list(inspect.signature(self.call).parameters)
        label_kwargs = find_labels(self.__class__)
        label_to_output = self.get_label_to_output_name_mapping()
        output_to_label = {val: key for key, val in label_to_output.items()}
        if not self._using_dummy_loss and parse(tf.__version__) < parse("2.11.0"):
            # Newer versions leave this out
            data = expand_1d(data)
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        # If the inputs are mutable dictionaries, make a shallow copy of them because we will modify
        # them during input/label pre-processing. This avoids surprising the user by wrecking their data.
        # In addition, modifying mutable Python inputs makes XLA compilation impossible.
        if isinstance(x, dict):
            x = x.copy()
        if isinstance(y, dict):
            y = y.copy()
        # When using a dummy loss, we ensure that separate labels are copied to the correct model arguments,
        # if those keys are not already present in the input dict
        if self._using_dummy_loss and y is not None:
            arg_names = list(inspect.signature(self.call).parameters)
            # If y is a tensor and the model only has one label-like input, map y to that input
            if len(label_kwargs) == 1 and isinstance(y, tf.Tensor):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                label_kwarg = next(iter(label_kwargs))
                if label_kwarg not in x:
                    x[label_kwarg] = y
            # Otherwise, copy keys from y to x as long as they weren't already present in x
            elif isinstance(y, dict):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                for key, val in y.items():
                    if key in arg_names and key not in x:
                        x[key] = val
                    elif output_to_label.get(key, None) in arg_names and key not in x:
                        x[output_to_label[key]] = val
        if y is None:
            y = {key: val for key, val in x.items() if key in label_kwargs}
            if not y and not self._using_dummy_loss:
                raise ValueError("Could not find label column(s) in input dict and no separate labels were provided!")

        if isinstance(y, dict):
            # Rename labels at this point to match output heads
            y = {label_to_output.get(key, key): val for key, val in y.items()}

        # Run forward pass.
        if self._using_dummy_loss and "return_loss" in arg_names:
            y_pred = self(x, return_loss=True, training=False)
        else:
            y_pred = self(x, training=False)
        if self._using_dummy_loss:
            loss = self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
        else:
            loss = None

        # This next block matches outputs to label keys. Tensorflow's standard method for doing this
        # can get very confused if any of the keys contain nested values (e.g. lists/tuples of Tensors)
        if isinstance(y, dict) and len(y) == 1:
            if list(y.keys())[0] in y_pred.keys():
                y_pred = y_pred[list(y.keys())[0]]
            elif list(y_pred.keys())[0] == "loss":
                y_pred = y_pred[1]
            else:
                y_pred = y_pred[0]
            _, y = y.popitem()
        elif isinstance(y, dict):
            # If the labels are a dict, match keys from the output by name
            y_pred = {key: val for key, val in y_pred.items() if key in y}
        elif isinstance(y, tuple) or isinstance(y, list):
            # If the labels are a tuple/list, match keys to the output by order, skipping the loss.
            if list(y_pred.keys())[0] == "loss":
                y_pred = y_pred.to_tuple()[1:]
            else:
                y_pred = y_pred.to_tuple()
            y_pred = y_pred[: len(y)]  # Remove unused fields in case those cause problems
        else:
            # If the labels are a single tensor, match them to the first non-loss tensor in the output
            if list(y_pred.keys())[0] == "loss":
                y_pred = y_pred[1]
            else:
                y_pred = y_pred[0]

        if loss is None:
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def create_model_card(
        self,
        output_dir,
        model_name: str,
        language: Optional[str] = None,
        license: Optional[str] = None,
        tags: Optional[str] = None,
        finetuned_from: Optional[str] = None,
        tasks: Optional[str] = None,
        dataset_tags: Optional[Union[str, List[str]]] = None,
        dataset: Optional[Union[str, List[str]]] = None,
        dataset_args: Optional[Union[str, List[str]]] = None,
    ):
        
        # Avoids a circular import by doing this when necessary.
        from .modelcard import TrainingSummary  # tests_ignore

        training_summary = TrainingSummary.from_keras(
            self,
            keras_history=self.history,
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
        )
        model_card = training_summary.to_model_card()
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(model_card)

    def set_input_embeddings(self, value):
       
        main_layer = getattr(self, self.base_model_prefix)

        if main_layer is None:
            raise NotImplementedError("The model does not implements the base_model_prefix attribute.")

        try:
            main_layer.set_input_embeddings(value)
        except AttributeError:
            logger.info("Building the model")
            self.build_in_name_scope()
            main_layer.set_input_embeddings(value)

    def get_output_embeddings(self) -> Union[None, keras.layers.Layer]:
        
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()

            try:
                return lm_head.get_output_embeddings()
            except AttributeError:
                logger.info("Building the model")
                self.build_in_name_scope()

                return lm_head().get_output_embeddings()

        return None  # Overwrite for models with output embeddings

    def set_output_embeddings(self, value):
        
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                lm_head.set_output_embeddings(value)
            except AttributeError:
                logger.info("Building the model")
                self.build_in_name_scope()
                lm_head.set_output_embeddings(value)

    def get_output_layer_with_bias(self) -> Union[None, keras.layers.Layer]:
        
        warnings.warn(
            "The method get_output_layer_with_bias is deprecated. Please use `get_lm_head` instead.", FutureWarning
        )
        return self.get_lm_head()

    def get_prefix_bias_name(self) -> Union[None, str]:
       
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return None

    def get_bias(self) -> Union[None, Dict[str, tf.Variable]]:
       
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                return lm_head.get_bias()
            except AttributeError:
                self.build_in_name_scope()

                return lm_head.get_bias()
        return None

    def set_bias(self, value):
       
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                lm_head.set_bias(value)
            except AttributeError:
                self.build_in_name_scope()
                lm_head.set_bias(value)

    def get_lm_head(self) -> keras.layers.Layer:
       
        return None

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None
    ) -> Union[keras.layers.Embedding, tf.Variable]:
        
        # TODO (joao): flagged for replacement (by `_v2_resized_token_embeddings`) due to embeddings refactor

        # Run the new code path if the model has a keras embeddings layer
        if isinstance(self.get_input_embeddings(), keras.layers.Embedding):
            return self._v2_resized_token_embeddings(new_num_tokens)

        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self._get_word_embedding_weight(self.get_input_embeddings())

        model_embeds = self._resize_token_embeddings(new_num_tokens)

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens

        return model_embeds

    def _v2_resized_token_embeddings(self, new_num_tokens: Optional[int] = None) -> keras.layers.Embedding:
       
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self.get_input_embeddings()

        model_embeds = self._v2_resize_token_embeddings(new_num_tokens)

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens

        return model_embeds

    def _get_word_embedding_weight(model, embedding_layer):
        # TODO (joao): flagged for delection due to embeddings refactor

        # If the variable holds the weights themselves, return them
        if isinstance(embedding_layer, tf.Tensor):
            return embedding_layer
        # Otherwise, try to get them from the layer's attributes

        embeds = getattr(embedding_layer, "weight", None)
        if embeds is not None:
            return embeds

        embeds = getattr(embedding_layer, "decoder", None)
        if embeds is not None:
            return embeds

        # The reason why the attributes don't exist might be
        # because the model is not built, so retry getting
        # the argument after building the model
        model.build_in_name_scope()

        embeds = getattr(embedding_layer, "weight", None)
        if embeds is not None:
            return embeds

        embeds = getattr(embedding_layer, "decoder", None)
        if embeds is not None:
            return embeds

        return None

    def _resize_token_embeddings(self, new_num_tokens):
        # TODO (joao): flagged for replacement (by `_v2_resize_token_embeddings`) due to embeddings refactor
        old_embeddings = self._get_word_embedding_weight(self.get_input_embeddings())
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)

        # if word embeddings are not tied, make sure that lm head bias is resized as well
        if self.get_bias() is not None:
            old_lm_head_bias = self.get_bias()
            new_lm_head_bias = self._get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)

            self.set_bias(new_lm_head_bias)

        # if word embeddings are not tied, make sure that lm head decoder is resized as well
        if self.get_output_embeddings() is not None:
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)

            self.set_output_embeddings(new_lm_head_decoder)

        self.set_input_embeddings(new_embeddings)

        return self.get_input_embeddings()

    def _v2_resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._v2_get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # If word embeddings are not tied, make sure that lm head bias is resized as well
        if self.get_bias() is not None:
            old_lm_head_bias = self.get_bias()
            new_lm_head_bias = self._v2_get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)
            self.set_bias(new_lm_head_bias)

        # If word embeddings are not tied, make sure that lm head decoder is resized as well.
        tied_weights = self.get_input_embeddings() == self.get_output_embeddings()
        if self.get_output_embeddings() is not None and not tied_weights:
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            # TODO (joao): this one probably needs a v2 version with other models
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)
            self.set_output_embeddings(new_lm_head_decoder)

        return self.get_input_embeddings()

    def _get_resized_lm_head_bias(self, old_lm_head_bias, new_num_tokens):
       
        # TODO (joao): flagged for replacement (by `_v2_get_resized_lm_head_bias`) due to embeddings refactor
        new_lm_head_bias = {}

        for attr, weight in old_lm_head_bias.items():
            first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            size_diff = new_num_tokens - old_num_tokens
            final_shape = [new_num_tokens] if first_dim is None else [first_dim, new_num_tokens]

            # initialize new bias
            if tf.math.greater(size_diff, 0):
                padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
                current_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape), constant_values=-1)
                num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
                mask_shape = [num_tokens_to_copy] if first_dim is None else [1, num_tokens_to_copy]
                bias_mask = tf.fill(tf.convert_to_tensor(mask_shape), True)
                bias_mask = tf.pad(bias_mask, tf.convert_to_tensor(padding_shape), constant_values=False)
            else:
                slice_from = [0] if first_dim is None else [0, 0]
                current_bias = tf.slice(
                    weight.value(), tf.convert_to_tensor(slice_from), tf.convert_to_tensor(final_shape)
                )
                bias_mask = tf.fill(tf.convert_to_tensor(final_shape), True)

            new_bias = self.add_weight(
                shape=final_shape,
                initializer="zeros",
                trainable=True,
                name=weight.name.split(":")[0],
            )
            init_bias = tf.where(bias_mask, current_bias, new_bias.value())

            new_bias.assign(init_bias)
            new_lm_head_bias[attr] = new_bias

        return new_lm_head_bias

    def _v2_get_resized_lm_head_bias(
        self, old_lm_head_bias: Dict[str, tf.Variable], new_num_tokens: int
    ) -> Dict[str, tf.Tensor]:
       
        new_lm_head_bias = {}

        for attr, weight in old_lm_head_bias.items():
            # Determine the size difference (depending on the shape)
            first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            size_diff = new_num_tokens - old_num_tokens

            # Copy the old bias values to the new bias
            if old_num_tokens > new_num_tokens:
                new_bias = weight.value()[..., :new_num_tokens]
            else:
                padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
                new_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape))

            new_lm_head_bias[attr] = new_bias
        return new_lm_head_bias

    def _get_resized_lm_head_decoder(self, old_lm_head_decoder, new_num_tokens):
        
        new_lm_head_decoder = old_lm_head_decoder
        is_input_output_equals = tf.reduce_any(
            self._get_word_embedding_weight(self.get_input_embeddings()) == old_lm_head_decoder
        )

        if old_lm_head_decoder is not None and not is_input_output_equals:
            old_embedding_dim = shape_list(old_lm_head_decoder)[1]
            decoder_mask, current_decoder = init_copy_embeddings(old_lm_head_decoder, new_num_tokens)
            new_lm_head_decoder = self.add_weight(
                shape=(new_num_tokens, old_embedding_dim),
                initializer="zeros",
                trainable=True,
                name=old_lm_head_decoder.name.split(":")[0],
            )
            init_decoder = tf.where(decoder_mask, current_decoder, new_lm_head_decoder.value())

            new_lm_head_decoder.assign(init_decoder)

        return new_lm_head_decoder

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None) -> tf.Variable:
       
        # TODO (joao): flagged for replacement (by `_v2_get_resized_embeddings`) due to embeddings refactor
        old_embedding_dim = shape_list(old_embeddings)[1]
        init_range = getattr(self.config, "initializer_range", 0.02)
        embeddings_mask, current_embeddings = init_copy_embeddings(old_embeddings, new_num_tokens)
        new_embeddings = self.add_weight(
            name=old_embeddings.name.split(":")[0],
            shape=[new_num_tokens, old_embedding_dim],
            initializer=get_initializer(init_range),
            dtype=tf.float32,
        )
        init_embeddings = tf.where(embeddings_mask, current_embeddings, new_embeddings.value())

        new_embeddings.assign(init_embeddings)

        return new_embeddings

    def _v2_get_resized_embeddings(
        self, old_embeddings: keras.layers.Embedding, new_num_tokens: int
    ) -> keras.layers.Embedding:
       

        # Get the initialization range for the embeddings
        init_range = 0.02  # default value
        potential_initialization_variable_names = [
            "initializer_range",  # most common
            "initializer_factor",  # e.g. T5
            "init_std",  # e.g BART
        ]
        for var_name in potential_initialization_variable_names:
            if hasattr(self.config, var_name):
                init_range = getattr(self.config, var_name)

        # Get a new (initialized) embeddings layer
        new_embeddings = keras.layers.Embedding(
            input_dim=new_num_tokens,
            output_dim=old_embeddings.output_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=init_range),
            name=old_embeddings.embeddings.name[:-13],  # exact same scoped name except "/embeddings:0"
        )
        new_embeddings(tf.constant([[0]]))

        # Copy the old embeddings to the new embeddings
        if old_embeddings.input_dim >= new_num_tokens:
            init_embeddings = old_embeddings.embeddings[:new_num_tokens]
        else:
            init_embeddings = tf.concat(
                [old_embeddings.embeddings, new_embeddings.embeddings[old_embeddings.input_dim :]], axis=0
            )
        new_embeddings.embeddings.assign(init_embeddings)
        return new_embeddings

    def prune_heads(self, heads_to_prune):
        
        raise NotImplementedError

    def save_pretrained(
        self,
        save_directory,
        saved_model=False,
        version=1,
        push_to_hub=False,
        signatures=None,
        max_shard_size: Union[int, str] = "5GB",
        create_pr: bool = False,
        safe_serialization: bool = False,
        token: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        
        use_auth_token = kwargs.pop("use_auth_token", None)

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

        if token is not None:
            kwargs["token"] = token

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        if saved_model:
            # If `torch_dtype` is in the config with a torch dtype class as the value, we need to change it to string.
            # (Although TF doesn't care about this attribute, we can't just remove it or set it to `None`.)
            if getattr(self.config, "torch_dtype", None) is not None and not isinstance(self.config.torch_dtype, str):
                self.config.torch_dtype = str(self.config.torch_dtype).split(".")[1]
            if signatures is None:
                serving_default = self.serving.get_concrete_function(self.input_signature)
                if any(spec.dtype == tf.int32 for spec in self.input_signature.values()):
                    int64_spec = {
                        key: tf.TensorSpec(
                            shape=spec.shape, dtype=tf.int64 if spec.dtype == tf.int32 else spec.dtype, name=spec.name
                        )
                        for key, spec in self.input_signature.items()
                    }
                    int64_serving = self.serving.get_concrete_function(int64_spec)
                    signatures = {"serving_default": serving_default, "int64_serving": int64_serving}
                else:
                    signatures = serving_default
            saved_model_dir = os.path.join(save_directory, "saved_model", str(version))
            self.save(saved_model_dir, include_optimizer=False, signatures=signatures)
            logger.info(f"Saved model created in {saved_model_dir}")

        # Save configuration file
        self.config.architectures = [self.__class__.__name__[2:]]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        self.config.save_pretrained(save_directory)
        if self.can_generate():
            self.generation_config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else TF2_WEIGHTS_NAME
        output_model_file = os.path.join(save_directory, weights_name)

        shards, index = tf_shard_checkpoint(self.weights, max_shard_size, weights_name=weights_name)

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")
            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
            ):
                os.remove(full_filename)

        if index is None:
            if safe_serialization:
                state_dict = {strip_model_name_and_prefix(w.name): w.value() for w in self.weights}
                safe_save_file(state_dict, output_model_file, metadata={"format": "tf"})
            else:
                self.save_weights(output_model_file)
            logger.info(f"Model weights saved in {output_model_file}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else TF2_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, save_index_file)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as index_file:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                index_file.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )
            for shard_file, shard in shards.items():
                if safe_serialization:
                    shard_state_dict = {strip_model_name_and_prefix(w.name): w.value() for w in shard}
                    safe_save_file(
                        shard_state_dict, os.path.join(save_directory, shard_file), metadata={"format": "tf"}
                    )
                else:
                    with h5py.File(os.path.join(save_directory, shard_file), mode="w") as shard_file:
                        layers = []
                        for layer in sorted(shard, key=lambda x: x.name):
                            if "model." in layer.name or len(layer.name.split("/")) == 1:
                                layer_name = layer.name
                            else:
                                layer_name = "/".join(layer.name.split("/")[1:])
                            param_dset = shard_file.create_dataset(
                                layer_name, layer.numpy().shape, dtype=layer.numpy().dtype
                            )
                            param_dset[:] = layer.numpy()
                            layers.append(layer_name.encode("utf8"))
                        save_attributes_to_hdf5_group(shard_file, "layer_names", layers)

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        # 这些行代码从 kwargs 中移除特定的键，并将它们赋值给变量，以便后续使用。
        from_pt = kwargs.pop("from_pt", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        load_weight_prefix = kwargs.pop("load_weight_prefix", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        tf_to_pt_weight_rename = kwargs.pop("tf_to_pt_weight_rename", None)
        # Not relevant for TF models
        _ = kwargs.pop("adapter_kwargs", None)
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
        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )
        # user_agent 字典
        user_agent = {"file_type": "model", "framework": "tensorflow", "from_auto_class": from_auto_class}
        # 这段代码检查了 from_pipeline 是否有值。如果 from_pipeline 不为空（即有值），那么会在 user_agent 字典中添加一个
        # 键值对 "using_pipeline"，其值为 from_pipeline 的值。这是为了记录模型是从哪个管道（pipeline）中加载的，有助于
        # 跟踪模型的使用情况。
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline
        # 这段代码检查了是否处于离线模式 (is_offline_mode() 返回 True) 并且 local_files_only 选项没有被显式地设置
        # 为 True。在这种情况下，它会记录一条信息，并将 local_files_only 设置为 True。这意味着当系统检测到处于
        # 离线模式时，它将强制只从本地文件系统加载模型，而不尝试从网络下载。
        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True
        # 这段代码检查了 use_safetensors 是否没有被显式地设置（即默认为 None），并且 is_safetensors_available() 函数
        # 返回 False（表示 safetensors 库不可用）。如果满足这两个条件，那么它将 use_safetensors 设置为 False
        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False

        # 这段代码的主要目的是确定如何加载模型配置。具体来说，它检查传入的 config 参数是否已经是一个 
        # PretrainedConfig 对象。如果不是，它将尝试从指定的路径加载配置文件；如果是，它将直接使用传入的 
        # config 对象，并将剩余的关键字参数存储在 model_kwargs 中。
        if not isinstance(config, PretrainedConfig):
            # 如果 config 本身已经是一个路径或模型名称（例如字符串），那么就直接使用 config 作为配置文件
            # 的路径。否则，使用 pretrained_model_name_or_path 作为配置文件的路径。
            config_path = config if config is not None else pretrained_model_name_or_path
            # 调用 cls.config_class.from_pretrained 方法来加载配置文件。这个方法通常会根据提供的路径或其他标识符来
            # 加载模型的配置信息，并返回两个值：一个是配置对象 config，另一个是包含未使用的关键字参数的字典 model_kwargs。
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                _commit_hash=commit_hash,
                **kwargs,
            )
        else:
            # 如果 config 已经是一个 PretrainedConfig 对象，那么就不需要再次加载配置文件。此时，只需要将
            # 传入的所有关键字参数存储到 model_kwargs 中即可。
            model_kwargs = kwargs
        # 如果 commit_hash 没有被显式设置（即为 None），则从配置对象 config 中获取 _commit_hash 属性的值。
        # 这样做是为了记录模型的具体版本，以便在需要时可以追溯。
        if commit_hash is None: 
            commit_hash = getattr(config, "_commit_hash", None)
        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        # 这段代码的主要功能是在给定的路径中查找模型权重文件，并根据不同的条件选择正确的权重文件进行加载。此外，还会检测是否加
        # 载的是分片（sharded）的检查点。
        # 初始化一个布尔变量 is_sharded 为 False，用于标记是否加载的是分片的检查点文件。
        is_sharded = False
        # 如果 pretrained_model_name_or_path 不为 None，将其转换为字符串，并检查是否为本地目录路径。
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            # 根据条件选择权重文件
            if is_local:
               #  PyTorch 检查点：如果 from_pt 为 True 并且找到了 .bin 文件，则优先使用 PyTorch 的权重文件。
                if from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint in priority if from_pt
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                # 分片的 PyTorch 检查点：如果 from_pt 为 True 并且找到了 .index 文件，则加载分片的 PyTorch 检查
                # 点，并设置 is_sharded 为 True。
                elif from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                    is_sharded = True
                # SafeTensors 检查点：如果 use_safetensors 为 True 并且找到了 .safetensors 文件，则加载
                # SafeTensors 格式的权重文件
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                # 分片的 SafeTensors 检查点：如果 use_safetensors 为 True 并且找到了 .index.json 文件，则加载
                # 分片的 SafeTensors 检查点，并设置 is_sharded 为 True。
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
                    is_sharded = True
                # TensorFlow 2.0 检查点：如果找到了 .ckpt 文件，则加载 TensorFlow 2.0 的权重文件。
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                # 分片的 TensorFlow 2.0 检查点：如果找到了 .index 文件，则加载分片的 TensorFlow 2.0 检查点，
                # 并设置 is_sharded 为 True。
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME)):
                    # Load from a sharded TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME)
                    is_sharded = True
                # SafeTensors 模式下未找到权重文件：提示用户确认是否使用了 safe_serialization=True 或者
                # 不应设置 use_safetensors=True。
                # At this stage we don't have a weight file so we will raise an error.
                elif use_safetensors:
                    raise EnvironmentError(
                        f"Error no file named {SAFE_WEIGHTS_NAME} or {SAFE_WEIGHTS_INDEX_NAME} found in directory {pretrained_model_name_or_path}. "
                        f"Please make sure that the model has been saved with `safe_serialization=True` or do not "
                        f"set `use_safetensors=True`."
                    )
                # 存在 PyTorch 权重文件但未指定 from_pt=True：提示用户应使用 from_pt=True 来加载 PyTorch 权重文件。
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)) or os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                ):
                    raise EnvironmentError(
                        f"Error no file named {TF2_WEIGHTS_NAME} or {SAFE_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} "
                        "but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those "
                        "weights."
                    )
                # 未找到任何支持的权重文件：提示用户未找到任何支持的权重文件，并列出可能的文件名
                else:
                    raise EnvironmentError(
                        f"Error no file named {TF2_WEIGHTS_NAME}, {SAFE_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory "
                        f"{pretrained_model_name_or_path}."
                    )
            # 如果 pretrained_model_name_or_path 指向了一个存在的文件（可能是权重文件或索引文件），则将这个
            # 文件设置为 archive_file，并标记为本地路径。
            elif os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                archive_file = pretrained_model_name_or_path + ".index"
                is_local = True
            # 如果 pretrained_model_name_or_path 是一个远程URL，则下载这个URL指向的文件，并设置 
            # resolved_archive_file 为其路径。
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            
            else:
                # set correct filename
                if from_pt:
                    filename = WEIGHTS_NAME
                elif use_safetensors is not False:
                    filename = SAFE_WEIGHTS_NAME
                else:
                    filename = TF2_WEIGHTS_NAME

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "resume_download": resume_download,
                        "local_files_only": local_files_only,
                        "token": token,
                        "user_agent": user_agent,
                        "revision": revision,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_gated_repo": False,
                        "_raise_exceptions_for_missing_entries": False,
                        "_commit_hash": commit_hash,
                    }
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == SAFE_WEIGHTS_NAME:
                        # Did not find the safetensors file, let's fallback to TF.
                        # No support for sharded safetensors yet, so we'll raise an error if that's all we find.
                        filename = TF2_WEIGHTS_NAME
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **cached_file_kwargs
                        )
                    if resolved_archive_file is None and filename == TF2_WEIGHTS_NAME:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME, **cached_file_kwargs
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None and filename == WEIGHTS_NAME:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, WEIGHTS_INDEX_NAME, **cached_file_kwargs
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        # Otherwise, maybe there is a PyTorch or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "proxies": proxies,
                            "token": token,
                            "cache_dir": cache_dir,
                            "local_files_only": local_files_only,
                        }
                        if has_file(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME, **has_file_kwargs):
                            is_sharded = True
                        elif has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {TF2_WEIGHTS_NAME} but there is a file for PyTorch weights. Use `from_pt=True` to"
                                " load this model from those weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME},"
                                f" {TF2_WEIGHTS_NAME} or {TF_WEIGHTS_NAME}"
                            )

                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception:
                    # For any other exception, we throw a generic error.

                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME} or {TF_WEIGHTS_NAME}"
                    )
            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
                filename = resolved_archive_file.split(os.path.sep)[-1]
            else:
                logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                _commit_hash=commit_hash,
            )

        safetensors_from_pt = False
        if filename == SAFE_WEIGHTS_NAME:
            with safe_open(resolved_archive_file, framework="tf") as f:
                safetensors_metadata = f.metadata()
            if safetensors_metadata is None or safetensors_metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
                raise OSError(
                    f"The safetensors archive passed at {resolved_archive_file} does not contain the valid metadata."
                    " Make sure you save your model with the `save_pretrained` method."
                )
            safetensors_from_pt = safetensors_metadata.get("format") == "pt"
        elif filename == SAFE_WEIGHTS_INDEX_NAME:
            with safe_open(resolved_archive_file[0], framework="tf") as f:
                safetensors_metadata = f.metadata()
            if safetensors_metadata is None or safetensors_metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
                raise OSError(
                    f"The safetensors archive passed at {resolved_archive_file} does not contain the valid metadata."
                    " Make sure you save your model with the `save_pretrained` method."
                )
            safetensors_from_pt = safetensors_metadata.get("format") == "pt"

        config.name_or_path = pretrained_model_name_or_path

        # composed models, *e.g.* TFRag, require special treatment when it comes to loading
        # pre-trained weights.
        if cls._requires_load_weight_prefix and model_kwargs.get("name") is not None:
            model_kwargs["load_weight_prefix"] = load_weight_prefix + "/" + model_kwargs.get("name")

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if tf_to_pt_weight_rename is None and hasattr(model, "tf_to_pt_weight_rename"):
            # TODO Matt: This is a temporary workaround to allow weight renaming, but requires a method
            #            to be defined for each class that requires a rename. We can probably just have a class-level
            #            dict and a single top-level method or something and cut down a lot of boilerplate code
            tf_to_pt_weight_rename = model.tf_to_pt_weight_rename

        if from_pt:
            from .modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model

            # Load from a PyTorch checkpoint
            return load_pytorch_checkpoint_in_tf2_model(
                model,
                resolved_archive_file,
                allow_missing_keys=True,
                output_loading_info=output_loading_info,
                _prefix=load_weight_prefix,
                tf_to_pt_weight_rename=tf_to_pt_weight_rename,
            )

        # we might need to extend the variable scope for composite models
        if load_weight_prefix is not None:
            with tf.compat.v1.variable_scope(load_weight_prefix):
                model.build_in_name_scope()  # build the network with dummy inputs
        else:
            model.build_in_name_scope()  # build the network with dummy inputs

        if safetensors_from_pt and not is_sharded:
            from .modeling_tf_pytorch_utils import load_pytorch_state_dict_in_tf2_model

            with safe_open(resolved_archive_file, framework="tf") as safetensors_archive:
                # Load from a PyTorch safetensors checkpoint
                # We load in TF format here because PT weights often need to be transposed, and this is much
                # faster on GPU. Loading as numpy and transposing on CPU adds several seconds to load times.
                return load_pytorch_state_dict_in_tf2_model(
                    model,
                    safetensors_archive,
                    tf_inputs=False,  # No need to build the model again
                    allow_missing_keys=True,
                    output_loading_info=output_loading_info,
                    _prefix=load_weight_prefix,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    tf_to_pt_weight_rename=tf_to_pt_weight_rename,
                )
        elif safetensors_from_pt:
            from .modeling_tf_pytorch_utils import load_sharded_pytorch_safetensors_in_tf2_model

            return load_sharded_pytorch_safetensors_in_tf2_model(
                model,
                resolved_archive_file,
                tf_inputs=False,
                allow_missing_keys=True,
                output_loading_info=output_loading_info,
                _prefix=load_weight_prefix,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                tf_to_pt_weight_rename=tf_to_pt_weight_rename,
            )

        # 'by_name' allow us to do transfer learning by skipping/adding layers
        # see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1339-L1357
        try:
            if is_sharded:
                for file in resolved_archive_file:
                    os.path.isfile(file), f"Error retrieving files {file}"
                if filename == SAFE_WEIGHTS_INDEX_NAME:
                    missing_keys, unexpected_keys, mismatched_keys = load_tf_sharded_weights_from_safetensors(
                        model,
                        resolved_archive_file,
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                        _prefix=load_weight_prefix,
                    )
                else:
                    missing_keys, unexpected_keys, mismatched_keys = load_tf_sharded_weights(
                        model,
                        resolved_archive_file,
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                        _prefix=load_weight_prefix,
                    )
            else:
                # Handles both H5 and safetensors
                missing_keys, unexpected_keys, mismatched_keys = load_tf_weights(
                    model,
                    resolved_archive_file,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    _prefix=load_weight_prefix,
                )
        except OSError as e:
            try:
                with open(resolved_archive_file) as f:
                    if f.read().startswith("version"):
                        raise OSError(
                            "You seem to have cloned a repository without having git-lfs installed. Please install "
                            "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                            "you cloned."
                        )
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                raise OSError(
                    "Unable to load weights from h5 file. "
                    "If you tried to load a TF 2.0 model from a PyTorch checkpoint, please set from_pt=True. "
                )

        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some layers from the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.warning(f"All model checkpoint layers were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some layers of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.warning(
                f"All the layers of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
            }

            return model, loading_info

        return model

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        max_shard_size: Optional[Union[int, str]] = "10GB",
        token: Optional[Union[bool, str]] = None,
        # (`use_auth_token` is deprecated: we have to keep it here as we don't have **kwargs)
        use_auth_token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        **base_model_card_args,
    ) -> str:
       
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

        if "repo_path_or_name" in base_model_card_args:
            warnings.warn(
                "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
                "`repo_id` instead."
            )
            repo_id = base_model_card_args.pop("repo_path_or_name")
        # Deprecation warning will be sent after for repo_url and organization
        repo_url = base_model_card_args.pop("repo_url", None)
        organization = base_model_card_args.pop("organization", None)

        if os.path.isdir(repo_id):
            working_dir = repo_id
            repo_id = repo_id.split(os.path.sep)[-1]
        else:
            working_dir = repo_id.split("/")[-1]

        repo_id = self._create_repo(
            repo_id, private=private, token=token, repo_url=repo_url, organization=organization
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save_pretrained(work_dir, max_shard_size=max_shard_size)
            if hasattr(self, "history") and hasattr(self, "create_model_card"):
                # This is a Keras model and we might be able to fish out its History and make a model card out of it
                base_model_card_args = {
                    "output_dir": work_dir,
                    "model_name": Path(repo_id).name,
                }
                base_model_card_args.update(base_model_card_args)
                self.create_model_card(**base_model_card_args)

            self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
            )

    @classmethod
    def register_for_auto_class(cls, auto_class="TFAutoModel"):
       
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")
        cls._auto_class = auto_class
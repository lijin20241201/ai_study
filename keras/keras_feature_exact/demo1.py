import os
os.environ["KERAS_BACKEND"] = "tensorflow"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import copy
import tensorflow as tf
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.models import utils
from keras_cv.src.models.backbones.backbone import Backbone
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_backbone_presets import (  # noqa: E501
    backbone_presets_with_weights,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_utils import (
    CrossStagePartial,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_utils import (
    DarknetConvBlock,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_utils import (
    DarknetConvBlockDepthwise,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_utils import Focus
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_utils import (
    SpatialPyramidPoolingBottleneck,
)
from keras_cv.src.utils.python_utils import classproperty
input_data = tf.ones(shape=(1, 224, 224, 3))
import keras_cv
# 预训练骨干
model = keras_cv.models.CSPDarkNetBackbone.from_preset(
    "csp_darknet_l_imagenet",input_shape=(224,224,3)
)
len(model.layers)
[model.get_layer(i).output for i in model.pyramid_level_inputs.values()]
output = model(input_data)
model.summary()
model.get_config()
a=tf.ones((2,6,6,8))
x = Focus(name="stem_focus")(a)
x.shape
# {'name': 'csp_dark_net_backbone',
#  'trainable': True,
#  'stackwise_channels': [128, 256, 512, 1024],
#  'stackwise_depth': [3, 9, 9, 3],
#  'include_rescaling': True,
#  'use_depthwise': False,
#  'input_shape': (224, 224, 3),
#  'input_tensor': None}
# [<KerasTensor shape=(None, 56, 56, 128), dtype=float32, sparse=False, name=keras_tensor_76>,
#  <KerasTensor shape=(None, 28, 28, 256), dtype=float32, sparse=False, name=keras_tensor_211>,
#  <KerasTensor shape=(None, 14, 14, 512), dtype=float32, sparse=False, name=keras_tensor_346>,
#  <KerasTensor shape=(None, 7, 7, 1024), dtype=float32, sparse=False, name=keras_tensor_427>]
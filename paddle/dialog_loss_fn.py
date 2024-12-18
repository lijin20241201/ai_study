import paddle.nn as nn
import paddle
import paddle.nn.functional as F
class DGULossFunction(nn.Layer):
    def __init__(self, task_name):
        super(DGULossFunction, self).__init__()
        self.task_name = task_name
        self.loss_fn = self.get_loss_fn()

    def get_loss_fn(self):
        if self.task_name in [
                'udc', 'atis_slot', 'atis_intent', 'mrda', 'swda'
        ]:
            return F.cross_entropy
        elif self.task_name == 'dstc2':
            return nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, logits, labels):
        if self.task_name in ['udc', 'atis_intent', 'mrda', 'swda']:
            loss = self.loss_fn(logits, labels)
        elif self.task_name == 'dstc2':
            loss = self.loss_fn(logits, paddle.cast(labels, dtype=logits.dtype))
        elif self.task_name == 'atis_slot':
            labels = paddle.unsqueeze(labels, axis=-1)
            loss = self.loss_fn(logits, labels)
        return loss
import paddle
from paddlenlp.utils.log import logger
from paddle.metric import Accuracy
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler

def create_data_loader(dataset_class, trans_func, batchify_fn, mode):
    dataset = dataset_class(data_dir, mode)
    dataset = MapDataset(dataset).map(trans_func,lazy=True)
    if mode == 'train':
        batch_sampler = DistributedBatchSampler(dataset,
                                                batch_size=batch_size,
                                                shuffle=True)
    else:
        batch_sampler = BatchSampler(dataset,
                                     batch_size=dev_batch_size,
                                     shuffle=False)
    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=batchify_fn,
                             return_list=True)
    return data_loader

def print_logs(global_step,step,logits, labels, loss, total_time, metric):
    if task_name in ['udc', 'atis_intent', 'mrda', 'swda']:
        if task_name == 'udc':
            metric = Accuracy()
        metric.reset()
        correct = metric.compute(logits, labels)
        metric.update(correct)
        acc = metric.accumulate()
        logger.info('global_step %d -batch %d - loss: %.4f - acc: %.4f - %.3f step/s' %
              (global_step,step, loss, acc, logging_steps/total_time))
    elif task_name == 'dstc2':
        metric.reset()
        metric.update(logits, labels)
        joint_acc = metric.accumulate()
        logger.info('global_step %d -batch %d  - loss: %.4f - joint_acc: %.4f - %.3f step/s' %
              (global_step,step, loss, joint_acc, logging_steps/total_time))
    elif task_name == 'atis_slot':
        metric.reset()
        metric.update(logits, labels)
        f1_micro = metric.accumulate()
        logger.info('global_step %d -batch %d - loss: %.4f - f1_micro: %.4f - %.3f step/s' %
              (global_step,step, loss, f1_micro, logging_steps/total_time))

@paddle.no_grad()
def evaluation_DGU(model, data_loader, metric):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        if task_name in ['atis_intent', 'mrda', 'swda']:
            correct = metric.compute(logits, labels) # 准确率指标
            metric.update(correct)
        else:
            metric.update(logits, labels) # 自定义指标
    metric_out = metric.accumulate()
    model.train()
    if task_name == 'udc':
        logger.info('R1@10: %.4f - R2@10: %.4f - R5@10: %.4f\n' %
              (metric_out[0], metric_out[1], metric_out[2]))
        return metric_out[0]*0.5+metric_out[1]*0.3+metric_out[2]*0.2
    elif task_name == 'dstc2':
        logger.info('Joint_acc: %.4f\n' % metric_out)
        return metric_out
    elif task_name == 'atis_slot':
        logger.info('F1_micro: %.4f\n' % metric_out)
        return metric_out
    elif task_name in ['atis_intent', 'mrda', 'swda']:
        logger.info('Acc: %.4f\n' % metric_out)
        return metric_out
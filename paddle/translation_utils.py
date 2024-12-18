import sys
import os
import itertools
from functools import partial
import numpy as np
from paddle.io import BatchSampler, DataLoader, Dataset
import paddle.distributed as dist
from paddlenlp.data import Pad, Vocab
from paddlenlp.datasets import load_dataset
from paddlenlp.data.sampler import SamplerHelper
import copy
import importlib
import paddle.nn as nn
from paddle.jit import to_static
from paddle.static import InputSpec
import paddle


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx: # 查找结束位置
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq

def create_input_specs():
    src_word = paddle.static.InputSpec(name="src_word",
                                       shape=[None, None],
                                       dtype="int64")
    trg_word = paddle.static.InputSpec(name="trg_word",
                                       shape=[None, None],
                                       dtype="int64")
    return [src_word, trg_word]


def apply_to_static(model):
    support_to_static = config.get('to_static', False)
    if support_to_static:
        specs = create_input_specs()
        model = to_static(model, input_spec=specs)
    return model

def min_max_filer(data, max_len, min_len=3):
    # 1 for special tokens.
    data_min_len = min(len(data[0]), len(data[1])) + 1
    data_max_len = max(len(data[0]), len(data[1])) + 1
    return (data_min_len >= min_len) and (data_max_len <= max_len)

def prepare_train_input(insts,
                        bos_idx,
                        eos_idx,
                        pad_idx,
                        pad_seq=1,
                        dtype="int64"):
  
    word_pad = Pad(pad_idx, dtype=dtype)
    src_max_len = (max([len(inst[0])
                        for inst in insts]) + pad_seq) // pad_seq * pad_seq
    trg_max_len = (max([len(inst[1])
                        for inst in insts]) + pad_seq) // pad_seq * pad_seq
    # src_inputs+[eos] 形状(n,s)
    src_word = word_pad([
        inst[0] + [eos_idx] + [pad_idx] * (src_max_len - 1 - len(inst[0]))
        for inst in insts
    ])
    # trg_in +[bos] 形状(n,s)
    trg_word = word_pad([[bos_idx] + inst[1] + [pad_idx] *
                         (trg_max_len - 1 - len(inst[1])) for inst in insts])
    # trg_out +[eos] 形状(n,s,1)
    lbl_word = np.expand_dims(word_pad([
        inst[1] + [eos_idx] + [pad_idx] * (trg_max_len - 1 - len(inst[1]))
        for inst in insts
    ]),axis=2)
    
    data_inputs = [src_word, trg_word, lbl_word]

    return data_inputs

def prepare_infer_input(insts,
                        bos_idx,
                        eos_idx,
                        pad_idx,
                        pad_seq=1,
                        dtype="int64"):
    
    word_pad = Pad(pad_idx, dtype=dtype)
    src_max_len = (max([len(inst[0])
                        for inst in insts]) + pad_seq) // pad_seq * pad_seq
    src_word = word_pad([
        inst[0] + [eos_idx] + [pad_idx] * (src_max_len - 1 - len(inst[0]))
        for inst in insts
    ])

    return [
        src_word,
    ]


class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"

class SentenceBatchCreator(object):

    def __init__(self, batch_size):
        self.batch = []
        self._batch_size = batch_size

    def append(self, info):
        self.batch.append(info)
        if len(self.batch) == self._batch_size:
            tmp = self.batch
            self.batch = []
            return tmp


class TokenBatchCreator(object):

    def __init__(self, batch_size, bsz_multi=1):
        self._batch = []
        self.max_len = -1
        self._batch_size = batch_size
        self._bsz_multi = bsz_multi

    def append(self, info):
        cur_len = info.max_len
        max_len = max(self.max_len, cur_len)
        if max_len * (len(self._batch) + 1) > self._batch_size:
            # Make sure the batch size won't be empty.
            mode_len = max(
                len(self._batch) // self._bsz_multi * self._bsz_multi,
                len(self._batch) % self._bsz_multi)
            result = self._batch[:mode_len]
            self._batch = self._batch[mode_len:]
            self._batch.append(info)
            self.max_len = max([b.max_len for b in self._batch])
            return result
        else:
            self.max_len = max_len
            self._batch.append(info)

    @property
    def batch(self):
        return self._batch


class SampleInfo(object):

    def __init__(self, i, lens, pad_seq=1):
        self.i = i
        # Take bos and eos into account
        self.min_len = min(lens[0], lens[1]) + 1
        self.max_len = (max(lens[0], lens[1]) + pad_seq) // pad_seq * pad_seq
        self.seq_max_len = max(lens[0], lens[1]) + 1
        self.src_len = lens[0] + 1
        self.trg_len = lens[1] + 1


class TransformerBatchSampler(BatchSampler):

    def __init__(self,
                 dataset,
                 batch_size,
                 pool_size=10000,
                 sort_type=SortType.NONE,
                 min_length=0,
                 max_length=100,
                 shuffle=False,
                 shuffle_batch=False,
                 use_token_batch=False,
                 clip_last_batch=False,
                 distribute_mode=True,
                 seed=0,
                 world_size=1,
                 rank=0,
                 pad_seq=1,
                 bsz_multi=8):
        for arg, value in locals().items():
            if arg != "self":
                setattr(self, "_" + arg, value)
        self._random = np.random
        self._random.seed(seed)
        # for multi-devices
        self._distribute_mode = distribute_mode
        self._nranks = world_size
        self._local_rank = rank
        self._sample_infos = []
        for i, data in enumerate(self._dataset):
            lens = [len(data[0]), len(data[1])]
            self._sample_infos.append(SampleInfo(i, lens, self._pad_seq))

    def __iter__(self):
        # global sort or global shuffle
        if self._sort_type == SortType.GLOBAL:
            infos = sorted(self._sample_infos, key=lambda x: x.trg_len)
            infos = sorted(infos, key=lambda x: x.src_len)
        else:
            if self._shuffle:
                infos = self._sample_infos
                self._random.shuffle(infos)
            else:
                infos = self._sample_infos

            if self._sort_type == SortType.POOL:
                reverse = True
                for i in range(0, len(infos), self._pool_size):
                    # To avoid placing short next to long sentences
                    reverse = not reverse
                    infos[i:i + self._pool_size] = sorted(
                        infos[i:i + self._pool_size],
                        key=lambda x: x.seq_max_len,
                        reverse=reverse)

        batches = []
        batch_creator = TokenBatchCreator(
            self._batch_size,
            self._bsz_multi) if self._use_token_batch else SentenceBatchCreator(
                self._batch_size * self._nranks)

        for info in infos:
            batch = batch_creator.append(info)
            if batch is not None:
                batches.append(batch)

        if not self._clip_last_batch and len(batch_creator.batch) != 0:
            batches.append(batch_creator.batch)

        if self._shuffle_batch:
            self._random.shuffle(batches)

        if not self._use_token_batch:
            # When producing batches according to sequence number, to confirm
            # neighbor batches which would be feed and run parallel have similar
            # length (thus similar computational cost) after shuffle, we as take
            # them as a whole when shuffling and split here
            batches = [[
                batch[self._batch_size * i:self._batch_size * (i + 1)]
                for i in range(self._nranks)
            ] for batch in batches]
            batches = list(itertools.chain.from_iterable(batches))
        self.batch_number = (len(batches) + self._nranks - 1) // self._nranks

        # for multi-device
        for batch_id, batch in enumerate(batches):
            if not self._distribute_mode or (batch_id % self._nranks
                                             == self._local_rank):
                batch_indices = [info.i for info in batch]
                yield batch_indices
        if self._distribute_mode and len(batches) % self._nranks != 0:
            if self._local_rank >= len(batches) % self._nranks:
                # use previous data to pad
                yield batch_indices

    def __len__(self):
        if hasattr(self, "batch_number"):  #
            return self.batch_number
        if not self._use_token_batch:
            batch_number = (len(self._dataset) + self._batch_size * self._nranks
                            - 1) // (self._batch_size * self._nranks)
        else:
            # For uncertain batch number, the actual value is self.batch_number
            batch_number = sys.maxsize
        return batch_number

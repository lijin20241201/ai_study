import os
import paddle
from paddlenlp.utils.log import logger
import pandas as pd
from tqdm import tqdm
from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import InputExample
import json

def read_json_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            # 过滤掉空内容的
            if not line: 
                continue
            line = json.loads(line)
            yield line

def read_text_pair(data_path, is_test=False):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if not is_test:
                if len(data) != 3:
                    continue
                query1,query2,label=data
                if not query1.strip() or not query2.strip() or not label.strip():
                    continue
                yield {'query1': query1, 'query2': query2, 'label': label}
            else:
                if not data[0].strip() or not data[1].strip():
                    continue
                yield {'query1': data[0], 'query2': data[1]}

def read_pair_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word,  label= line.strip().split('\t')
            if not word.strip() or not str(label).strip():
                continue
            # print(word,label)
            yield {'text': word, 'label': label}

def get_label_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        labels= [i.strip() for i in f.readlines()]
        label2id=dict(zip(labels,range(len(labels))))
        id2label=dict(zip(label2id.values(),label2id.keys()))
        return label2id,id2label

def read_pair_by_pd(src_path):
    df=pd.read_csv(src_path, sep='\t',header=None)  
    for index, row in df.iterrows():
        query,title = row
        yield {'query': str(query), 'title': str(title)}

# 对于一些不规则的三数据组,里面有引号,引号内可能有制表符的,用pd读
def read_data_by_pd(src_path, is_predict=False):
    data = pd.read_csv(src_path, sep='\t')
    for index, row in tqdm(data.iterrows()):
        query = row['query']
        title = row['title']
        neg_title = row['neg_title']
        yield {'query': query, 'title': title, 'neg_title': neg_title}

def read_data_by_pd_test(src_path, is_predict=False):
    data = pd.read_csv(src_path, sep='\t')
    for index, row in tqdm(data.iterrows()):
        query = row['query']
        title = row['title']
        label = row['label']
        yield {'query': query, 'title': title, 'label': label}

def read_texts(data_path, is_test=False):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if not is_test:
                if len(data) != 3:
                    continue
                if len(data[0].strip()) == 0 or len(data[1].strip()) == 0 or len(data[2].strip()) == 0:
                    continue
                yield {'text_a': data[0], 'text_b': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                if len(data[0].strip()) == 0 or len(data[1].strip()) == 0:
                    continue
                yield {'text_a': data[0], 'text_b': data[1]}

def read_simcse_text(data_path):
    with open(data_path,encoding='utf-8') as f:
        for line in f:
            data=line.strip().split('\t')
            if len(data) != 2:
                continue
            query,title=data
            if not query.strip() or not title.strip():
                continue   
            yield {"query": query.strip(), "title": title.strip()}

def gen_text_file(similar_text_pair_file):
    text2similar_text = {}
    texts = []
    with open(similar_text_pair_file, 'r', encoding='utf-8') as f:
        for line in f:
            splited_line = line.rstrip().split("\t")
            if len(splited_line) != 2:
                continue
            text, similar_text =  splited_line
            if len(text.strip())==0 or len(similar_text.strip())==0:
                continue
            text2similar_text[text.strip()] = similar_text.strip()
            texts.append({"text": text.strip()})
    return texts, text2similar_text


def read_text_label(data_path):
    with open(data_path,encoding='utf-8') as f:
        for line in f:
            split_line=line.rstrip().split('\t')
            if len(split_line) !=2:
                continue
            if ' ' in split_line[0]:
                text=split_line[0].replace(' ','_')
            text=list(text)
            label=split_line[1].split(' ')
            assert len(text)==len(label), f'{text},{label}'
            yield {"text": text, "label": label}

def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels
    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]

def gen_id2corpus(corpus_file):
    id2corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            id2corpus[idx] = line.rstrip()
    return id2corpus

def read_single_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word =line.rstrip()
            if not word.strip():
                continue
            yield {'text': word}

def read_text_single(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = line.rstrip()
            if not data:
                continue
            yield {"text_a": data, "text_b": data}


import numpy as np

def convert_example_test(example, tokenizer, max_seq_length=512, pad_to_max_seq_len=False):
    result = []
    for key, text in example.items():
        encoded_inputs = tokenizer(\
            text=text,max_length=max_seq_length,truncation=True,pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]
    return result

def convert_example_mask(example, tokenizer, max_seq_length=512):
    result = []
    for key, text in example.items():
        if "label" in key:
            # do_evaluate
            result += [example["label"]]
        else:
            # do_train
            encoded_inputs = tokenizer(text=text,
                                       max_seq_len=max_seq_length,
                                       return_attention_mask=True)
            input_ids = encoded_inputs["input_ids"]
            token_type_ids = encoded_inputs["token_type_ids"]
            attention_mask = encoded_inputs["attention_mask"]
            result += [input_ids, token_type_ids, attention_mask]
    return result

def convert_example(example, tokenizer, max_seq_length=512):
    result = []
    for key, text in example.items():
        if 'label' in key:
            # do_evaluate
            result += [example['label']]
        else:
            # do_train
            encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length)
            input_ids = encoded_inputs["input_ids"]
            token_type_ids = encoded_inputs["token_type_ids"]
            result += [input_ids, token_type_ids]
    return result
# 单塔
def convert_pointwise_example(example,
                              tokenizer,
                              max_seq_length=512,
                              is_test=False):

    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(text=query,
                               text_pair=title,
                               max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array(example["label"], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

# 双塔
def convert_pairwise_example(example,
                             tokenizer,
                             max_seq_length=512,
                             phase="train"):

    if phase == "train":
        # 查询,正标题,负标题
        query, pos_title, neg_title = example["query"], example[
            "title"], example["neg_title"]
        # query+pos_title
        pos_inputs = tokenizer(text=query,
                               text_pair=pos_title,
                               max_seq_len=max_seq_length)
        # query+neg_title
        neg_inputs = tokenizer(text=query,
                               text_pair=neg_title,
                               max_seq_len=max_seq_length)
        
        pos_input_ids = pos_inputs["input_ids"]
        pos_token_type_ids = pos_inputs["token_type_ids"]
        neg_input_ids = neg_inputs["input_ids"]
        neg_token_type_ids = neg_inputs["token_type_ids"]

        return (pos_input_ids, pos_token_type_ids, neg_input_ids,
                neg_token_type_ids)

    else:
        query, title = example["query"], example["title"]

        inputs = tokenizer(text=query,
                           text_pair=title,
                           max_seq_len=max_seq_length)

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        if phase == "eval":
            return input_ids, token_type_ids, example["label"]
        elif phase == "predict":
            return input_ids, token_type_ids
        else:
            raise ValueError("not supported phase:{}".format(phase))

def convert_question_example(example, tokenizer, max_seq_length=512, is_test=False):
    query, title = example["query1"], example["query2"]
    encoded_inputs = tokenizer(text=query,
                               text_pair=title,
                               max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids
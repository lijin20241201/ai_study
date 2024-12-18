import time
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = 19530
# topK 是指你希望从所有候选向量中返回的最相似的向量的数量。这是搜索结果的一个子集，通常按照与查询向量的相似度（
# 如 L2 距离的倒数）进行排序, 指定最终返回的相似向量的数量。
top_k = 50

# IVF（倒排文件与平内积）是一种基于量化方法的索引结构，特别适用于高维向量的快速近似搜索。IVF_FLAT意味着
# IVF结构中，每个量化后的簇（centroid）内部使用平直的（flat）数据结构来存储向量，即不进行进一步的量化或编码
# metric_type指定了向量间相似度度量的类型为L2距离（欧几里得距离）。L2距离是衡量两个点在n维空间中直接距离的
# 标准，常用于高维空间中的相似度比较。
# nlist:这是IVF索引特有的一个参数，nlist指定了量化时生成的簇（centroid）的数量。在IVF_FLAT索引中，这个参数
# 直接影响搜索的准确性和性能。nlist越大，搜索时可能需要检查的簇就越多，从而可能提高搜索的准确性，但也会增加搜索时
# 间。反之，nlist越小，搜索速度更快，但可能会牺牲一些准确性。
# nlist在向量索引中通常指的是将所有索引向量分成多少份（即多少个簇或桶）。在构建向量索引时，特别是使用像IVF（Inverted
# File with Flat inner product）这样的量化索引方法时，数据集中的所有向量会被聚类成nlist个簇。每个簇由一个质心（ce
# ntroid）表示，质心是该簇内所有向量的某种形式的中心或平均。
index_config = { # 索引配置
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {
        "nlist": 1000
    },
}
# 这指定了搜索时使用的向量间相似度度量类型为L2距离。这应与索引创建时使用的度量类型相匹配。
# 这里的nprobe(探针)参数在IVF索引的搜索过程中非常重要。它指定了在搜索过程中应该检查的簇的数量。注意，这里的top_k
# 通常不是指最终返回的相似向量的数量，而是搜索过程中探索的簇的数量。nprobe的值越大，搜索到的结果可能越准确，但搜索
# 时间也会相应增加。top_k这个变量名可能会引起一些混淆，因为在实际应用中，你可能希望根据应用场景来动态设置nprobe的
# 值，而不是直接将其与最终返回的相似向量的数量（即通常所说的top_k结果）相关联。
# nprobe 控制搜索的广度和深度，即探索多少簇来找到潜在的相似向量
search_params = {
    "metric_type": "L2",
    "params": {
        "nprobe": 200
    },
}
fmt = "\n=== {:30} ===\n"

class VecToMilvus():
    def __init__(self):
        print(fmt.format("start connecting to Milvus"))
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = None

    def has_collection(self, collection_name):
        try:
            has = utility.has_collection(collection_name)
            print(f"Does collection {collection_name} exist in Milvus: {has}")
            return has
        except Exception as e:
            print("Milvus has_table error:", e)

    def creat_collection(self, collection_name,schema):
        try:
            print(fmt.format("Create collection {}".format(collection_name)))
            self.collection = Collection(collection_name,
                                         schema,
                                         consistency_level="Strong")
        except Exception as e:
            print("Milvus create collection error:", e)

    def drop_collection(self, collection_name):
        try:
            utility.drop_collection(collection_name)
        except Exception as e:
            print("Milvus delete collection error:", e)

    def create_index(self, index_name):
        try:
            print(fmt.format("Start Creating index"))
            self.collection.create_index(index_name, index_config)
            print(fmt.format("Start loading"))
            self.collection.load()
        except Exception as e:
            print("Milvus create index error:", e)

    def has_partition(self, partition_tag):
        try:
            result = self.collection.has_partition(partition_tag)
            return result
        except Exception as e:
            print("Milvus has partition error: ", e)

    def create_partition(self, partition_tag):
        try:
            self.collection.create_partition(partition_tag)
            print('create partition {} successfully'.format(partition_tag))
        except Exception as e:
            print('Milvus create partition error: ', e)

    def insert(self, entities, collection_name,schema,index_name, partition_tag=None):
        try:
            if not self.has_collection(collection_name):
                self.creat_collection(collection_name,schema)
                self.create_index(index_name)
            else:
                self.collection = Collection(collection_name)
            if (partition_tag
                    is not None) and (not self.has_partition(partition_tag)):
                self.create_partition(partition_tag)

            self.collection.insert(entities, partition_name=partition_tag)
            print(
                f"Number of entities in Milvus: {self.collection.num_entities}"
            )  # check the num_entites
        except Exception as e:
            print("Milvus insert error:", e)

class RecallByMilvus():

    def __init__(self):
        print(fmt.format("start connecting to Milvus"))
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = None

    def get_collection(self, collection_name):
        try:
            print(fmt.format("Connect collection {}".format(collection_name)))
            self.collection = Collection(collection_name)
        except Exception as e:
            print("Milvus create collection error:", e)

    def search(self,
               vectors,
               embedding_name,
               collection_name,
               partition_names=[],
               output_fields=[]):
        try:
            self.get_collection(collection_name)
            result = self.collection.search(vectors,
                                            embedding_name,
                                            search_params,
                                            limit=top_k,
                                            partition_names=partition_names,
                                            output_fields=output_fields)
            return result
        except Exception as e:
            print('Milvus recall error: ', e)

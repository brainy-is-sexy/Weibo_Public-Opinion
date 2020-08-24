# coding: utf-8

import pandas as pd
from gensim.models import FastText
from Data_preprocess.utils import load_curpus

# ### FastText词向量
# 该任务语料库较小，用fastText可以增加n-gram特征，比传统word2vec要好

# #### 对语料库进行清洗和分词
def load_data():
    data = load_curpus("../../data/emotion/train.txt") + load_curpus("../../data/emotion/test.txt")
    df = pd.DataFrame(data, columns=["content", "sentiment"])
    return df


# #### 训练词向量
def train(df):
    model = FastText(df["content"],
                 size=100,
                 window=5,
                 min_count=3, # 只保留出现次数大于3的词语
                 iter=1000,  # 10000次训练
                 min_n=2,     # 默认为3,因为文本是中文这里改为2
                 max_n=4,     # 默认为6,因为文本是中文这里改为5
                 word_ngrams=1)
    # 保存模型
    model.save("../../model/FastText_model.txt")

if __name__ == '__main__':
    train(load_data())


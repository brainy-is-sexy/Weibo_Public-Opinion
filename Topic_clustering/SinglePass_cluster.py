import numpy as np
import os
import shutil
from gensim import corpora, models, similarities, matutils
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import Data_preprocessing.Parsing_algorithms as fenci

class Single_Pass_Cluster(object):
    def __init__(self,dataname,cutname):
        self.dataname=dataname
        self.cutname=cutname

    def loadData(self,dataname,cutname):
        Data = []
        Words=[]
        with open(dataname,encoding='utf-8') as f:
            for line in f.readlines():
                Data.append(line)

        with open(cutname) as f:
            for line in f.readlines():
                dat = list(line.strip().split(' '))
                Words.append(dat)
        return Data,Words

    def get_Tfidf_vector_representation(self, word_segmentation, pivot=10, slope=0.1):
        # 得到文本数据的空间向量表示
        dictionary = corpora.Dictionary(word_segmentation)
        corpus = [dictionary.doc2bow(text) for text in word_segmentation]
        tfidf = models.TfidfModel(corpus, pivot=pivot, slope=slope)
        corpus_tfidf = tfidf[corpus]
        return corpus_tfidf

    def getMaxSimilarity(self, dictTopic, vector):
        maxValue = 0
        maxIndex = -1
        for k, cluster in dictTopic.items():
            oneSimilarity = np.mean([matutils.cossim(vector, v) for v in cluster])
            # oneSimilarity = np.mean([cosine_similarity(vector, v) for v in cluster])
            if oneSimilarity > maxValue:
                maxValue = oneSimilarity
                maxIndex = k
        return maxIndex, maxValue

    def single_pass(self, corpus, texts, theta):
        dictTopic = {}
        clusterTopic = {}
        numTopic = 0
        cnt = 0
        for vector, text in zip(corpus, texts):
            if numTopic == 0:
                dictTopic[numTopic] = []
                dictTopic[numTopic].append(vector)
                clusterTopic[numTopic] = []
                clusterTopic[numTopic].append(text)
                numTopic += 1
            else:
                maxIndex, maxValue = self.getMaxSimilarity(dictTopic, vector)
                # 将给定语句分配到现有的、最相似的主题中
                if maxValue >= theta:
                    dictTopic[maxIndex].append(vector)
                    clusterTopic[maxIndex].append(text)

                # 或者创建一个新的主题
                else:
                    dictTopic[numTopic] = []
                    dictTopic[numTopic].append(vector)
                    clusterTopic[numTopic] = []
                    clusterTopic[numTopic].append(text)
                    numTopic += 1
            cnt += 1
            if cnt % 1500 == 0:
                print("processing {}...".format(cnt))
        return dictTopic, clusterTopic

    def fit_transform(self, theta):
        datMat, word_segmentation= self.loadData(self.dataname,self.cutname)

        # 得到文本数据的空间向量表示
        corpus_tfidf = self.get_Tfidf_vector_representation(word_segmentation)
        dictTopic, clusterTopic = self.single_pass(corpus_tfidf, datMat, theta)

        #归类新话题
        filepath='../Temp/group'
        shutil.rmtree(filepath)
        os.mkdir(filepath)
        for ct in clusterTopic.keys():
            with open('../Temp/group/'+str(ct)+'.txt','w') as f:
                for i in clusterTopic[ct]:
                    f.write(i)

        print("---------"*5)
        print("发现 {} 个新话题 ".format(len(dictTopic)))
        print("---------"*5)
        # 按聚类语句数量对主题进行排序，找到重要的聚类群
        clusterTopic_list = sorted(clusterTopic.items(), key=lambda x: len(x[1]), reverse=True)

        result = open('../Temp/result.txt', 'w')
        for k in clusterTopic_list[:30]:
            cluster_title = '\n'.join(k[1])
            # print(''.join(cluster_title))
            # 得到每个聚类中的的主题关键词
            word = TextRank4Keyword()
            word.analyze(''.join(fenci.jieba_depart(''.join(cluster_title))), window=5, lower=True)
            w_list = word.get_keywords(num=10, word_min_len=2)
            sentence = TextRank4Sentence()
            sentence.analyze('\n'.join(k[1]), lower=True)
            s_list = sentence.get_key_sentences(num=3, sentence_min_len=5)[:30]
            result.write("【主题索引】:{} \n【主题语量】：{} \n【主题关键词】： {} \n【主题中心句】 ：\n{}\n\n".format(k[0], len(k[1]),','.join([i.word for i in w_list]),'\n'.join([i.sentence for i in s_list])))
            #print("【主题索引】:{} \n【主题语量】：{} \n【主题关键词】： {} \n【主题中心句】 ：\n{}".format(k[0], len(k[1]),','.join([i.word for i in w_list]),'\n\n'.join([i.sentence for i in s_list])))
        print("话题归类完成！")

if __name__ == '__main__':
    single_pass_cluster = Single_Pass_Cluster('../Temp/test.txt','../Temp/output.txt')
    single_pass_cluster.fit_transform(theta=0.005)

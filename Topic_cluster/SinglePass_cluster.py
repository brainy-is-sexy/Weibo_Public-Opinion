import jieba.analyse
import numpy as np
import os
import shutil
from gensim import corpora, models, matutils
from textrank4zh import  TextRank4Sentence

"""
Single-Pass发现新话题
"""
class Single_Pass_Cluster(object):
    def __init__(self,dataname,cutname):
        self.dataname=dataname  #原始数据
        self.cutname=cutname    #分词数据

    #加载数据
    def loadData(self,dataname,cutname):
        #获取原始微博句子
        Data = []
        with open(dataname,encoding='utf-8') as f:
            for line in f.readlines():
                Data.append(line)
        #获取分词的数据
        Words = []
        with open(cutname) as f:
            for line in f.readlines():
                dat = list(line.strip().split(' '))
                Words.append(dat)
        return Data,Words

    #VSM向量模型和Tf-idf将文本向量化
    def get_Tfidf_vector_representation(self, word_segmentation, pivot=10, slope=0.1):
        # 得到文本数据的空间向量表示
        dictionary = corpora.Dictionary(word_segmentation)
        corpus = [dictionary.doc2bow(text) for text in word_segmentation]
        tfidf = models.TfidfModel(corpus, pivot=pivot, slope=slope)
        corpus_tfidf = tfidf[corpus]
        return corpus_tfidf

    #计算文本相似度
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

    #提取关键词及摘要
    def getKeywordsAndSentence(self,clusterTopic_list):
        result = open('../result/report.txt', 'w')
        for k in clusterTopic_list[:30]:
            cluster_title = '\n'.join(k[1])
            # 设置停用词词典
            jieba.analyse.set_stop_words("../data/Stopword.txt")
            #使用Tf-idf提取关键词
            w_list_tf = []
            for x, w in jieba.analyse.extract_tags(cluster_title, topK=10, withWeight=True,
                                                   allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd')):  #
                w_list_tf.append(x)
            #使用TextRank提取关键词
            w_list_tr = []
            for x, w in jieba.analyse.textrank(cluster_title, topK=10, withWeight=True,
                                               allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd')):
                w_list_tr.append(x)
            # 得到每个聚类中的的主题关键词
            # word = TextRank4Keyword()
            # word.analyze(''.join(fenci.jieba_depart1(''.join(cluster_title))), window=5, lower=True)
            # w_list = word.get_keywords(num=10, word_min_len=2)

            #使用TextRank提取摘要
            sentence = TextRank4Sentence()
            sentence.analyze('\n'.join(k[1]), lower=True)
            s_list = sentence.get_key_sentences(num=3, sentence_min_len=5)[:30]
            #保存结果
            result.write(
                "【主题索引】:{} \n【主题语量】：{} \n【主题关键词(tf-idf)】： {} \n【主题关键词(textRank)】： {} \n【主题中心句】 ：\n{}\n\n".format(k[0],
                len(k[1]),','.join(str(i) for i in w_list_tf), ','.join( str(i) for i in w_list_tr), '\n'.join( [ i.sentence for i in s_list])))
        print("提取关键词和摘要成功！")

    #Singlepass
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

    #开始聚类
    def fit_transform(self, theta):
        #加载数据
        datMat, word_segmentation= self.loadData(self.dataname,self.cutname)

        # 得到文本数据的空间向量表示
        corpus_tfidf = self.get_Tfidf_vector_representation(word_segmentation)
        dictTopic, clusterTopic = self.single_pass(corpus_tfidf, datMat, theta)
        print("发现 {} 个新话题 ".format(len(dictTopic)))

        #新话题归类
        filepath='../result/group/'
        shutil.rmtree(filepath)
        os.mkdir(filepath)
        for ct in clusterTopic.keys():
            with open(filepath+str(ct)+'.txt','w') as f:
                for i in clusterTopic[ct]:
                    f.write(i)
        print("话题归类完成！")

        # 按聚类语句数量对主题进行排序,找到语句数量最多的主题
        clusterTopic_list = sorted(clusterTopic.items(), key=lambda x: len(x[1]), reverse=True)
        #提取关键词及摘要
        self.getKeywordsAndSentence(clusterTopic_list)


if __name__ == '__main__':
    single_pass_cluster = Single_Pass_Cluster('../Temp/data.txt','../Temp/output.txt')
    single_pass_cluster.fit_transform(theta=0.025)

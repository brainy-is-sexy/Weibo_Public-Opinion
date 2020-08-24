import jieba
import jieba.posseg

"""
中文分词处理、词性标注、去停分词
"""
# 定义词性列表
# post_list = ['n','v','vd','vn','l','d','s','nr','ns','nt','nw','nz','a','ad','an','PER','LOC','ORG','TIME']

class Cut_words():
    # 创建停用词列表
    def stopwordslist(self):
        stopwords = [line.strip() for line in open('..\data\Stopword.txt', encoding='UTF-8').readlines()]
        return stopwords

    #jieba分词
    def jieba_depart(self,sentence,stopwords):
        # 对文档中的每一行进行中文分词
        words = jieba.cut(sentence.strip())

        # 将输出结果保存到outstr
        outstr = []
        # 去停用词
        for word in words:
            if  word == ' ':
                continue
            if word not in stopwords:
                outstr.append(word)
        return outstr

    def start(self,sentence):
        stopwords=self.stopwordslist()
        self.jieba_depart(sentence,stopwords)

# 加载数据
def load_data(inputfile, outputfile):
    # 打开文件
    inputs = open(inputfile, 'r', encoding='utf-8')#
    outputs = open(outputfile, 'w')
    return inputs, outputs

def Start():
    # 加载数据
    inputs, outputs = load_data("../Temp/data.txt", "../Temp/output.txt")
    print("中文分词成功！")
    print("--------------" * 5)

if __name__ == '__main__':
    Start()




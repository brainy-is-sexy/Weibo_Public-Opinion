import jieba
import pynlpir

# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('D:\AudioRecognition\数据\Stopword.txt',encoding='UTF-8').readlines()]
    return stopwords

# 结巴分词
def jieba_depart(sentence,stopwords):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if (word not in stopwords and word !=' '):
            outstr += word
            outstr += " "
    return outstr

#中科院分词
def nlpir_depart(sentence,stopwords):
    # 输出结果为outstr
    outstr = ''
    # 对文档中的每一行进行中文分词
    pynlpir.open()
    split_content = pynlpir.segment(sentence,pos_tagging = False)
    #split_content = pynlpir.segment(sentence,pos_english=False)

    # 去停用词
    for word in split_content:
        if (word not in stopwords and word != ' '):
            outstr += str(word)+' '
    pynlpir.close()
    return outstr

if __name__ == "__main__":
    # 给出待分词的文档路径
    filename = "D:\Desktop\\11.txt"
    #保存分词后文档路径
    outfilename = "D:\Desktop\out1.txt"

    inputs = open(filename, 'r',encoding='utf-8')
    outputs = open(outfilename, 'w')

    # 创建一个停用词列表
    stopwords = stopwordslist()

    # 将输出结果写入out.txt中
    for line in inputs:
        #使用结巴分词
        #line_seg = jieba_depart(line,stopwords)
        #使用中科院分词
        line_seg = nlpir_depart(line, stopwords)
        outputs.write(line_seg + '\n')

    outputs.close()
    inputs.close()
    print("删除停用词和分词成功")
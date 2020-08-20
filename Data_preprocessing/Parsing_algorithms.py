import jieba
import pynlpir
import jieba.analyse
import jieba.posseg


# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('..\Stopword.txt', encoding='UTF-8').readlines()]
    return stopwords


# 结巴分词
def jieba_depart(sentence):
    # 创建一个停用词列表
    stopwords = stopwordslist()

    # 定义词性列表
    # post_list = ['n', 'nr', 'nt', 'f', 'ns', 'nz', 'j', 'i', 'a', 'l', 'z', 'an']
    post_list = ['c', 'o', 'p', 'r', 'u', 'w', 'y', 'm', 'un', 'x', 'v', 'q','eng']
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.posseg.cut(sentence.strip())

    # 筛选符合词性的词语
    dict_data = {}
    for w in sentence_depart:
        dict_data[w.word] = w.flag
    table = {k: v for k, v in dict_data.items() if v not in post_list}
    words = list(table.keys())
    #print(table)

    # 输出结果为outstr
    outstr = []
    # 去停用词
    for word in words:
        if (word not in stopwords and word != ' '):
            outstr.append(word)
    return outstr


# 中科院分词
def nlpir_depart(sentence, stopwords):
    # 输出结果为outstr
    outstr = ''
    # 对文档中的每一行进行中文分词
    pynlpir.open()
    split_content = pynlpir.segment(sentence, pos_tagging=False)
    # split_content = pynlpir.segment(sentence,pos_english=False)

    # 去停用词
    for word in split_content:
        if (word not in stopwords and word != ' '):
            outstr += str(word) + ' '
    pynlpir.close()
    return outstr


# 加载数据
def load_data(inputfile, outputfile):
    # 打开文件
    inputs = open(inputfile, 'r', encoding='utf-8')
    outputs = open(outputfile, 'w')

    return inputs, outputs


def Start():
    # 加载数据
    inputs, outputs = load_data("../Temp/test.txt", "../Temp/output.txt")

    # 将输出结果写入
    for line in inputs:
        # 使用结巴分词
        line_seg = jieba_depart(line)
        outputs.write(' '.join(line_seg) + '\n')

    print("中文分词成功！")
    print("--------------" * 5)

if __name__ == '__main__':
    Start()




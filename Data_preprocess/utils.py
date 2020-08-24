import jieba
import re


# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('..\data\Stopword.txt', encoding='UTF-8').readlines()]
    return stopwords

def tokenize(text,stopwords):
    """
    带有语料清洗功能的分词函数, 包含数据预处理, 可以根据自己的需求重载
    """
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用户名)
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    icons = re.findall("\[.+?\]", text)             # 提取出所有表情图标
    text = re.sub("\[.+?\]", "IconMark", text)      # 将文本中的图标替换为`IconMark`

    tokens = []
    for k, w in enumerate(jieba.lcut(text)):
        w = w.strip()
        if "IconMark" in w:                         # 将IconMark替换为原图标
            for i in range(w.count("IconMark")):
                tokens.append(icons.pop(0))
        elif w and w != '\u200b' and w.isalpha() and w not in stopwords:   # 只保留有效文本
                tokens.append(w)
    return tokens

#训练使用
def load_curpus(path):
    """
    加载语料库
    """
    stopwords=stopwordslist()
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = tokenize(content,stopwords)             # 分词
            data.append((content, int(seniment)))
    return data

#Sentimentanlysis
def load_curpus1(path):
    """
    加载语料库
    """
    stopwords=stopwordslist()
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [ seniment, content] = line.split(",", 1)
            content = tokenize(content,stopwords)             # 分词
            data.append((content, int(seniment)))
    return data

#情感词典
def load_curpus2(line):
    """
    加载语料库
    """
    stopwords = stopwordslist()

    [seniment, content] = line.split(",", 1)
    content = tokenize(content,stopwords) #分词
    return content


if __name__ == '__main__':
    load_curpus("../data/emotion/train.txt")

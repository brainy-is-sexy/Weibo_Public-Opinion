from snownlp import SnowNLP
from snownlp import sentiment


#获取情感分数
def getScore(source):
    line = source.readlines()
    sentimentslist = []
    #得到每一句话的情感分
    for i in line:
        s = SnowNLP(i)
        sentimentslist.append(s.sentiments)

    # 判断极性
    pos=0
    neg=0
    for i in sentimentslist:
        if i<0.5:
            neg+=1
        else:
            pos+=1
    return pos,neg

#训练新模型
def train():
    # 重新训练模型
    sentiment.train('./neg.txt', './pos.txt')
    # 保存好新训练的模型
    sentiment.save('sentiment.marshal')

#区间转换为[-0.5, 0.5]
def getExchange(sentimentslist):
    result = []
    i = 0
    while i<len(sentimentslist):
        result.append(sentimentslist[i]-0.5)
        i = i + 1
    return result

if __name__ == '__main__':
    source = open("D:/Desktop/庆余年.txt", "r", encoding='utf-8')#


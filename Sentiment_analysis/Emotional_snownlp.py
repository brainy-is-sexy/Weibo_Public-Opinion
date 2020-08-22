from snownlp import SnowNLP

#获取情感分数
def getScore(source):
    line = source.readlines()
    sentimentslist = []
    for i in line:
        s = SnowNLP(i)
        #print(s.sentiments)
        sentimentslist.append(s.sentiments)
    return  sentimentslist


#区间转换为[-0.5, 0.5]
def getExchange(sentimentslist):
    result = []
    i = 0
    while i<len(sentimentslist):
        result.append(sentimentslist[i]-0.5)
        i = i + 1
    po=0
    ne=0
    for j in result:
        if j>0: po+=1
        else: ne+=1
    print(str(po)+"  "+str(ne)+" "+str(len(result)))
    po=float(po/len(result))
    ne=float(ne/len(result))
    return po,ne

if __name__ == '__main__':
    source = open("../Temp/test.txt", "r", encoding='utf-8')
    po,ne=getExchange(getScore(source))
    print(str(po)+"  "+str(ne))
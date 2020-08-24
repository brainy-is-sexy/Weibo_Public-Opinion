import pandas as pd
from Data_preprocess.utils import load_curpus2

class DUTIR:
    def __init__(self):
        self.Positive, self.Negative, self.Happy, self.Good, \
        self.Surprise, self.Sad, self.Fear, self.Disgust, self.Anger=self.getMood(self.getDICT())

    #情感词典读取
    def getDICT(self):
        df = pd.read_excel('../data/大连理工大学中文情感词汇本体NAU.xlsx')
        df = df[['词语', '词性种类', '词义数', '词义序号', '情感分类', '强度', '极性']]
        return df

    #获取七种情绪的运用
    def getMood(self,df):
        Happy = []
        Good = []
        Surprise = []
        Anger = []
        Sad = []
        Fear = []
        Disgust = []

        # df.iterrows()功能是迭代遍历每一行
        for idx, row in df.iterrows():
            if row['情感分类'] in ['PA', 'PE']:
                Happy.append(row['词语'])
            if row['情感分类'] in ['PD', 'PH', 'PG', 'PB', 'PK']:
                Good.append(row['词语'])
            if row['情感分类'] in ['PC']:
                Surprise.append(row['词语'])
            if row['情感分类'] in ['NB', 'NJ', 'NH', 'PF']:
                Sad.append(row['词语'])
            if row['情感分类'] in ['NI', 'NC', 'NG']:
                Fear.append(row['词语'])
            if row['情感分类'] in ['NE', 'ND', 'NN', 'NK', 'NL']:
                Disgust.append(row['词语'])
            if row['情感分类'] in ['NAU']:  # 修改: 原NA算出来没结果
                Anger.append(row['词语'])

        # 正负计算
        Positive = Happy + Good + Surprise
        Negative = Anger + Sad + Fear + Disgust
        print('情绪词语列表整理完成')
        return Positive,Negative,Happy,Good,Surprise,Sad,Fear,Disgust,Anger

    # 情感计算
    def emotion_caculate(self,wordlist):
        #情感词统计
        positive = 0
        negative = 0
        anger = 0
        disgust = 0
        fear = 0
        sad = 0
        surprise = 0
        good = 0
        happy = 0

        for word in wordlist:
            if word in self.Positive:
                positive += 1
            if word in self.Negative:
                negative += 1
            if word in self.Anger:
                anger += 1
            if word in self.Disgust:
                disgust += 1
            if word in self.Fear:
                fear += 1
            if word in self.Sad:
                sad += 1
            if word in self.Surprise:
                surprise += 1
            if word in self.Good:
                good += 1
            if word in self.Happy:
                happy += 1

        emotion_info = {
            'length': len(wordlist),
            'positive': positive,
            'negative': negative,
            'anger': anger,
            'disgust': disgust,
            'fear': fear,
            'good': good,
            'sadness': sad,
            'surprise': surprise,
            'happy': happy,
        }
        return emotion_info

    def start(self,filepath,output):
        happy = 0
        good = 0
        surprise = 0
        anger = 0
        sad = 0
        fear = 0
        disgust = 0
        #打开分词完成的文本
        with open(filepath,'r',encoding='utf-8') as f:
            for line in f:
                #传入一行的分词列表
                data = load_curpus2(line)
                result=self.emotion_caculate(data)
                happy=happy+result['happy']
                good = good + result['good']
                surprise = surprise + result['surprise']
                anger = anger + result['anger']
                sad = sad + result['sadness']
                fear = fear + result['fear']
                disgust = disgust + result['disgust']

        with open(output, 'w') as f:
            f.write("开心："+str(happy)+"\n")
            f.write("称赞：" + str(good) + "\n")
            f.write("惊讶：" + str(surprise) + "\n")
            f.write("愤怒：" + str(anger) + "\n")
            f.write("伤心：" + str(sad) + "\n")
            f.write("害怕：" + str(fear)+ "\n")
            f.write("厌恶：" +str( disgust) + "\n")


if __name__ == '__main__':
    dutir=DUTIR()
    dutir.start("D:\毕设\代码\Weibo_Public-Opinion\data\\topics\\毕业.txt","D:/Desktop/report.txt")


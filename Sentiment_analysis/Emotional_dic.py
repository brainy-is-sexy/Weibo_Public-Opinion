import pandas as pd
import time
import Data_preprocessing.Parsing_algorithms as fenci
#-------------------------------------情感词典读取-------------------------------
#注意：
#1.词典中怒的标记(NA)识别不出被当作空值,情感分类列中的NA都给替换成NAU
#2.大连理工词典中有情感分类的辅助标注(有NA),故把情感分类列改好再替换原词典中
def getDICT():
    df = pd.read_excel('../data/大连理工大学中文情感词汇本体NAU.xlsx')
    #print(df.head(10))
    df = df[['词语', '词性种类', '词义数', '词义序号', '情感分类', '强度', '极性']]
    #df.head()
    return df
# -------------------------------------七种情绪的运用-------------------------------
def getMood(df):
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

    # 正负计算不是很准 自己可以制定规则
    Positive = Happy + Good + Surprise
    Negative = Anger + Sad + Fear + Disgust
    #print('情绪词语列表整理完成')
    return Positive,Negative,Happy,Good,Surprise,Sad,Fear,Disgust,Anger


# ---------------------------------------情感计算---------------------------------
def emotion_caculate(wordlist):
    Positive, Negative, Happy, Good, Surprise, Sad, Fear, Disgust, Anger = getMood(getDICT())

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
        if word in Positive:
            positive += 1
        if word in Negative:
            negative += 1
        if word in Anger:
            anger += 1
        if word in Disgust:
            disgust += 1
        if word in Fear:
            fear += 1
        if word in Sad:
            sad += 1
        if word in Surprise:
            surprise += 1
        if word in Good:
            good += 1
        if word in Happy:
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

    #indexs = ['length', 'positive', 'negative', 'anger', 'disgust', 'fear', 'sadness', 'surprise', 'good', 'happy']
    # return pd.Series(emotion_info, index=indexs), anger_list, disgust_list, fear_list, sad_list, surprise_list, good_list, happy_list
    return emotion_info

if __name__ == '__main__':
    f = open('../Temp/output.txt','r')
    file = open("../Temp/emotion.txt",'w')
    for line in f:
        re=emotion_caculate(list(line.strip().split(' ')))
        file.write(str(re)+"\n")


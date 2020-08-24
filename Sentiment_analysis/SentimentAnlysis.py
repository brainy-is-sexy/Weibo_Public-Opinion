# coding: utf-8

import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq
from Data_preprocess.utils import load_curpus1,load_curpus
from gensim.models import FastText


class Sentiment():
    def __init__(self,path):
        #向量长度
        self.max_length = 128
        # 加载之前训练好的FastText模型
        self.model=FastText.load("../model/FastText_model.txt")
        #待分析数据
        self.path=path

    #加载数据
    def load_data(self):
        data = {}
        for f in glob.glob(self.path):
            topic = os.path.split(f)[-1].split(".")[0]
            data[topic] = load_curpus(f)
        return data

    #调整向量长度
    def vector_length(self,data):
        data_X, data_length = {}, {}
        for topic, corpus in data.items():
            _data_X, _data_length = [], []
            for content, sentiment in corpus:
                X = []
                for w in content[:self.max_length]:
                    if w in self.model:
                        X.append(np.expand_dims(self.model[w], 0))
                if X:
                    length = len(X)
                    X = X + [np.zeros_like(X[0])] * (self.max_length - length)
                    X = np.concatenate(X)
                    X = np.expand_dims(X, 0)
                    _data_X.append(X)
                    _data_length.append(length)
            data_X[topic] = _data_X
            data_length[topic] = _data_length
        return data_X,data_length


    def train(self):
        data = self.load_data()
        data_X, data_length=self.vector_length(data)
        # 文本数
        batch_size = 100
        lr = 1e-3
        hidden_size = 100

        X = tf.placeholder(shape=(batch_size, self.max_length, 100), dtype=tf.float32, name="X")
        L = tf.placeholder(shape=(batch_size), dtype=np.int32, name="L")
        y = tf.placeholder(shape=(batch_size, 1), dtype=np.float32, name="y")
        dropout = tf.placeholder(shape=(), dtype=np.float32, name="dropout")
        with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
            def lstm_cell(hidden_size, cell_id=0):
                # LSTM细胞生成器
                cell = rnn.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name='cell%d' % cell_id)
                cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout)
                return cell

            context = tf.get_variable("context", shape=(1, hidden_size))
            context = tf.tile(context, [batch_size, 1])
            fw_cell = lstm_cell(hidden_size, 0)
            bw_cell = lstm_cell(hidden_size, 1)
            fw_zero = fw_cell.zero_state(batch_size, tf.float32)
            bw_zero = fw_cell.zero_state(batch_size, tf.float32)
            encoder_output, encoder_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                             cell_bw=bw_cell,
                                                                             inputs=X,
                                                                             sequence_length=L,
                                                                             initial_state_fw=fw_zero,
                                                                             initial_state_bw=bw_zero,
                                                                             dtype=tf.float32)
            attention_context = tf.concat(encoder_output, axis=2)
            attention_mech = seq2seq.BahdanauAttention(hidden_size * 2,
                                                       memory=attention_context,
                                                       memory_sequence_length=L,
                                                       name="AttentionMechanism")
            attention_cell = seq2seq.AttentionWrapper(cell=lstm_cell(hidden_size, 2),
                                                      attention_mechanism=attention_mech,
                                                      attention_layer_size=hidden_size,
                                                      alignment_history=True,
                                                      output_attention=True,
                                                      name="AttentionCell")
            attention_zero = attention_cell.zero_state(batch_size, tf.float32)
            attention_output, attention_state = attention_cell.call(context, attention_zero)
            aligments = attention_state[3]

            W1 = tf.get_variable("W1", shape=(hidden_size, 50))
            b1 = tf.get_variable("b1", shape=(50,))
            W2 = tf.get_variable("W2", shape=(50, 1))
            b2 = tf.get_variable("b2", shape=(1,))
            fcn1 = tf.nn.xw_plus_b(attention_output, W1, b1)
            logists = tf.nn.xw_plus_b(fcn1, W2, b2)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logists, labels=y))
            op = tf.train.AdamOptimizer(lr).minimize(loss)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.Session(config=config)

        #加载模型
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkPoint = tf.train.get_checkpoint_state("../model/attention")
        saver.restore(sess, checkPoint.model_checkpoint_path)

        # 对不同主题的文本进行情感分类
        sentiment = {}
        prediction = []
        for topic in data_X.keys():
            _X = np.concatenate(data_X[topic] + [np.zeros_like(data_X[topic][0])] * (batch_size - len(data_X[topic])))
            _L = np.array(data_length[topic] + [1] * (batch_size - len(data_length[topic])))
            result = sess.run(tf.nn.sigmoid(logists), feed_dict={X: _X, L: _L, dropout: 1.})
            for i in result[:len(data_X[topic])]:
                if i > 0.5:
                    prediction.append(1)
                else:
                    prediction.append(0)
            sentiment[topic] = prediction

        for topic, res in sentiment.items():
            print("主题为【%s】的微博中, 正面:%d, 负面:%d" % (topic, res.count(1), res.count(0)))
        return prediction

if __name__ == '__main__':
    path="D:\毕设\代码\Weibo_Public-Opinion\data\\topics\\毕业.txt"
    se=Sentiment(path)
    prediction=se.train()
    file=open("D:/Desktop/test.txt",'w')
    for p in prediction:
        if(p>0.5):
            file.write("1\n")
        else:
            file.write("0\n")



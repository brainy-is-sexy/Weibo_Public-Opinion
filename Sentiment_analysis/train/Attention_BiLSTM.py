# coding: utf-8

import numpy as np
import tensorflow as tf
from sklearn import metrics
from Data_preprocess.utils import load_curpus
from gensim.models import FastText
from tensorflow.contrib import rnn, seq2seq

class BiLstm():
    def __init__(self):
        self.max_length = 128
        # 加载之前训练好的FastText模型
        self.model=FastText.load("../../model/FastText_model.txt")

    #加载数据
    def load_data(self):
        train_data = load_curpus("../../data/emotion/train.txt")
        test_data = load_curpus("../../data/emotion/test.txt")
        return train_data,test_data

    # 为保证输入神经网络的向量长度一致, 要对长度不足max_length的句子用零向量补齐, 对长度超过max_length的句子进行截断
    def vector_length(self,train_data,test_data):
        X_train, train_length, y_train = [], [], []
        for content, sentiment in train_data:
            X, y = [], sentiment
            for w in content[:self.max_length]:
                if w in self.model:
                    X.append(np.expand_dims(self.model[w], 0))
            if X:
                length = len(X)
                X = X + [np.zeros_like(X[0])] * (self.max_length - length)
                X = np.concatenate(X)
                X = np.expand_dims(X, 0)
                X_train.append(X)
                train_length.append(length)
                y_train.append(y)

        X_test, test_length, y_test = [], [], []
        for content, sentiment in test_data:
            X, y = [], sentiment
            for w in content[:self.max_length]:
                if w in self.model:
                    X.append(np.expand_dims(self.model[w], 0))
            if X:
                length = len(X)
                X = X + [np.zeros_like(X[0])] * (self.max_length - length)
                X = np.concatenate(X)
                X = np.expand_dims(X, 0)
                X_test.append(X)
                test_length.append(length)
                y_test.append(y)
        return X_train,train_length,X_test,test_length,y_train,y_test

    #Attention+BiLSTM
    # 由于tensorflow的Attention是基于Seq2Seq结构的，我这里也采用了这种结构，不过并没有使用“解码器”
    def train(self,X_train,train_length,X_test,test_length,y_train,y_test):
        batch_size = 512
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

            # BiLSTM部分
            fw_cell = lstm_cell(hidden_size, 0)
            bw_cell = lstm_cell(hidden_size, 1)
            fw_zero = fw_cell.zero_state(batch_size, tf.float32)
            bw_zero = fw_cell.zero_state(batch_size, tf.float32)

            # Seq2Seq版的dynamic_rnn
            encoder_output, encoder_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                     cell_bw=bw_cell,
                                                                     inputs=X,
                                                                     sequence_length=L,
                                                                     initial_state_fw=fw_zero,
                                                                     initial_state_bw=bw_zero,
                                                                     dtype=tf.float32)

            # Attention模块
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

            # Attention加权得到的context向量
            attention_zero = attention_cell.zero_state(batch_size, tf.float32)
            attention_output, attention_state = attention_cell.call(context, attention_zero)
            aligments = attention_state[3]

            # 用context向量直接用MLP做二分类
            W1 = tf.get_variable("W1", shape=(hidden_size, 50))
            b1 = tf.get_variable("b1", shape=(50,))
            W2 = tf.get_variable("W2", shape=(50, 1))
            b2 = tf.get_variable("b2", shape=(1,))
            fcn1 = tf.nn.xw_plus_b(attention_output, W1, b1)
            logists = tf.nn.xw_plus_b(fcn1, W2, b2)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logists, labels=y))  # 交叉熵
            op = tf.train.AdamOptimizer(lr).minimize(loss)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.Session(config=config)

        total_step = 1001
        step = 0
        cursor = 0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        while step < total_step:
            _X, _L, _y = X_train[cursor: cursor + batch_size], train_length[cursor: cursor + batch_size], y_train[
                                                                                    cursor: cursor + batch_size]
            cursor += batch_size
            if len(_X) < batch_size:
                cursor = batch_size - len(_X)
                _X += X_train[: cursor]
                _L += train_length[: cursor]
                _y += y_train[: cursor]
            _X = np.concatenate(_X)
            _L = np.reshape(np.array(_L, dtype=np.int32), (-1))
            _y = np.reshape(np.array(_y, dtype=np.float32), (batch_size, 1))
            _, l = sess.run([op, loss], feed_dict={X: _X, L: _L, y: _y, dropout: .75})
            if step % 100 == 0:
                print("step:", step, " loss:", l)
                saver.save(sess, '../../model/attention/model', global_step=step)
            step += 1

        _X = np.concatenate(X_test + [np.zeros_like(X_test[0])] * (batch_size - len(X_test)))
        _L = np.array(test_length + [1] * (batch_size - len(test_length)))

        result = sess.run(tf.nn.sigmoid(logists), feed_dict={X: _X, L: _L, dropout: 1.})
        prediction = []
        for i in result[:len(X_test)]:
            if i > 0.5:
                prediction.append(1)
            else:
                prediction.append(0)

        #效果测评
        # 相比纯LSTM提升没那么明显，主要是因为该任务相对简单，语料少。迁移至更复杂任务后注意力的作用会越来越明显
        print(metrics.classification_report(y_test, prediction))
        print("准确率:", metrics.accuracy_score(y_test, prediction))

if __name__ == '__main__':
    #创建对象
    lstm= BiLstm()
    train_data,test_data=lstm.load_data()
    X_train,train_length,X_test,test_length,y_train,y_test=lstm.vector_length(train_data,test_data)
    lstm.train(X_train,train_length,X_test,test_length,y_train,y_test)



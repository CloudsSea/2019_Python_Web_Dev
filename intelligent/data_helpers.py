import re

import tensorflow as tf
import numpy as np
from os import path,listdir


from string import punctuation
from string import digits
import jieba
from gensim.models import Word2Vec

rule = re.compile(u'[^a-zA-Z.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'+digits+punctuation+'\u4e00-\u9fa5]+')

epoch_read = 500
lstm_num = 128
classes_num = 8
words_vector_length = 256
batch_size = 20
lstm_out_size = 96
vocal_size = 906


def get_train_data():
    model = Word2Vec.load('./model/word2vec/wv.model')
    words_list = model.wv.vocab
    print(words_list)
    x_batches = []
    y_batches_classify = []
    y_batches_predict = []
    lines = []
    lables = []
    with open('./data/temp_result_cut.txt', 'r', encoding='utf-8') as f:
        lines_all = f.readlines()
        for line in lines_all:
            words = line.split(" ")
            lines.append(words)
    # print(lines)
    with open('./data/label_copy.txt', 'r', encoding='utf-8') as f:
        lables = f.readlines()
    # print(lables)
    lables = [float(label) for label in lables]

    line_label_pair = zip(lines,lables)

    lines_label_sorted = sorted(line_label_pair, key=lambda l: len(l[0]))
    # print(lines_label_sorted)
    print('=============================>')
    lines_sorted,labels_sorted = zip(*lines_label_sorted)


    lines_vector = []
    for  line in lines_sorted:
        temp = [words_list[word].index for word in line if word != '\n' and word != '']
        lines_vector.append(temp)
    # TODO 下面的语句得到的是map,不知道是什么原因
    #lines_vector = [list(map(lambda word: words_list[word].index, line) for line in lines_sorted)]

    epoch = len(lines_sorted) // batch_size
    for i in range(epoch):
        start_index = batch_size * i
        end_index = start_index + batch_size
        x_temp = lines_vector[start_index:end_index]
        print(x_temp)
        y_batch_classify = labels_sorted[start_index:end_index]
        length = max(map(len, x_temp))
        x_batch = np.full(shape=[batch_size, length], fill_value=906, dtype=np.int)
        for i in range(batch_size):
            x_batch[i,:len(x_temp[i])] = x_temp[i]

        y_batch_predict = np.copy(x_batch)
        y_batch_predict[:,:-1] = x_batch[:,1:]

        x_batches.append(x_batch)
        y_batches_classify.append(y_batch_classify)
        y_batches_predict.append(y_batch_predict)

    print("epoch: %s \n" % epoch)
    print("x_batches: %s \n" % x_batches)
    print("y_batches_classify: %s \n" % y_batches_classify)
    print("y_batches_predict: %s \n" % y_batches_predict)
    return epoch,x_batches,y_batches_classify,y_batches_predict



def clean_sentence(line):
    new_content,old_content = line.split('//',1)
    line = re.sub(rule, '', line)
    return line;

def read_data():
    # TODO 1.两个#中间是话题; 可以存起来; 然后过滤掉2. @后面是人名,用他来代替? ;  3.很多引用的,去重; 4. 去掉http链接
    # TODO 5.//之前是引用, 从前往后检索, 6.转发微博(过滤)  6.[表情]  表情是很重员的情感分析的部分,怎么处理; 为表情创建一个字典?还是只去掉中括号(决定用逗号代替括号
    # TODO 6. 嘉伦的刘海的微博视频 7. 啊啊啊啊啊啊啊啊 8. ヽ(‘⌒´メ)ノ 8 . （分享自 @音悦台 ）
    # TODO 9 查看图片  10  回复@XXX:  11.  ... 全文 12.微博饭票 13 位置: 句子最后, 城市·地点
    # TODO 13 做一个统计, 空格之间常见短语过滤  14. 《xxx》书名 15. 过滤英文
    temp_file = './data/temp.txt'
    print('hello')
    with open(temp_file, 'r', encoding='utf-8') as f:
        pass

        line = f.readline()
        print(line)
        new_line = clean_sentence(line)
        print(new_line)

def demo3():
    lines = []
    file_dir = 'D:/Yun/Yun2019/thesis'
    file_name  = 'weibo_2019-05-20_01.59.48.txt'
    lines = []

    # ids = np.load('./')
    # words_vector = np.load('./')


    # 根据维基百科训练词向量
    file_list = listdir(file_dir)
    weibo_file = path.join(file_dir,file_name)
    print(file_list)

    if True:
        with open(weibo_file,'r',encoding='utf-8') as f:
            pass
            for i in range(epoch_read):
                line = f.readline()
                # print(line)
                # line = clean_line(line)
                if not line:
                    continue
                print(line)
                lines.append(line)
        with open('./data/temp.txt', 'w',encoding='utf-8') as f:
            for line in lines:
                f.write(line)

    pass



def model_predict(output_datas):
    ids = np.load('./idsMatrix.npy')
    # 文字清洗
    # 排序(长短) 处理多长的语句, 语句的统计? 最长的 3/4的长度
    #
    # (数据洗牌)求批的数组; (存数据库?)每个批次统一长度
    # ? 问题, 每次训练长度不同是否可以?
    x_ph = tf.placeholder(dtype=np.int32,shape=[batch_size,None])
    y_ph = tf.placeholder(dtype=np.int32,shape=[batch_size,None])
    x_lookup = tf.nn.embedding_lookup(ids,x_ph)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_num)
    if output_datas is None:
        lstm_cell.zero_state(1,tf.float64)
    else:
        lstm_cell.zero_state(batch_size,tf.float64)


    #tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=0.5)
    outputs,status = tf.nn.dynamic_rnn(lstm_cell,x_lookup,dtype=tf.float64)
    # batch lstm_length  out
    x_input = tf.reshape(outputs,[-1,lstm_num])

    w1 = tf.Variable(initial_value=tf.truncated_normal(shape=[lstm_num,vocal_size+1],dtype=tf.float64))
    b1 = tf.Variable(initial_value=tf.constant(0.1,dtype=tf.float64))

    y_predict = tf.matmul(x_input,w1) + b1
    # batch classes_num
    y_true = tf.one_hot(tf.reshape(y_ph, [-1]),depth=vocal_size+1)
    loss_all = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict)
    loss = tf.reduce_mean(loss_all)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    equals_list = tf.equal(tf.argmax(y_predict,1),tf.argmax(y_true,1))
    accurcy = tf.reduce_mean(tf.cast(equals_list,dtype=tf.float64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        epoch, x_batches, y_batches_classify, y_batches_predict = get_train_data()
        for i in range(epoch):
            for truns in range(epoch_read):
                x_batch = x_batches[i]
                y_batch = y_batches_predict[i]
                loss_new,accurcy_new,_ = sess.run([loss,accurcy,optimizer],feed_dict={
                    x_ph : x_batch,
                    y_ph : y_batch
                })
                print('loss: %f;accucury: %f' % (loss_new,accurcy_new))
                if i % 10 == 0:
                    saver.save(sess,save_path='./model/classes/commentclassify')




#
def model_classify(output):
    ids = np.load('./idsMatrix.npy')
    # 文字清洗
    # 排序(长短) 处理多长的语句, 语句的统计? 最长的 3/4的长度
    #
    # (数据洗牌)求批的数组; (存数据库?)每个批次统一长度
    # ? 问题, 每次训练长度不同是否可以?
    x_ph = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
    y_ph = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    x_lookup = tf.nn.embedding_lookup(ids, x_ph)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_num)
    # tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=0.5)
    outputs, status = tf.nn.dynamic_rnn(lstm_cell, x_lookup, dtype=tf.float64)
    # batch lstm_length  out
    value = tf.transpose(outputs, [1, 0, 2])
    print(value.get_shape())
    print(value.get_shape()[0])
    # x_input = tf.gather(value, int(value.get_shape()[0]) - 1)
    x_input = value[-1,:,:]
    print(x_input)
    w1 = tf.Variable(initial_value=tf.truncated_normal(shape=[lstm_num, classes_num],dtype=tf.float64))
    b1 = tf.Variable(initial_value=tf.constant(0.1, dtype=tf.float64))

    y_predict = tf.matmul(x_input, w1) + b1
    # batch classes_num
    y_true = tf.one_hot(y_ph, depth=classes_num)
    loss_all = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
    loss = tf.reduce_mean(loss_all)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    equals_list = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true, 1))
    accurcy = tf.reduce_mean(tf.cast(equals_list, dtype=tf.float64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        epoch, x_batches, y_batches_classify, y_batches_predict = get_train_data()
        for truns in range(epoch_read):
            for i in range(epoch):
                x_batch = x_batches[i]
                y_batch = y_batches_classify[i]
                loss_new, accurcy_new,_ = sess.run([loss, accurcy,optimizer], feed_dict={
                    x_ph: x_batch,
                    y_ph: y_batch
                })
                print('loss: %f;accucury: %f' % (loss_new, accurcy_new))
                if i % 10 == 0:
                    saver.save(sess, save_path='./model/predict/commentpredict')

    # 1.词向量模型的训练,并存储 --> 转成 TensorFlow中常用的格式 word_list, vector
    # 2.1 模型1: 取100条记录, 存入一个文件,另外一个文件夹标记每一句的 情感分析
    # 2.2 模型2: 结合素材, 训练预测上下文的模型, 两重终止符;   训练的时候,每个微博作为一个整体;
        # 预测的时候,遇到句号就终止;


if __name__ == '__main__':
    # demo3()
    # read_data()
    # get_train_data("")
    model_classify(" ")
    # model_predict()
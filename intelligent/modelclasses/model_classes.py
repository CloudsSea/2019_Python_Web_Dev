#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json
from gensim.models import Word2Vec
from tensorflow.keras.layers import (Bidirectional,
                                     Embedding,
                                     GRU,
                                     GlobalAveragePooling1D,
                                     GlobalMaxPooling1D,
                                     Concatenate,
                                     SpatialDropout1D,
                                     BatchNormalization,
                                     Dropout,
                                     Dense,
                                     Activation,
                                     concatenate,
                                     Input,
                                     Reshape,
                                     LSTM
                                    )
from tensorflow.keras.utils import to_categorical
import time
import re
import jieba
from gensim.models import Word2Vec
# In[2]:

def cut_words(sentence):
    #print sentence
    return " ".join(jieba.cut(sentence))


def build_model(sent_length, embeddings_weight, class_num):
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="sentence_cuted",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(GRU(200, return_sequences=True))(x)
    #     x = Bidirectional(GRU(200, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)  # 全军平均池化
    max_pool = GlobalMaxPooling1D()(x)  # 全局最大池化

    conc = concatenate([avg_pool, max_pool])  # 特征放到一起，

    x = Dense(1000)(conc)  # 全连接层
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(500)(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    output = Dense(class_num, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def preprocess(inputs,labels):
    #最简单的预处理函数:	转numpy为Tensor、分类问题需要处理label为one_hot编码、处理训练数据
    #把numpy数据转为Tensor
    labels = tf.cast(labels, dtype=tf.int32)
    #labels 转为one_hot编码
    labels = tf.one_hot(labels, depth=embedding_matrix.shape[0])
    return inputs,labels

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


DIR = os.path.dirname(__file__)


MAX_SEQUENCE_LENGTH = 100
maxlen = 100
checkpoint_dir = os.path.join( DIR,'checkpoints_classes')


json_str= json.load(open(os.path.join( DIR,'tokenizer_config.json'), 'r'))
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_str)
word_vocab = tokenizer.word_index
embedding_matrix = np.load(os.path.join( DIR,'idsMatrix.npy'))


model = build_model(maxlen, embedding_matrix,3)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

def do_classes(text):
    text = cut_words(text)
    text_array = []
    text_array.append(text)
    # text_array.append(text)
    # print(text_array)
    a = pd.DataFrame(text_array)
    a = a.rename(columns={0:'sentence'})
    # p_col=['省份','id','编码']
    # province.columns=p_col
    print(a)
    sequence = tokenizer.texts_to_sequences(a['sentence'].values)
    print(sequence)

    sentences = []

    input = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=maxlen,
                                                  padding='pre', truncating='pre', value=0.0)
    print(input)
    print("原文\n"+text)

    preds = model.predict(input, verbose=0)
    print(preds)

    result =  np.argmax(preds,axis=1) - 1
    if result == -1:
        print('伤心')
    elif result == 0:
        print('客观')
    else:
        print('开心')

if __name__ == '__main__':
    print(len(word_vocab))
    print(embedding_matrix.shape[0])
    print(embedding_matrix.shape[1])
    do_classes("好好享受快乐")








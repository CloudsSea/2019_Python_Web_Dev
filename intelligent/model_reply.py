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
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import time
import re
import jieba
from gensim.models import Word2Vec
# In[2]:

def cut_words(sentence):
    #print sentence
    return " ".join(jieba.cut(sentence))

def build_model(sent_length, embeddings_weight):
    content = Input(shape=(sent_length,), dtype='int32')

    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)
    x = embedding(content)

    x = LSTM(128)(x)
    x = Dense(embedding_matrix.shape[0])(x)
    output = Activation(activation="softmax")(x)

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




MAX_SEQUENCE_LENGTH = 100
maxlen = 40
checkpoint_dir = './checkpoints_reply'

file_name = './word2vec/Word2Vec_word_200.model'
print('loading')
model_word2vec = Word2Vec.load(file_name)
print("add word2vec finished....")
json_str= json.load(open('tokenizer_config.json', 'r'))
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_str)
word_vocab = tokenizer.word_index
embedding_matrix = np.load('./idsMatrix.npy')
word_vocal_reverse = {}
word_vocal_reverse[0] = 'SPACE'

for i,word in word_vocab.items():
    word_vocal_reverse[word] = i

model = build_model(maxlen, embedding_matrix)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

def generate_reply(text):
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
    #     for diversity in [0.2, 0.5, 1.0, 1.2]:
    replys = []
    for diversity in [0.2, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        print('----- Generating with seed:')
        for i in range(20):#400
            # x_pred = np.zeros((1, maxlen))
            # for t, char in enumerate(sentence):
            #     #print(t,char)
            #     #x_pred[0, t, char_indices[char]] = 1.
            #     x_pred[0, t] = char

            preds = model.predict(input, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = word_vocal_reverse[next_index]
            generated += next_char
            sequence = sequence[1:]
            sequence.append(next_index)
        print(generated)
        replys.append(generated)
    return replys

if __name__ == '__main__':
    print(len(word_vocal_reverse))
    print(embedding_matrix.shape[0])
    generate_reply("武汉加油")








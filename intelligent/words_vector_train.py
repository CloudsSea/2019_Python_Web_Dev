from os import path,listdir
import numpy as np
import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs,sys

import logging,os
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re

import tensorflow as tf

rule = re.compile(u'@')
rule1 = re.compile(u'，，')

def clean_sentences(line):
    line = re.sub(rule, '', line)
    line = re.sub(rule1, '，', line)
    return line

def cut_words(sentence):
    #print sentence
    return " ".join(jieba.cut(sentence))


def read_txt_cut():
    with codecs.open('./data/temp_result_clean.txt','r',encoding="utf8") as f:
    # with codecs.open('./data/in_the_name_of_people.txt','r',encoding="utf8") as f:
        line = f.readline()
        line_num = 1
        curr = []
        while line:
            line_num +=1

            curr.append(line)
            next_line = f.readline()
            next_line = clean_sentences(next_line)
            if line_num % 16 == 0 or not next_line:
                after_cut = map(cut_words, curr)
                with codecs.open("./data/temp_result_cut.txt", 'w',encoding="utf8") as target:
                    target.writelines(after_cut)
                    print('saved', line_num, 'articles')

            line = next_line

def train_vector():
    # program = os.path.basename(sys.argv[0])
    # logger = logging.getLogger(program)
    # logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    # logging.root.setLevel(level=logging.INFO)
    # logger.info("running %s" % ' '.join(sys.argv))
    # # check and process input arguments
    # if len(sys.argv) < 4:
    #     print(globals()['__doc__'] % locals())
    #     sys.exit(1)

    inp, outp1, outp2 = ('./data/temp_result_cut.txt','./model/word2vec/wv.model','./model/word2vec/wv.vector')
    model = Word2Vec(LineSentence(inp), size=100, window=5, min_count=1, workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

def clean_demo():
    with codecs.open('./data/temp_result_clean.txt','r',encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            print(clean_sentences(line))


def theVectors():
    x = tf.placeholder(dtype=tf.float32,shape=[2,100])


def save_ids():
    model = Word2Vec.load('./model/word2vec/wv.model')
    vector_dim = 100
    print(model)
    embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    np.save('idsMatrix', embedding_matrix)


def similar():
    model = Word2Vec.load('./model/word2vec/wv.model')
    vector_dim = 100
    print(model)
    embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix)
    print(len(model.wv.vocab))

    testwords = ['鲜花', '好看','']
    words_list_np = np.zeros((2,))
    for i in range(2):
        words_list_np[i] = model.wv.vocab[testwords[i]].index
        print(model.wv.vocab[testwords[i]].index)
        # print(model.wv[testwords[i]])

    # x = tf.placeholder(dtype=tf.int32,shape=[None])
    # x_lookup = tf.nn.embedding_lookup(embedding_matrix,x)
    #
    # with tf.Session() as sess:
    #
    #     embedding_new= sess.run([x_lookup],feed_dict={x:words_list_np})
    #     print(embedding_new)


if __name__ == '__main__':
    # clean_demo()
    # read_txt_cut()
    # train_vector()
    similar()
    # save_ids()

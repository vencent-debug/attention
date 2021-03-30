import gensim
from gensim.models import KeyedVectors
import os
import util
import pandas as pd
import numpy as np
import scipy as sp
import torch
import random
from scipy.sparse import coo_matrix
# 导入数据：分隔符为空格


def data_idx(number_walks):
    number_walks = number_walks
    raw_data = pd.read_csv('cora/cora.content', sep='\t', header=None)
    num = raw_data.shape[0]  # 样本点数2708
    # 将论文的编号转[0,2707]
    a = list(raw_data.index)
    b = list(raw_data[0])
    c = zip(b, a)
    map = dict(c)
    # 将词向量提取为特征,第二行到倒数第二行
    features = raw_data.iloc[:, 1:-1]
    # 检查特征：共1433个特征，2708个样本点
    print(features.shape)
    labels = pd.get_dummies(raw_data[1434])
    raw_data_cites = pd.read_csv('cora/cora.cites', sep='\t', header=None)
    undirected = True
    # 创建一个规模和邻接矩阵一样大小的矩阵
    matrix = np.zeros((num, num))
    # 创建邻接矩阵
    for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
        x = map[i];
        y = map[j]  # 替换论文编号为[0,2707]
        matrix[x][y] = matrix[y][x] = 1  # 有引用关系的样本点之间取1
    # 查看邻接矩阵的元素和（按每列汇总）
    # print(sorted((sum(matrix))))

    x = np.array(matrix)

    cx = coo_matrix(x)  # 转换为coo matrix形式
    #print(cx)
    G = util.Graph()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G[i].append(j)
    if undirected:
        G.make_undirected()
    G.make_consistent()
    print(G)
    num_walks = len(G.nodes()) * number_walks  # 总共产生的游走序列的数量
    return  num_walks,G



def embeddingWord():
    embeddings_file = "cora.embeddings"
    matfile = cx
    adj_matrix_name = "network"
    label_matrix_name = "group"
    num_shuffles = 10



def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])
    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    # 归一化, 用位置嵌入的每一行除以它的模长
    #denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
    #position_enc = position_enc / (denominator + 1e-8)
    return positional_encoding


#这个函数的目的生成每个句子的wordEM+PE
def makeSquToembed(model,walk, walk_length,embed_dim,word_vector_dict):
    all_squence_vectors = []
    simple_positional_encoding = get_positional_encoding(max_seq_len=walk_length, embed_dim=embed_dim)


    for i in range(int(len(walk)/100)):
        word_vectors = []

        for j in range(walk_length):

            word_vectors = list(model[walk[i][j]] for word in word_vector_dict)
            word_vectors.append(word_vectors+simple_positional_encoding[j])

        all_squence_vectors.append(word_vectors)

    return  all_squence_vectors

#并将获得的词及其对应的词向量按字典的格式写入word_vector_dict
def wordandWE(model):
    word_vector_dict ={}
    for word in model.wv.index2word:
        word_vector_dict[word] = list(model[word])
    print(word_vector_dict['0'])
    return  word_vector_dict



def producevacabulary(walks):
    # print(walks)
    # print('pause')
    with open('vacabulary_train.txt','w') as f:
        for i in range(int(len(walks)*0.8)):
            for j in range(len(walks[0])):
                f.write(walks[i][j]+",")
            f.write('\n')
    with open('vacabulary_test.txt','w') as f:
        for i in range(int(len(walks)*0.8),len(walks)):
            for j in range(len(walks[0])):
                f.write(walks[i][j]+",")
            f.write('\n')
    print("写好了")

def main():
    seed = 0
    number_walks = 1  # 在全图上执行随机游走的次数(每次都会对图中的所有点进行随机游走)
    walk_length = 40  # 随机游走的长度
    num_walks,G = data_idx(number_walks)  # 总共产生的游走序列的数量

    data_size = num_walks * walk_length
    representation_size = 512  # 词向量的维度
    window_size = 10  # word2vec训练时窗口的大小
    workers = 4  # 并行进程数




    walks = util.build_deepwalk_corpus(G,
                              num_paths=number_walks,
                              path_length=walk_length,
                              alpha=0,
                              rand=random.Random(seed))
    producevacabulary(walks)


    # model = gensim.models.Word2Vec(walks,
    #                  size=representation_size,
    #                  window=window_size,
    #                  min_count=0, sg=1, hs=1,
    #                  workers=workers)
#    model.wv.save_word2vec_format("cora_embeddings")
    embeddings_file = 'cora_embeddings'
    # 加载词向量
    model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
    word_vector_dict = wordandWE(model)
    # print(model)
    # embeddings = np.array(model[i] for i in range(2708))
    # print(type(embeddings))
    #
    all_squence_vectors = makeSquToembed(model,
                                         walks,
                                         walk_length,
                                         representation_size,
                                         word_vector_dict)
    print("end")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


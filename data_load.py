from construct_vacabularu import Hyperparams as hp

import numpy as np

import pandas as pd

def load_allpath_vocab():
    # 英文
    vocab = [line.split()[0] for line in  open('vacabulary.txt','r').read().splitlines()]
    print(len(vocab))


    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word



def load_src_data():
    raw_data = pd.read_csv('cora/cora.content', sep='\t', header=None)
    num = raw_data.shape[0]  # 样本点数2708
    print(num)
    # 将论文的编号转[0,2707]
    a = list(raw_data.index)
    b = list(raw_data[0])
    c = zip(b, a)
    map = dict(c)
    print(map)
    return map


def create_data(source_sents, target_sents): #source_sents存放源语言句子的列表, target_sents目标语言句子
    de2idx= idx2de = load_src_data()
    en2idx= idx2en = load_src_data()
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):#使用zip()函数同时遍历两个句子列表
        # x,y 一个新句子
        # 对句子进行编码de2idx.get(word, 1) for word in
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()]  # 1: OOV, </S>: End of Text
        # dict.get(key, default=None)
        #
        # # 举例
        #
        # word2idx.get(word, 1)
        # for word in sentence
        #
        # # 意思是 如果word 没在词典中，返回默认值 1

        # 给每一个句子的末尾加上终止符，并遍历句子中的每一个单词，将已经存在于word2idx中的那个单词对应的ID添加到新的列表中，
        # 如果这个单词不存在于word2idx中，
        # 那么就返回 ID‘1’到新列表中组成一个新的‘ID句子’(其中1代表)

        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]

        if max(len(x), len(y)) <= hp.maxlen:  # 句子中单词的最大数限制
            x_list.append(np.array(x))  # 源语言ID句子
            y_list.append(np.array(y))  # 目标语言ID句子
            Sources.append(source_sent)  # 源语言句子
            Targets.append(target_sent)  # 目标语言句子
            # 超过长度阈值的丢弃

    # Pad 填充 对应site特殊词中的编号0：
    X = np.zeros([len(x_list), hp.maxlen], np.int32)  #二维0矩阵：句子个数*最大句长
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)

    for i, (x, y) in enumerate(zip(x_list, y_list)):  # (x, y) = （源句， 目标句）
        # 对每一个ID句子做填充，左侧填充0个0，右侧填充hp.maxlen-len(x)个0，并且0也是四个特殊词中的一个：编号0
        X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))  # 填充
        Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))
    return X, Y, Sources, Targets
#X和Y的shape为[len(x_list), hp.maxlen]，Sources, Targets的shape为[1, len(x_list)]

# 加载训练集，对训练集做数据处理，返回定长ID句子
def load_train_data():
    de_sents_train = [line.split()[0] for line in  open('vacabulary_test.txt','r').read().splitlines()]
    en_sents_test =[line.split()[0] for line in  open('vacabulary_test.txt','r').read().splitlines()]
    X, Y, Sources, Targets = create_data(de_sents_train, en_sents_test)
    return X, Y

# 加载测试集，对测试集做数据处理，返回定长ID句子

def load_test_data():
    de_sents_test = [line.split()[0] for line in open('vacabulary_train.txt', 'r').read().splitlines()]
    en_sents_test = [line.split()[0] for line in open('vacabulary_train.txt', 'r').read().splitlines()]
    return en_sents_test, en_sents_test


def main():
    word2idx, idx2word = load_allpath_vocab()
    map = load_src_data()
    X, Y =  load_train_data()
    print(X)



if __name__ == '__main__':
    main()
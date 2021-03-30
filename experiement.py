import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd



def load_src_data():
    raw_data = pd.read_csv('cora/cora.content', sep='\t', header=None)
    num = raw_data.shape[0]  # 样本点数2708
#    print(num)
    # 将论文的编号转[0,2707]
    a = list(raw_data.index)
    b = list(raw_data[0])
    c = zip(b, a)
    map = dict(c)
    # print(map)
    return map



#
# sentences  =[line.split()[0] for line in  open('vacabulary_test.txt','r').read().splitlines()]
# src_vocab =load_src_data()
# src_vocab_size = len(src_vocab)
#
# tgt_vocab =load_src_data()
#
# idx2word = {i: w for i, w in enumerate(tgt_vocab)}
# tgt_vocab_size = len(tgt_vocab)
#
# src_len = 40  # enc_input max sequence length
# tgt_len = 40  # dec_input(=dec_output) max sequence length

def make_data(sentences):


    return torch.Tensor(sentences), torch.Tensor(sentences), torch.Tensor(sentences)



def main():


    sentences  =[line.split()[0] for line in  open('vacabulary_test.txt','r').read().splitlines()]
    print(type(sentences))


    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

if __name__ == '__main__':
    main()
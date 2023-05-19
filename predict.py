import torch
import numpy as np
from importlib import import_module
import argparse
import os
import pickle as pkl
import torch.nn.functional as F
import pdb
import time
import re

parser = argparse.ArgumentParser(description="Classification based Transformer")
parser.add_argument("--model", type=str, default="TextCNN")
parser.add_argument("--dataset", type=str, default="THUCNews")
# parser.add_argument("--text",type=str )
parser.add_argument('--use_word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
args = parser.parse_args()

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
dataset_name = args.dataset
# ThuNews
if dataset_name == "THUCNews":
    key = {
        0: '住宿服务',
        1: '餐饮服务',
        2: '道路旅客运输服务',
        3: '基础电信服务',
        4: '汽油',
        5: '交通运输服务'
    }

model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
x = import_module('models.' + model_name)

# 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
embedding = args.embedding
if model_name == 'FastText':
    from utils_fasttext import build_dataset, build_iterator, get_time_dif

    embedding = 'random'
config = x.Config(dataset_name, embedding)
if os.path.exists(config.vocab_path):
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    config.n_vocab = len(vocab)

model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location=torch.device('cuda')))  #
model.eval()


def biGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets


def triGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    t2 = sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets


def build_predict_text(text, use_word):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    token = tokenizer(text)
    seq_len = len(token)
    pad_size = config.pad_size
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size

    words_line = []
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))

    if args.model == "FastText":
        buckets = config.n_gram_vocab
        bigram = []
        trigram = []
        # ------ngram------
        for i in range(pad_size):
            bigram.append(biGramHash(words_line, i, buckets))
            trigram.append(triGramHash(words_line, i, buckets))

        ids = torch.LongTensor([words_line]).to(config.device)
        seq_len = torch.LongTensor([seq_len]).to(config.device)
        bigram_ts = torch.LongTensor([bigram]).to(config.device)
        trigram_ts = torch.LongTensor([trigram]).to(config.device)

        return ids, seq_len, bigram_ts, trigram_ts
    else:

        # ids = torch.LongTensor([words_line]).cuda()
        ids = torch.LongTensor([words_line]).to(config.device)
        seq_len = torch.LongTensor(seq_len).to(config.device)
        return ids, seq_len


def predict(text):
    data = build_predict_text(text, args.use_word)
    with torch.no_grad():
        outputs = model(data)
        num = torch.argmax(outputs)
        pred = F.softmax(outputs, dim=0)
        score = pred.view(-1)[num]
        score = score.cpu().numpy()
        print(score)
        num = num.cpu().numpy()
        if score < 0.51:
            num = -1
    return key[int(num)]
    # return num


# if __name__ == "__main__":
#     while 1:
#         x = input('请输入：')
#         if x == 'q':
#             exit(-1)
#         _start = time.time()
#         keywords = []
#         y = predict(x)
#         _end = time.time()
#         if y != -1:
#             print('预测：', str(y))

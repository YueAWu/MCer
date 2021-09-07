"""
Create by Juwei Yue on 2019-11-2
Data and function
"""

import json
import math
import os
import pickle
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module, Parameter, init

import numpy as np
from sklearn import preprocessing


class Data:
    """
    Load data
    """
    def __init__(self, data, one=True, direct=True):
        self.adj_matrix = data[0]
        if one:
            self.adj_matrix = self.trans_to_one(self.adj_matrix, data[2])
        if not direct:
            self.adj_matrix = self.trans_to_undirected(self.adj_matrix)
        self.event_chain = data[1]
        self.label = data[2]
        self.len = len(self.label)
        self.start = 0
        self.flag_epoch = True

    @staticmethod
    def trans_to_one(matrix, label):
        matrix = torch.where(matrix > 0, torch.ones_like(matrix), matrix)
        for i in range(matrix.size(0)):
            for j in range(7):
                if matrix[i][j][j+1] == 0:
                    matrix[i][j][j+1] = 1
            if matrix[i][7][8+label[i]] == 0:
                matrix[i][7][8+label[i]] = 1
        return matrix

    @staticmethod
    def trans_to_undirected(matrix):
        return torch.add(matrix, matrix.permute(0, 2, 1))

    def next_batch(self, batch_size):
        start = self.start
        end = self.start + batch_size if self.start + batch_size < self.len else self.len
        self.start = self.start + batch_size
        if self.start < self.len:
            self.flag_epoch = True
        else:
            self.start = self.start % self.len
            self.flag_epoch = False
        return [to_cuda(self.event_chain[start: end]),
                to_cuda(self.adj_matrix[start: end]),
                to_cuda(self.label[start: end])]

    def all_data(self, index=None):
        if index is None:
            return [to_cuda(self.event_chain), to_cuda(self.adj_matrix), to_cuda(self.label)]
        else:
            return [to_cuda(self.event_chain.index_select(0, index)),
                    to_cuda(self.adj_matrix.index_select(0, index)),
                    to_cuda(self.label.index_select(0, index))]


def to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def get_word_embedding():
    try:
        return np.load("data/metadata/word_embedding.npy")
    except FileNotFoundError:
        with open("data/metadata/deepwalk_128_unweighted_with_args.txt") as f:
            index_embedding = {}
            for line in f:
                line = line.strip().split()
                if len(line) == 2:
                    continue
                index_embedding[line[0]] = np.array(line[1:], dtype=np.float32)
            index_embedding["0"] = np.zeros(len(index_embedding["0"]), dtype=np.float32)
        word_embedding = []
        for i in range(len(index_embedding)):
            word_embedding.append(index_embedding[str(i)])
        word_embedding = np.array(word_embedding, dtype=np.float32)
        np.save("data/metadata/word_embedding", word_embedding)
        return word_embedding


def best_result(model, model_state, best_val_epoch, best_val_acc, test_acc,
                val_acc_history, test_result, hyper_parameter):
    with open("data/result/best_result.txt", encoding="utf-8") as f:
        result = [line for line in f]
        result = [result[i:i + 3] for i in range(0, len(result), 3)]

    new_result = []
    best_history = [model + ":\n",
                    "\tEpoch %d, Best Val Acc: %f, Test Acc: %f\n" % (best_val_epoch, best_val_acc, test_acc),
                    "\t%s\n" % hyper_parameter]

    init_history = 0
    for r in result:
        if model != r[0].strip().split(":")[0]:
            new_result.append(r)
            init_history += 1
            continue
        best_test_acc = float(r[1].split(",")[-1].split(":")[1].strip())
        if test_acc > best_test_acc:
            pickle.dump(val_acc_history, open("data/result/model/" + model + "_acc_history.pickle", "wb"), 2)
            pickle.dump(test_result, open("data/result/model/" + model + "_test_result.pickle", "wb"), 2)
            remove_model_file("data/result/model/", "^" + model + "_model")
            torch.save(model_state, "data/result/model/" + model + "_model_acc_%s.model" % test_acc)
            new_result.append(best_history)
        else:
            new_result.append(r)
    if init_history == len(result):
        pickle.dump(val_acc_history, open("data/result/model/" + model + "_acc_history.pickle", "wb"), 2)
        pickle.dump(test_result, open("data/result/model/" + model + "_test_result.pickle", "wb"), 2)
        remove_model_file("data/result/model/", "^" + model + "_model")
        torch.save(model_state, "data/result/model/" + model + "_model_acc_%s.model" % test_acc)
        new_result.append(best_history)

    with open("data/result/best_result.txt", "w", encoding="utf-8") as f:
        for nr in new_result:
            for r in nr:
                f.write(r)


def remove_model_file(path, file_pattern):
    for _, _, files in os.walk(path):
        for file in files:
            if re.match(file_pattern, file):
                os.remove(os.path.join(path, file))


def match_model_file(path, file_pattern):
    for _, _, files in os.walk(path):
        for file in files:
            if re.match(file_pattern, file):
                return os.path.join(path, file)
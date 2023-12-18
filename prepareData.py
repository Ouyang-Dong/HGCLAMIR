import csv
import os
import torch as t
import numpy as np
from math import e
import pandas as pd
from scipy import io


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)


def read_mat(path, name):
    matrix = io.loadmat(path)
    matrix = t.FloatTensor(matrix[name])
    return matrix


def read_md_data(path, validation):
    result = [{} for _ in range(validation)]
    for filename in os.listdir(path):
        data_type = filename[filename.index('_')+1:filename.index('.')-1]
        num = int(filename[filename.index('.')-1])
        result[num-1][data_type] = read_csv(os.path.join(path, filename))
    return result


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def Gauss_M(adj_matrix, N):
    GM = np.zeros((N, N))
    rm = N * 1. / sum(sum(adj_matrix * adj_matrix))
    for i in range(N):
        for j in range(N):
            GM[i][j] = e ** (-rm * (np.dot(adj_matrix[i, :] - adj_matrix[j, :], adj_matrix[i, :] - adj_matrix[j, :])))
    return GM


def Gauss_D(adj_matrix, M):
    GD = np.zeros((M, M))
    T = adj_matrix.transpose()
    rd = M * 1. / sum(sum(T * T))
    for i in range(M):
        for j in range(M):
            GD[i][j] = e ** (-rd * (np.dot(T[i] - T[j], T[i] - T[j])))
    return GD


def prepare_data(opt):
    dataset = {}

    dd_data = pd.read_csv(opt.data_path + '/dis_sem_sim_2.0.csv',index_col=0)
    dd_mat = np.array(dd_data)

    mm_data = pd.read_csv(opt.data_path + '/mi_fun_sim_2.0.csv',index_col=0)
    mm_mat = np.array(mm_data)

    mi_dis_data = pd.read_csv(opt.data_path + '/mi_dis_mat_2.0.csv',index_col=0)

    dataset['md_p'] = t.FloatTensor(np.array(mi_dis_data))
    dataset['md_true'] = dataset['md_p']

    all_zero_index = []
    all_one_index = []
    for i in range(dataset['md_p'].size(0)):
        for j in range(dataset['md_p'].size(1)):
            if dataset['md_p'][i][j] < 1:
                all_zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                all_one_index.append([i, j])


    np.random.seed(0)
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)


    zero_tensor = t.LongTensor(all_zero_index)
    zero_index = zero_tensor.split(int(zero_tensor.size(0) / 10), dim=0)
    one_tensor = t.LongTensor(all_one_index)
    one_index = one_tensor.split(int(one_tensor.size(0) / 10), dim=0)

    
    cross_zero_index = t.cat([zero_index[i] for i in range(9)])
    cross_one_index = t.cat([one_index[j] for j in range(9)])
    new_zero_index = cross_zero_index.split(int(cross_zero_index.size(0) / opt.validation), dim=0)
    new_one_index = cross_one_index.split(int(cross_one_index.size(0) / opt.validation), dim=0)
    dataset['md'] = []
    for i in range(opt.validation):
        a = [i for i in range(opt.validation)]
        if opt.validation != 1:
            del a[i]
        dataset['md'].append({'test': [new_one_index[i], new_zero_index[i]],
                              'train': [t.cat([new_one_index[j] for j in a]), t.cat([new_zero_index[j] for j in a])]})

    
    dataset['independent'] = []
    in_zero_index_test = zero_index[-2]
    in_one_index_test = one_index[-2]
    dataset['independent'].append({'test': [in_one_index_test, in_zero_index_test],
                                   'train': [cross_one_index,cross_zero_index]})


    DGSM = Gauss_D(dataset['md_p'].numpy(), dataset['md_p'].size(1))
    MGSM = Gauss_M(dataset['md_p'].numpy(), dataset['md_p'].size(0))


    nd = mi_dis_data.shape[1]
    nm = mi_dis_data.shape[0]

    ID = np.zeros([nd, nd])

    for h1 in range(nd):
        for h2 in range(nd):
            if dd_mat[h1, h2] == 0:
                ID[h1, h2] = DGSM[h1, h2]
            else:
                ID[h1, h2] = (dd_mat[h1, h2] + DGSM[h1, h2]) / 2

    IM = np.zeros([nm, nm])

    for q1 in range(nm):
        for q2 in range(nm):
            if mm_mat[q1, q2] == 0:
                IM[q1, q2] = MGSM[q1, q2]
            else:
                IM[q1, q2] = (mm_mat[q1, q2] + MGSM[q1, q2]) / 2

    dataset['ID'] = t.from_numpy(ID)
    dataset['IM'] = t.from_numpy(IM)

    return dataset

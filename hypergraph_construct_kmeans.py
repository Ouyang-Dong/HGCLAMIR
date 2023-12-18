import numpy as np
import torch
from sklearn.cluster import KMeans


def hyperedge_concat(*H_list):

    H = None
    for h in H_list:
        # if h is not None and h != []:
        if h is not None and len(h) != 0:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def _construct_edge_list_from_cluster(X, clusters):
    """
    construct edge list (numpy array) from cluster for single modality
    :param X: feature
    :param clusters: number of clusters for k-means
    :param adjacent_clusters: a node's adjacent clusters
    :param k_neighbors: number of a node's neighbors
    :return:
    """
    N = X.shape[0]
    kmeans = KMeans(n_clusters=clusters, init = 'k-means++', random_state=0).fit(X)

    assignment = kmeans.labels_

    H = np.zeros([N, clusters])
   
    for i in range(N):
        H[i, assignment[i]] = 1

    return H

def construct_H_with_Kmeans(X, clusters, split_diff_scale=False):

    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(clusters) == int:
        clusters = [clusters]

    H = []
    for clusters in clusters:
        H_tmp = _construct_edge_list_from_cluster(X, clusters)

        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H



def _generate_G_from_H(H, variable_weight=False):
  
    H = np.array(H) 

    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)

    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
 
    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))

  
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        G = torch.Tensor(G)
        return G


def generate_G_from_H(H, variable_weight=False):

    if type(H) != list:   
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:    
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


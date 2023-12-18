
import math
import numpy as np
import hypergraph_construct_KNN
import hypergraph_construct_kmeans

def constructHW_knn(X,K_neigs,is_probH):

    """incidence matrix"""
    H = hypergraph_construct_KNN.construct_H_with_KNN(X,K_neigs,is_probH)

    G = hypergraph_construct_KNN._generate_G_from_H(H)

    return G

def constructHW_kmean(X,clusters):

    """incidence matrix"""
    H = hypergraph_construct_kmeans.construct_H_with_Kmeans(X,clusters)

    G = hypergraph_construct_kmeans._generate_G_from_H(H)

    return G

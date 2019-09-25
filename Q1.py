import numpy as np
from numpy import ma
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import re
import argparse
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import linalg
from scipy.stats import pearsonr

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="metabolic.edgelist.txt")
    parser.add_argument('--directed', type=int, default=1)
    args = parser.parse_args()
    return args

def get_degree(A):
    kin = np.asarray(A.sum(axis=0)).flatten()
    kout = np.asarray(A.sum(axis=1)).flatten()
    return(kin, kout)

#directed and undirected
def degree_distribution(A, networkName, directed=True):
    binNum = 30

    if (directed):
        (kin, kout) = get_degree(A)
        bins = np.linspace(0, np.log10(np.max(kin)), num=binNum)
        digitized = np.digitize(np.log10(kin), bins)
        bin_counts = np.asarray([digitized.tolist().count(i) for i in range(0,len(bins))])
        bin_counts = ma.log10(bin_counts)
        #fit the line
        a,b = ma.polyfit(bins, bin_counts, 1, full=False)
        print('best fit in degree line:\ny = {:.2f} + {:.2f}x'.format(b, a))
        yfit = [b + a * xi for xi in bins]
        fig, axs = plt.subplots(2, 1)
        axs[0].scatter(bins, bin_counts)
        axs[0].plot(bins, yfit, color="orange")
        axs[0].set_title('in-degree distribution')
        axs[0].set_xlabel('Degree (d) log base 10', fontsize="small")
        axs[0].set_ylabel('Frequency log base 10', fontsize="small")
        axs[0].set_ylim(bottom=0)

        bins = np.linspace(0, np.log10(np.max(kout)), num=binNum)
        digitized = np.digitize(np.log10(kout), bins)
        bin_counts = np.asarray([digitized.tolist().count(i) for i in range(0,len(bins))])
        bin_counts = ma.log10(bin_counts)
        print('best fit out degree line:\ny = {:.2f} + {:.2f}x'.format(b, a))
        yfit = [b + a * xi for xi in bins]
        axs[1].scatter(bins, bin_counts)
        axs[1].plot(bins, yfit, color="orange")
        axs[1].set_title('out-degree distribution')
        axs[1].set_xlabel('Degree (d) log base 10', fontsize="small")
        axs[1].set_ylabel('Frequency log base 10', fontsize="small")
        plt.subplots_adjust(hspace=0.01)
        plt.tight_layout()
        plt.savefig(networkName + 'degree.pdf')
        plt.close()

    if (not directed):
        (kin,kout) = get_degree(A)
        print (kin.shape)
        #bin the statistics
        bins = np.linspace(0, np.log10(np.max(kin)), num=binNum)
        digitized = np.digitize(np.log10(kin), bins)
        bin_counts = np.asarray([digitized.tolist().count(i) for i in range(0,len(bins))])
        bin_counts = ma.log10(bin_counts)
        #fit the line
        a,b = ma.polyfit(bins, bin_counts, 1, full=False)
        print('best fit line:\ny = {:.2f} + {:.2f}x'.format(b, a))
        yfit = [b + a * xi for xi in bins]
        plt.scatter(bins, bin_counts)
        plt.plot(bins, yfit, color="orange")
        plt.title('degree distribution')
        plt.xlabel('Degree (d) log base 10', fontsize="small")
        plt.ylabel('Frequency log base 10', fontsize="small")
        plt.ylim(bottom=0)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.tight_layout()
        plt.savefig(networkName + 'degree.pdf')
        plt.close()

#only undirected
def clustering_coefficient(A, kin, kout, n, networkName, directed):
    binNum = 30
    A3 = A.dot(A).dot(A)

    if (not directed):
        cin = np.zeros(n, dtype=float)
        for i in range(0,n):
            if (kin[i] >= 2):
                cin[i] = A3[i,i] / (kin[i]*(kin[i] - 1))

        bins = np.linspace(0, np.max(cin), num=binNum)
        print ("average clustering coefficient is " + str(np.mean(cin)))
        # digitized = np.digitize(cin, bins)
        # bin_counts = np.asarray([digitized.tolist().count(i) for i in range(0,len(bins))])

        # plt.scatter(bins, bin_counts)
        plt.hist(cin, bins=bins)
        plt.title('clustering coefficient distribution')
        plt.xlabel('Local Clustering Coefficient', fontsize="small")
        plt.ylabel('Frequency', fontsize="small")
        plt.yscale('log')
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(networkName + 'cc.pdf')
        plt.close()

        return cin

#directed and undirected
def shortest_path(A, networkName, directed=False):
    if (not directed):
        dist_mtx = sparse.csgraph.shortest_path(A, method='auto', directed=directed, unweighted=True)
        print ("dist_mtx calculated")
        #print (np.min(dist_mtx))
        numPaths = np.mean(dist_mtx, axis=0)
        #print (np.min(numPaths))
        avg = np.mean(numPaths)
        print ("average length of shortest path is " + str(avg))
        unique, counts = np.unique(numPaths, return_counts=True)
        plt.bar(unique, counts)
        plt.title('shortest path distribution')
        plt.xlabel('length of shortest path', fontsize="small")
        plt.ylabel('number of nodes', fontsize="small")
        plt.tight_layout()
        plt.savefig(networkName + 'sp.pdf')
        plt.close()

    if (directed):
        dist_mtx = sparse.csgraph.shortest_path(A, method='auto', directed=directed, unweighted=True)
        #in-degree
        #print (np.min(dist_mtx))
        Pin = np.mean(dist_mtx, axis=0)
        #print (np.min(Pin))
        avg = np.mean(Pin)
        print ("average length of in degree shortest path is " + str(avg))
        unique, counts = np.unique(Pin, return_counts=True)
        plt.bar(unique, counts)
        plt.title('shortest path distribution')
        plt.xlabel('length of shortest path', fontsize="small")
        plt.ylabel('number of nodes', fontsize="small")
        plt.tight_layout()
        plt.savefig(networkName + 'sp.pdf')
        plt.close()

#only undirected
def connected_components(A, networkName, directed=False):
    n_components, labels = sparse.csgraph.connected_components(csgraph=A, directed=directed, return_labels=True)
    componentID, numNodes = np.unique(labels, return_counts=True)
    print ("number of connected component is: " + str(n_components))
    print ("portion of nodes in GCC is: " + str(np.max(numNodes) / A.shape[0]))


#only undirected
def eigenvalue_distribution(A, networkName, directed=False):
    (kin, kout) = get_degree(A)
    n = kin.shape[0]
    binNum = 30
    if (not directed):
        D = csc_matrix(A.shape, dtype=np.int8)
        for i in range(0, A.shape[0]):
            D[i,i] = kin[i]
        L = D - A
        eigenvalues, vecs = linalg.eigs(L.asfptype(), k=(n-2))
        eigenvalues = eigenvalues.real
        #spectralGap = np.where(eigenvalues > 0, eigenvalues, np.inf).argmin()
        spectralGap = np.min(eigenvalues[eigenvalues > 0])
        print ("spectral gap is " + str(spectralGap))
        bins = np.linspace(0, np.max(eigenvalues), num=binNum)
        plt.hist(eigenvalues, bins=bins)
        plt.title('eigenvalue distribution')
        plt.xlabel('eigenvalue', fontsize="small")
        plt.ylabel('frequency', fontsize="small")
        plt.tight_layout()
        plt.savefig(networkName + 'ei.pdf')
        plt.ylim(bottom=0)
        plt.close()

#only undirected
def degree_correlations(A, networkName, directed=False):
    (kin, kout) = get_degree(A)
    #assuming only dealing with undirected networks here
    if (not directed):
        #degree correlation matrix E
        E = np.zeros((np.max(kin)+1,np.max(kin)+1))
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[0]):
                if (A[i,j] == 1):
                    k1=kin[i];
                    k2=kin[j];
                    E[k1,k2] = E[k1,k2] + 1;
        E = E / (np.sum(kin))

        Posk = kin[kin != 0]
        knn_ki = pd.DataFrame(Posk)
        knn_ki[1] = np.divide(A.dot(kin)[kin != 0], Posk)
        knn_k = knn_ki.groupby(0).mean()
        pearson_correlation = pearsonr(knn_k.index, knn_k[1])
        # knn = np.zeros(kin.shape[0])
        # for i in range(0,knn.shape[0]):
        #     print (A[i].shape)
        #     print (kin.shape)
        #     knn[i] = float(np.sum(np.dot(A[i], kin)) / kin[i])
        # pearson_correlation = pearsonr(kin, knn)

        print("overall correlation is " + str(pearson_correlation))
        plt.imshow(E, cmap='gray_r', origin='lower')
        plt.colorbar()
        plt.title('degree correlations')
        plt.xlabel('Degree d', fontsize="small")
        plt.ylabel('Degree d', fontsize="small")
        plt.savefig(networkName + 'dc.pdf')
        plt.close()

def degree_cc(A, networkName, cin, kin, directed=False):

    plt.scatter(kin, cin)
    plt.title('degree-clustering coefficient relation')
    plt.xlabel('Degree (d)', fontsize="small")
    plt.ylabel('local clustering coefficient', fontsize="small")
    plt.ylim(bottom=0)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(networkName + 'dlcc.pdf')
    plt.close()


#Actor undirected
#Collaboration undirected
#Internet undirected
#Power Grid undirected
#Protein undirected 
#Phone Calls undirected
#Citation directed
#Metabolic directed
#Email directed
#WWW directed


#python Q1.py --file metabolic.edgelist.txt --directed 1


def execute(networkName, directed):
    plt.switch_backend('agg')
    d_dir = "networks/"
    
    mpl.rcParams['lines.markersize'] = 5

    edgelist = open(d_dir+networkName, "r")
    print ("accessing " + networkName)
    print ("is directed : " + str(directed))

    lines = list(edgelist.readlines())
    n=0
    for line in lines:
    	values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+",line)
    	if (int(values[0]) > n):
    		n = int(values[0])
    	if (int(values[1]) > n):
    		n = int(values[1])
    n = n + 1
    print ("number of nodes n is " + str(n))
    
    #adjacency matrix A
    A = csc_matrix((n,n), dtype=np.int8)
    for line in lines:
        values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+",line)
        i = int(values[0])
        j = int(values[1])
        if (i == j):
            continue
        A[i,j] = 1
        if (not directed):
            A[j,i] = 1

    print ("there are " + str(A.count_nonzero()) + " edges in the adjacency matrix")

    degree_distribution(A, networkName, directed=directed)
    shortest_path(A, networkName, directed=directed)
    
    #load A as undirected
    if (directed):
        A = csc_matrix((n,n), dtype=np.int8)
        for line in lines:
            values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+",line)
            i = int(values[0])
            j = int(values[1])
            if (i == j):
                continue
            A[i,j] = 1
            A[j,i] = 1
    directed = False
    (kin, kout) = get_degree(A)
    cin = clustering_coefficient(A, kin, kout, n, networkName, directed)
    connected_components(A, networkName, directed=directed)
    eigenvalue_distribution(A, networkName, directed=directed)
    degree_correlations(A, networkName, directed=directed)
    degree_cc(A, networkName, cin, kin, directed=directed)
    edgelist.close()

def main():
    args = get_args()
    networkName = args.file 
    directed = False
    if (args.directed == 1):
        directed = True
    execute(networkName, directed)

if __name__ == "__main__":
    main()






import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import re
import argparse
from scipy.sparse import csc_matrix

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="ABnetwork")
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--edges', type=int, default=100)
    args = parser.parse_args()
    return args

def save_edgelist(A, n, networkName):
    Fname = networkName + ".edgelistABtweek.txt"
    Fedgelist = open( Fname,"w")
    for i in range(0, n):
        for j in range(0, n):
            if A[i,j] == 1:
                Fedgelist.write(str(i) + "     " + str(j) + " \n")
    Fedgelist.close()
    


# tweak the AB model to have a constant term in probability for all nodes
def add_node(d, A, n, incre, index):
    epsilon = 0.5
    plist = (d + epsilon) / ( np.sum(d) + epsilon * n)
    i = index
    exist = []

    for k in range(0, incre):
        choices = np.random.choice(np.arange(n), 1, p=plist)
        j = choices[0]
        while j in exist:
            choices = np.random.choice(np.arange(n), 1, p=plist)
            j = choices[0]
        A[i,j] = 1
        d[i] = d[i] + 1
        d[j] = d[j] + 1
        #update plist
        plist = (d + epsilon) / ( np.sum(d) + epsilon * n)
        exist.append(j)
    return (d,A)


def execute(networkName, n, e):

    #assume there is always edge between first 2 nodes
    #incre is the number of edges to be added per iteration
    incre = (e-1) // (n-2) + 1
    print (incre)
    #number of edges to be added to the last iteration
    remains = (e-1) - (n-2) * (incre-1)
    print (remains) 
    #assuming all edges are directed
    d = np.zeros(n)
    A = csc_matrix((n,n))
    A[0,1] = 1
    d[0] = 1
    d[1] = 1
    print("network initiated")
    for i in range(2, n):
        if (remains == 0):
            (d, A) = add_node(d, A, n, incre-1, i)
        else:
            (d, A) = add_node(d, A, n, incre, i)
            remains = remains - 1
    print ("The network has " + str(A.shape[0]) + " nodes")
    print ("The network now has " + str(A.count_nonzero()) + " edges")
    print ("saving edgelist")
    save_edgelist(A, n, networkName)




    #store the current list of degrees of nodes in d
    #store the adjacency matrix in A


def main():
    args = get_args()
    networkName = args.file 
    n = args.size
    e = args.edges
    execute(networkName, n, e)

if __name__ == "__main__":
    main()






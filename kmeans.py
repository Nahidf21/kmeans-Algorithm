import numpy as np
import random
import matplotlib.pyplot as plt

def dataset():
    """
    Notice that there are no labels in this dataset
    """
    D = np.array([[3, 1.4],
                  [1, 1.3],
                  [0, 0],
                  [3, 3],
                  [3.5, 1.2],
                  [2, 2.5],
                  [1, 0.9]])
    return D


def initial_centroids(D, M):
    """
    :param D: the data points
    :param M: the number of centroids
    :return: the initial, randomly assigned, centroids
    """
    arr = np.arange(M)  # 0, 1, ..., M-1
    np.random.shuffle(arr)
    ix = arr[:M]
    return D[ix,:]


def all_distances(x, C):
    """
    all distances from the current data point x to every centroid in C
    """

    d = []
    for c in C:
        dist = np.linalg.norm(x-c)
        d.append((dist))

    d = np.array(d)
    return d


def grouping(D, C):
    """
    Go through each point D_i and find the closest centroid C_j to D_i,
    then put D_i in the group G_j.
    :param D: the set of data points (each point is on a row)
    :param C: the set of centrodis (each centroid is on a row
    :return: G: the set of collections
    """

    # the list of lists where list i has the indices of data points closest
    # to centroid i.
    G = []
    for c in range(len(C)):
        G.append([])

    # the number of data points
    N = len(D)

    for n in range(N):
        x = D[n,:]  # the current instance
        d = all_distances(x, C)
        imin = np.argmin(d) # index of the smallest distance
        G[imin].append(x)   # x becomes the member of the group G[imin]

    return G


def averaging(G):
    # the number of centroids
    M = len(G)
    A = len(G[0][0])    # the number of attributes

    # updated centroids
    C = np.zeros((M, A))

    for k in range(M):
        temp = np.zeros(A)
        for D in G[k]:
            temp += D
        C[k,:] = temp / len(G[k])

    return C


def kmeans(Cinit, D):
    C = Cinit
    for _ in range(10):
        # grouping
        G = grouping(D, C)

        # averaging (update centroids by averaging the members in each group)
        C = averaging(G)
        print(C)

    return (C, G)


def myplot(P, G, marker):
    k = 0
    for g in G:
        for D in g:
            plt.plot(D[0], D[1], marker[k])
        k += 1
    plt.xlabel('a_1')
    plt.ylabel('a_2')

def myplot_centroids(C, marker):
    k = 0
    for c in C:
        plt.plot(c[0], c[1], marker[k])
        k += 1
    plt.xlabel('a_1')
    plt.ylabel('a_2')



# we are looking for 2 centroids (splitting into 2 groups)
M = 2
D = dataset()
Cinit = initial_centroids(D, M)
(C, G) = kmeans(Cinit, D)

#plot the data points
markers = ('b*', 'r*')
myplot(D, G, markers)

markers = ('bo', 'ro')
myplot_centroids(C, markers)

plt.show()
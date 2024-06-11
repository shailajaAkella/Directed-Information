import math
import numpy as np
from scipy.spatial.distance import pdist, squareform

# author: Shailaja Akella, shailaja.akella93@gmail.com

class directed_information:

    def __init__(self, multiplier_X=None, multiplier_Y=None, memory=5, alpha=1.01):

        """
        :param memory: total length of past history to consider
        :param multiplier_X: multiplier to adjust kernel width of the Gram matrix for X
        :param multiplier_Y: multiplier to adjust kernel width of the Gram matrix for Y
        :param alpha: order of entropy
        """

        self.multiplier_X = multiplier_X
        self.multiplier_Y = multiplier_Y
        self.alpha = alpha
        self.memory = memory
        self.di = None

    def GaussMat(self, mat, mult=None):

        """
        Calculates the Gram matrix using Gaussian kernel evaluates the distances between input samples after projection to RKHS
        :param mat: input matrix
        :param mult: Multiplier to adjust kernel width of the Gram matrix for matrix
        :return: Gram matrix
        """

        # Ensure mat is 2-dimensional
        if len(mat.shape) == 1:
            mat = mat.reshape(-1, 1)

        # Get the dimensions of mat
        N, T = mat.shape

        # Calculate bandwidth using Silverman's rule
        if mult:
            sig = (mult) * (1.06 * np.nanstd(mat)) * (N ** (-1 / 5))
        else:
            sig = (1.06 * np.nanstd(mat)) * (N ** (-1 / 5))

        # scott's rule: sig = N**(-1/(4+T))

        # Calculate pairwise squared Euclidean distances
        pairwise_sq_dists = squareform(pdist(mat, 'sqeuclidean'))

        # Calculate gram matrix using Gaussian kernel
        return np.exp(-pairwise_sq_dists / sig ** 2)

    def alpha_entropy(self, mat, mult=None):
        """
        Calculates entropy

        :param mat: input matrix
        :param mult: Multiplier to adjust kernel width of the Gram matrix for matrix
        :return: entropy
        """

        # Get Gram matrices
        N = len(mat)
        K = self.GaussMat(mat, mult=mult) / N

        # entropy estimation using the eigen spectrum
        L = np.linalg.eigvalsh(K)
        absL = np.abs(L)
        H = (1 / (1 - self.alpha)) * math.log2(np.min([np.sum(absL ** self.alpha), 0.9999]))

        return H

    def joint_alpha_entropy(self, mat1, mat2, mat3=None, mult1=None, mult2=None, mult3=None):
        """
        Calculates joint alpha entropy between two or three variables
        :param mat1: input matrix1
        :param mat2: input matrix2
        :param mat3: input matrix3
        :param mult1: Multiplier to adjust kernel width of the Gram matrix for matrix1
        :param mult2: Multiplier to adjust kernel width of the Gram matrix for matrix2
        :param mult3: Multiplier to adjust kernel width of the Gram matrix for matrix3
        :return: joint entropy
        """

        if mat3 is None:
            mat3 = []

        # Get gram matrices
        N = len(mat1)
        K1 = self.GaussMat(mat1, mult=mult1) / N
        K2 = self.GaussMat(mat2, mult=mult2) / N

        # Calculate Hadamard product
        if len(mat3) == 0:
            prodK = K1 * K2 * N
        else:
            K3 = self.GaussMat(mat3, mult=mult3) / N
            prodK = K1 * K2 * K3 * (N ** 2)

        # calculate joint alpha entropy using the eigen-spectrum
        L = np.linalg.eigvalsh(prodK)
        absL = np.abs(L)
        H = (1 / (1 - self.alpha)) * math.log2(np.min([np.sum(absL ** self.alpha), 0.9999]))  # joint entropy

        return H

    def conditional_mi(self, x_past, y_past, y_present):
        """
        Calculates the conditional mutual information at time point i,
        I(Xpast; Ypresent | Ypast) = H(Xpast, Ypast) + H(Ypresent, Ypast) - H(Xpast, Ypresent, Ypast) - H(Ypast)

        :return: conditional mutual information
        """

        # H(Ypast)
        H_ypast = self.alpha_entropy(y_past, mult=self.multiplier_Y)

        # H(Xpast, Ypast)
        H_xpast_ypast = self.joint_alpha_entropy(x_past, y_past, mult1=self.multiplier_X, mult2=self.multiplier_Y)

        # H(Ypresent, Ypast)
        H_ypresent_ypast = self.joint_alpha_entropy(y_present, y_past, mult1=self.multiplier_Y, mult2=self.multiplier_Y)

        # H(Xpast, Ypresent, Ypast)
        H_xyyp = self.joint_alpha_entropy(x_past, y_present, y_past, mult1=self.multiplier_X, mult2=self.multiplier_Y,
                                          mult3=self.multiplier_Y)

        return H_ypresent_ypast  + H_xpast_ypast - H_xyyp - H_ypast

    def DI(self, X, Y):
        """
        calculates directed information from (X -> Y)
        :param X: causal time series
        :param Y: effect time series
        :return directed information
        """

        di = 0
        T = len(X)

        for m in range(1, self.memory):

            # initialize
            x_past, y_past, y_present = np.zeros((T - m, m)), np.zeros((T - m, m)), np.zeros(T - m)

            # gather past and present instances
            for t in range(T - m):
                x_past[t] = X[t:t + m]
                y_past[t] = Y[t:t + m]
                y_present[t] = Y[t + m]

            # evaluate conditional mutual information and add to DI
            di += self.conditional_mi(x_past, y_past, y_present)

        self.di = di

        return self.di

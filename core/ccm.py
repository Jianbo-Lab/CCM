"""
This script contains an object and a function for kernel feature selection via conditional covariance minimization.
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
import kernel
def center(X):
    """Returns the centered version of the given square matrix, namely

        (I - (1/n) 1 1^T) X (I - (1/n) 1 1^T)
            = X - (1/n) 1 1^T X - (1/n) X 1 1^T + (1/n^2) 1 1^T X 1 1^T.

    Args:
        X: An (n, n) matrix.

    Returns:
        The row- and column-centered version of X.
    """

    mean_col = tf.reduce_mean(X, axis=0, keep_dims=True)
    mean_row = tf.reduce_mean(X, axis=1, keep_dims=True)
    mean_all = tf.reduce_mean(X)
    return X - mean_col - mean_row + mean_all

def project(v, z):
    """Returns the Euclidean projection of the given vector onto the positive
    simplex, i.e. the set

        {w : \sum_i w_i = z, w_i >= 0}.

    Implements the core given in Figure 1 of Duchi et al. (2008) [1].

    [1] http://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

    Args:
        v: A vector.
        z: The desired sum of the components. Must be strictly positive.

    Returns:
        The Euclidean projection of v onto the positive simplex of size z.
    """

    assert v.ndim == 1
    assert z > 0

    z = float(z)

    mu = np.sort(v)[::-1]
    mu_cumsum = np.cumsum(mu)
    max_index = np.nonzero(mu * np.arange(1, len(v) + 1) > mu_cumsum - z)[0][-1]
    theta = (mu_cumsum[max_index] - z) / (max_index + 1)
    return np.maximum(v - theta, 0)

class CCM(object):
    """
    This object implements kernel feature selection via conditional covariance minimization.

    Example usage:
    X = np.random.randn(100,10); Y = (X[0] > 0).astype(float)

    epsilon = 0.1; k = 1
    fs = CCM(X, Y, transform_Y = None, epsilon = epsilon)
    w = fs.solve_gradient_descent(k=1)
    ranks = fs.ranks 
    print(w)
    print((-w).argsort()[:10]) 

    """
    def __init__(
            self,
            X,
            Y, 
            transform_Y,
            epsilon,
            D_approx = None
    ): 

        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert X.ndim == 2
        assert Y.ndim == 1

        n, d = X.shape
        assert Y.shape == (n,)
        self.d = d

        # Whitening transform for X.
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        # Use Gaussian kernel with automatically chosen sigma.
        sigma = np.median(np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)) 
        kernel_X = kernel.GaussianKernel(sigma)

        # Transform Y.
        assert transform_Y in (None, "binary", "one-hot")

        if transform_Y == "binary":
            values = sorted(set(Y.ravel()))
            assert len(values) == 2
            Y_new = np.zeros(n)
            Y_new[Y == values[0]] = -1
            Y_new[Y == values[1]] = 1
            Y = Y_new

        elif transform_Y == "one-hot":
            values = sorted(set(Y.ravel()))
            Y_new = np.zeros((n, len(values)))
            for i, value in enumerate(values):
                Y_new[Y == value, i] = 1
            Y = Y_new

        if Y.ndim == 1:
            Y = Y[:, np.newaxis]

        Y = Y - Y.mean(0)

        # Build graph for loss and gradients computation. 
        with tf.Graph().as_default():
            self.w = tf.placeholder(tf.float64, shape=d) 
            self.inputs = [self.w]

            if D_approx is None:

                G_X_w = center(kernel_X(X * self.w))

                G_X_w += n * epsilon * np.eye(n)

                G_X_w_inv = tf.matrix_inverse(G_X_w)

            else:

                U_w = kernel_X.random_features(X * self.w, D_approx)

                V_w = tf.matmul(
                        tf.subtract(
                            tf.eye(n, dtype=tf.float64),
                            tf.divide(
                                tf.ones((n,n), dtype=tf.float64),
                                tf.constant(float(n), dtype=tf.float64))),
                        U_w)

                # eq. 21, arXiv:1707.01164, omitting constant term
                
                G_X_w_inv = tf.matmul(
                                tf.scalar_mul(
                                    tf.constant(-1.0, dtype=tf.float64),
                                    V_w),
                                tf.matmul(
                                    tf.matrix_inverse(
                                        tf.add(
                                            tf.matmul(
                                                V_w,
                                                V_w,
                                                transpose_a=True),
                                            tf.multiply(
                                                tf.constant(n * epsilon, dtype=tf.float64),
                                                tf.eye(D_approx, dtype=tf.float64)))),
                                    V_w,
                                    transpose_b=True))

            self.loss = tf.trace(tf.matmul(
                    Y, tf.matmul(G_X_w_inv, Y), transpose_a=True))

            self.gradients = tf.gradients(self.loss, self.inputs)

            self.sess = tf.Session()

    def solve_gradient_descent(self, num_features, learning_rate = 0.001, iterations = 1000, verbose=True):

        assert num_features <= self.d

        # Initialize w. 
        w = project(np.ones(self.d), num_features)
        inputs = [w]

        def clip_and_project(w):
            """ clip and project w onto the simplex."""
            w = w.clip(0, 1)
            if w.sum() > num_features:
                w = project(w, num_features)
            return w

        for iteration in range(1, iterations + 1):

            # Compute loss and print.
            if verbose:
                loss = self.sess.run(self.loss, feed_dict=dict(
                    zip(self.inputs, inputs)))
                print("iteration {} loss {}".format(iteration, loss))

            # Update w with projected gradient method. 
            gradients = self.sess.run(self.gradients, feed_dict=dict(zip(self.inputs, inputs)))
            for i, gradient in enumerate(gradients):
                inputs[i] -= learning_rate * gradient
            inputs[0] = clip_and_project(inputs[0])

        # Compute rank of each feature based on weight.
        # Random permutation to avoid bias due to equal weights.
        idx = np.random.permutation(self.d) 
        permutated_weights = inputs[0][idx]  
        permutated_ranks=(-permutated_weights).argsort().argsort()+1
        self.ranks = permutated_ranks[np.argsort(idx)]


        return inputs[0] 

def ccm(X, Y, num_features, type_Y, epsilon, learning_rate = 0.001, 
    iterations = 1000, D_approx = None, verbose = True): 
    """
    This function carries out feature selection via CCM.
    Args:
        X: An (n, d) numpy array.

        Y: An (n,) numpy array.

        num_features: int. Number of selected features.

        type_Y: str. Type of the response variable. 
        Possible choices: 'ordinal','binary','categorical','real-valued'.

        learning_rate: learning rate of project gradient method.

        iterations: number of iterations for optimization.

        D_approx: optional, rank-D kernel approximation

        verbose: If print loss at each update.

    Return:
        ranks: An (d,) numpy array, containing permutations of 
        [1,2,...,d]

    """
    assert type_Y in ('ordinal','binary','categorical','real-valued')
    if type_Y == 'ordinal' or type_Y == 'real-valued':
        transform_Y = None 
    elif type_Y == 'binary':
        transform_Y = 'binary'
    elif type_Y == 'categorical':
        transform_Y = 'one-hot'

    fs = CCM(X, Y, transform_Y, epsilon, D_approx = D_approx)
    w = fs.solve_gradient_descent(num_features, learning_rate, iterations, verbose)
    if verbose:
        print('The weights on featurs are: ', w)
    ranks = fs.ranks 
    return ranks


if __name__ == "__main__": 
    
    X = np.random.randn(100,10); Y = (X[:,0] > 0).astype(float)

    epsilon = 0.1; num_features = 1; type_Y = 'binary'
    print(ccm(X, Y, num_features, type_Y, epsilon)) 













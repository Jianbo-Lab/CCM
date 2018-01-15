import math
import tensorflow as tf

class Kernel(object):
    """A class representing a kernel function."""

    def __call__(self, X):
        """Returns the Gram matrix for the given data.

        Args:
            X: An (n, d) matrix, consisting of n data points with d dimensions.

        Returns:
            An (n, n) matrix where the (i, j)th entry is k(x_i, x_j).
        """

        pass

    def random_features(self, X, D):
        """Returns a random Fourier feature map for the given data, following
        the approach in Rahimi and Recht [1].
        
        [1] https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
        
        Args:
            X: An (n, d) matrix, consisting of n data points with d dimensions. 
            D (int): The number of features in the feature map.

        Returns:
            An (n, D) matrix for which the inner product of row i and row j is
            approximately k(x_i, x_j).
        """

        raise NotImplementedError

class LinearKernel(Kernel):
    """The linear kernel, defined by the inner product

        k(x_i, x_j) = x_i^T x_j.
    """

    def __call__(self, X):
        return tf.matmul(X, X, transpose_b=True)

class PolynomialKernel(Kernel):
    """The polynomial kernel, defined by

        k(x_i, x_j) = (a x_i^T x_j + b)^d

    with coefficient a, bias b, and degree d.
    """

    def __init__(self, a, b, d):
        self.a = a
        self.b = b
        self.d = d

    def __call__(self, X):
        return (self.a * tf.matmul(X, X, transpose_b=True) + self.b) ** self.d

class LaplacianKernel(Kernel):
    """The Laplacian kernel, defined by

        k(x_i, x_j) = exp(-||x_i - x_j||_1 / sigma)

    with bandwidth parameter sigma.
    """

    def __init__(self, sigma):
        assert sigma > 0
        self.sigma = sigma

    def __call__(self, X):
        X_rowdiff = tf.expand_dims(X, 1) - tf.expand_dims(X, 0)
        return tf.exp(-tf.reduce_sum(tf.abs(X_rowdiff), 2) / self.sigma)

class GaussianKernel(Kernel):
    """The Gaussian kernel, defined by

        k(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 sigma^2))

    with bandwidth parameter sigma.
    """

    def __init__(self, sigma):
        assert sigma > 0
        self.sigma = sigma

    def __call__(self, X):
        X_rowdiff = tf.expand_dims(X, 1) - tf.expand_dims(X, 0)
        return tf.exp(-tf.reduce_sum(X_rowdiff ** 2, 2) / (2 * self.sigma ** 2))

    def random_features(self, X, D):
        omega = tf.random_normal(
            tf.stack([tf.shape(X)[1], D]), stddev=1.0 / self.sigma, dtype=X.dtype)
        b = tf.random_uniform([D], maxval=2 * math.pi, dtype=X.dtype)
        return math.sqrt(2.0 / D) * tf.cos(tf.matmul(X, omega) + b)

class EqualityKernel(Kernel):
    """The equality kernel, defined by

        k(x_i, x_j) = f(x_i_1 == x_j_1, x_i_2 == x_j_2, ...)

    where f is either "mean" or "product".
    """

    def __init__(self, composition="product"):
        assert composition in ("mean", "product")
        self.composition = composition

    def __call__(self, X):
        X_equal = tf.to_double(tf.equal(tf.expand_dims(X, 0), tf.expand_dims(X, 1)))
        reduce = {
            "mean": tf.reduce_mean,
            "product": tf.reduce_prod
        }[self.composition]
        return reduce(X_equal, reduction_indices=2)

#
# Twisted Gaussian (banana) distribution toy log pdf.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy.stats


class TwistedGaussianLogPDF(pints.LogPDF):
    """
    Twisted multivariate Gaussian 'banana' with un-normalised density [1]:

    .. math::
        p(x_1, x_2, x_3, ..., x_n) \\propto
            \\pi(\\phi(x_1, x_2, x_2, ..., x_n))

    where pi is the multivariate normal density with covariance matrix
    :math:`\\Sigma=\\text{diag}(100, 1, 1, ..., 1)` and

    .. math::
        \\phi(x_1,x_2,x_3,...,x_n) = (x_1, x_2 + b x_1^2 - V b, x_3, ..., x_n),

    Arguments:

    ``dimension``
        Problem dimension (``n``), must be 2 or greater.
    ``b``
        "Bananicity": ``b = 0.01`` induces mild non-linearity in target
        density, while non-linearity for ``b = 0.1`` is high.
        Must be greater than or equal to zero.
    ``V``
        Offset (see equation).

    *Extends:* :class:`pints.LogPDF`.

    [1] Adaptive proposal distribution for random walk Metropolis algorithm
    Haario, Saksman, Tamminen (1999) Computational Statistics.
    """
    def __init__(self, dimension=10, b=0.1, V=100):
        # Check dimension
        self._n_parameters = int(dimension)
        if self._n_parameters < 2:
            raise ValueError('Dimension must be 2 or greater.')

        # Check parameters
        self._b = float(b)
        if self._b < 0:
            raise ValueError('Argument `b` cannot be negative.')
        self._V = float(V)

        # Create phi
        self._sigma = np.eye(self._n_parameters)
        self._sigma[0, 0] = 100
        self._phi = scipy.stats.multivariate_normal(
            np.zeros(self._n_parameters), self._sigma)

    def __call__(self, x):
        y = np.array(x, copy=True, dtype='float')
        y[0] = float(y[0]) / np.sqrt(self._V)
        y[1] += self._b * ((x[0] ** 2) - self._V)
        return self._phi.logpdf(y)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`.
        """
        L = self.__call__(x)

        V = self._V
        b = self._b
        dx_first = 0.5 * (
            -x[0] / (50 * V) - 4 * b * x[0] * (b * (x[0]**2 - V) + x[1])
        )
        dy_first = -b * (x[0]**2 - V) - x[1]
        if len(x) > 2:
            dL = [[dx_first, dy_first], (-np.array(x[2:])).tolist()]
            dL = [item for sublist in dL for item in sublist]
        else:
            dL = [dx_first, dy_first]
        return L, np.array(dL)

    def kl_divergence(self, samples):
        """
        Calculates the approximate Kullback-Leibler divergence between a
        given list of samples and the distribution underlying this LogPDF.

        The returned value is (near) zero for perfect sampling, and then
        increases as the error gets larger.

        See: https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
        """
        # Check size of input
        if not len(samples.shape) == 2:
            raise ValueError('Given samples list must be nx2.')
        if samples.shape[1] != self._n_parameters:
            raise ValueError(
                'Given samples must have length ' + str(self._n_parameters))

        # Untwist the given samples, making them Gaussian again
        y = np.array(samples, copy=True, dtype='float')
        y[:, 0] /= np.sqrt(self._V)
        y[:, 1] += self._b * ((samples[:, 0] ** 2) - self._V)

        # Calculate the Kullback-Leibler divergence between the given samples
        # and the multivariate Gaussian distribution underlying this banana.
        # From wikipedia:
        #
        # k = dimension of distribution
        # dkl = 0.5 * (
        #       trace(s1^-1 * s0)
        #       + (m1 - m0)T * s1^-1 * (m1 - m0)
        #       + log( det(s1) / det(s0) )
        #       - k
        #       )
        #
        # For this distribution, s1 is the identify matrix, and m1 is zero,
        # so it simplifies to
        #
        # dkl = 0.5 * (trace(s0) + m0.dot(m0) - log(det(s0)) - k))
        #
        m0 = np.mean(y, axis=0)
        s0 = np.cov(y.T)
        s1 = self._sigma
        m1 = np.zeros(self.n_parameters())
        s1_inv = np.linalg.inv(s1)
        return 0.5 * (
            np.trace(np.matmul(s1_inv, s0)) +
            np.matmul(np.matmul(m1 - m0, s1_inv), m1 - m0) -
            np.log(np.linalg.det(s0)) +
            np.log(np.linalg.det(s1)) -
            self._n_parameters)

    def distance(self, samples):
        """
        Returns approximate Kullback-Leibler divergence of samples from
        underyling distribution (see `kl_divergence`)
        """
        return self.kl_divergence(samples)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._n_parameters

    def sample(self, n):
        """
        Generates samples from the underlying distribution.
        """
        if n < 0:
            raise ValueError('Number of samples cannot be negative.')

        x = self._phi.rvs(n)
        x[:, 0] *= np.sqrt(self._V)
        x[:, 1] -= self._b * (x[:, 0] ** 2 - self._V)
        return x

    def suggested_bounds(self):
        """
        Returns suggested boundaries for prior (typically used in performance
        testing)
        """
        # based on independent sampling think the following hard bounds are ok
        bounds = [[-50, 50], [-100, 100]]
        return np.transpose(bounds).tolist()

#
# High-dimensional normal log-pdf.
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


class HighDimensionalNormalLogPDF(pints.LogPDF):
    """
    High-dimensional multivariate normal log pdf, with tricky off-diagonal
    covariances.

    *Extends:* :class:`pints.LogPDF`.
    """
    def __init__(self, dimension=20, rho=0.5):
        self._n_parameters = int(dimension)
        if self._n_parameters < 1:
            raise ValueError('Dimension must be 1 or greater.')
        rho = float(rho)
        # bounds must satisfy:
        # https://stats.stackexchange.com/questions/72790/
        # bound-for-the-correlation-of-three-random-variables
        if rho > 1.0:
            raise ValueError('rho must be between -1 / (dims - 1) and 1.')
        if rho < -float(1.0 / (self._n_parameters - 1)):
            raise ValueError('rho must be between -1 / (dims - 1) and 1.')
        self._rho = rho

        # Construct mean array
        self._mean = np.zeros(self._n_parameters)

        # Construct covariance matrix where diagonal variances = i
        # and off-diagonal covariances = rho * sqrt(i) * sqrt(j)
        cov = np.arange(1, 1 + self._n_parameters).reshape(
            (self._n_parameters, 1))
        cov = cov.repeat(self._n_parameters, axis=1)
        cov = np.sqrt(cov)
        cov = self._rho * cov * cov.T
        np.fill_diagonal(cov, 1 + np.arange(self._n_parameters))
        self._cov = cov

        # Construct scipy 'random variable'
        self._var = scipy.stats.multivariate_normal(self._mean, self._cov)

    def __call__(self, x):
        return self._var.logpdf(x)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._n_parameters

    def rho(self):
        """ Returns rho (correlation between dimensions) """
        return self._rho

    def kl_divergence(self, samples):
        """
        Returns approximate Kullback-Leibler divergence between samples
        and underlying distribution.

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

        # Calculate the Kullback-Leibler divergence between the given samples
        # and the multivariate normal distribution underlying this banana.
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
        y = np.array(samples, copy=True)
        m0 = np.mean(y, axis=0)
        s0 = np.cov(y.T)
        s1 = self._cov
        m1 = self._mean
        s1_inv = np.linalg.inv(s1)
        return 0.5 * (
            np.trace(np.matmul(s1_inv, s0)) +
            np.matmul(np.matmul(m1 - m0, s1_inv), m1 - m0) -
            np.log(np.linalg.det(s0)) +
            np.log(np.linalg.det(s1)) -
            self._n_parameters)

    def suggested_bounds(self):
        """
        Returns suggested boundaries for prior (typically used in performance
        testing)
        """
        # maximum variance in one dimension is n_parameters, so use
        # 3 times sqrt of this as prior bounds
        magnitude = 3 * np.sqrt(self.n_parameters())
        bounds = np.tile([-magnitude, magnitude], (self.n_parameters(), 1))
        return np.transpose(bounds).tolist()

    def distance(self, samples):
        """
        Returns approximate Kullback-Leibler divergence between samples
        and underlying distribution
        """
        return self.kl_divergence(samples)

    def sample(self, n_samples):
        """
        Generates independent samples from the underlying distribution.
        """
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError(
                'Number of samples must be greater than or equal to 1.')
        return self._var.rvs(n_samples)

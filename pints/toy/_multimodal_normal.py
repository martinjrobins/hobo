#
# Multi-model normal log pdf
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


class MultimodalNormalLogPDF(pints.LogPDF):
    """
    Multimodal (un-normalised) multivariate Normal distribution.

    By default, the distribution is on a 2-dimensional space, with modes at
    at ``(0, 0)`` and ``(10, 10)`` with independent unit covariance matrices.

    Examples::

        # Default 2d, bimodal
        f = pints.toy.MultimodalNormalLogPDF()

        # 3d bimodal
        f = pints.toy.MultimodalNormalLogPDF([[0, 1, 2], [10, 10, 10]])

        # 2d with 3 modes
        f = pints.toy.MultimodalNormalLogPDF([[0, 0], [5, 5], [5, 0]])

    Arguments:

    ``modes``
        A list of points that will form the modes of the distribution. Must all
        have the same dimension.
        If not set, the method will revert to the bimodal distribution
        described above.
    ``covariances``
        A list of covariance matrices, one for each mode. If not set, a unit
        matrix will be used for each.

    *Extends:* :class:`pints.LogPDF`.
    """
    def __init__(self, modes=None, covariances=None):

        # Check modes
        if modes is None:
            self._n_parameters = 2
            self._modes = [[0, 0], [10, 10]]
        else:
            if len(modes) < 1:
                raise ValueError(
                    'Argument `modes` must be `None` or a non-empty list of'
                    ' modes.')
            self._modes = [pints.vector(mode) for mode in modes]
            self._n_parameters = len(modes[0])
            for mode in self._modes:
                if len(mode) != self._n_parameters:
                    raise ValueError(
                        'All modes must have same dimension.')

        # Check covariances
        if covariances is None:
            self._covs = [np.eye(self._n_parameters)] * len(self._modes)
        else:
            if len(covariances) != len(self._modes):
                raise ValueError(
                    'Number of covariance matrices must equal number of'
                    ' modes.')
            self._covs = [np.array(cov, copy=True) for cov in covariances]
            for cov in self._covs:
                if cov.shape != (self._n_parameters, self._n_parameters):
                    raise ValueError(
                        'Covariance matrices must have shape (d, d), where d'
                        ' is the dimension of the given modes.')

        # Create scipy 'random variables'
        self._vars = [
            scipy.stats.multivariate_normal(mode, self._covs[i])
            for i, mode in enumerate(self._modes)]

    def __call__(self, x):
        f = np.sum([var.pdf(x) for var in self._vars])
        return -float('inf') if f == 0 else np.log(f)

    def n_parameters(self):
        """ See :meth:`pints.LogPDF.n_parameters()`. """
        return self._n_parameters

    def sample(self, n_samples):
        """
        Generates independent samples from the underlying distribution.
        """
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError(
                'Number of samples must be greater than or equal to 1.')

        samples = np.zeros((n_samples, self._n_parameters))
        num_modes = len(self._modes)
        for i in range(n_samples):
            rand_mode = np.random.choice(num_modes, 1)[0]
            samples[i, :] = self._vars[rand_mode].rvs(1)
        return samples

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`.
        """
        L = self.__call__(x)

        # See page 45 of http://www.math.uwaterloo.ca/~hwolkowi//matrixcookbook.pdf

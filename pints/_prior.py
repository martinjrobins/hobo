#
# Defines different prior distributions
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy


class Prior(object):
    """
    Represents a prior distribution on a vector of variables.

    Given any point ``x`` in the parameter space, a prior function needs to be
    able to evaluate the probability density ``f(x)`` (such that the integral
    over ``f(x)dx`` equals ``1``).
    """
    def dimension(self):
        """
        Returns the dimension of the space this prior is defined on.
        """
        raise NotImplementedError

    def __call__(self, x):
        """
        Returns the probability density for point ``x``.
        """
        raise NotImplementedError

    def random_sample(self,numSamples):
        """
        Returns a random sample from the prior.
        """
        raise NotImplementedError


class ComposedPrior(Prior):
    """
    *Extends:* :class:`Prior`

    Prior composed of one or more sub-priors.
    The evaluation of the composed prior assumes the input priors are all
    independent from each other

    For example: ``p = ComposedPrior(prior1, prior2, prior2)``.
    """
    def __init__(self, *priors):
        # Check if sub-priors given
        if len(priors) < 1:
            raise ValueError('Must have at least one sub-prior')

        # Check if proper priors, count dimension
        self._dimension = 0
        for prior in priors:
            if not isinstance(prior, Prior):
                raise ValueError('All sub-priors must extend Prior')
            self._dimension += prior.dimension()

        # Store
        self._priors = priors

    def dimension(self):
        """See :meth:`Prior.dimension()`."""
        return self._dimension

    def __call__(self, x):
        output = 1.0
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.dimension()
            output *= prior(x[lo:hi])
        return output


class UniformPrior(Prior):
    """
    *Extends:* :class:`Prior`

    Defines a uniform prior over a given range.

    For example: ``p = UniformPrior([1,1,1], [10, 10, 100])``, or
    ``p = UniformPrior(Boundaries([1,1,1], [10, 10, 100]))``.
    """
    def __init__(self, lower_or_boundaries, upper=None):
        # Parse input arguments
        if upper is None:
            if not isinstance(lower_or_boundaries, pints.Boundaries):
                raise ValueError(
                    'UniformPrior requires a lower and an upper bound, or a'
                    ' single Boundaries object.')
            self._boundaries = lower_or_boundaries
        else:
            self._boundaries = pints.Boundaries(lower_or_boundaries, upper)

        # Cache dimension
        self._dimension = self._boundaries.dimension()

        # Cache output value
        self._value = 1.0 / np.product(self._boundaries.range())

    def dimension(self):
        """See :meth:`Prior.dimension()`."""
        return self._dimension

    def __call__(self, x):
        return self._value if self._boundaries.check(x) else 0

    def random_sample(self, samples=1):
        """See :methd:`Prior.randomSample()`,"""
        m_samples = np.zeros((num_samples, self._dimension))
        lower = self._boundaries.lower()
        upper = self._boundaries.upper()
        for i in range(0,self._dimension):
            m_samples[:,i] = np.random.uniform(low=lower[i], high=upper[i],
                size=samples)
        return m_samples


class MultivariateNormalPrior(Prior):
    """
    *Extends:* :class:`Prior`

    Defines a multivariate normal prior with a given mean and covariance
    matrix.

    For example::

        p = NormalPrior(np.array([0,0]),
                        np.array([[1, 0],[0, 1]]))`

    """
    def __init__(self, mean, cov):
        # Parse input arguments
        if not isinstance(mean, np.array):
            raise ValueError(
                'NormalPrior mean argument requires a NumPy array')

        if not isinstance(cov, np.array):
            raise ValueError('NormalPrior cov argument requires a NumPy array')

        if mean.ndim != 1:
            raise ValueError('NormalPrior mean must be one dimensional')

        if cov.ndim != 2:
            raise ValueError('NormalPrior cov must be a matrix')

        if mean.shape[0] != cov.shape[0] or mean.shape[0] != cov.shape[1]:
            raise ValueError('mean and cov sizes do not match')

        self._mean = mean
        self._cov = cov
        self._dimension = mean.shape[0]

    def dimension(self):
        """See :meth:`Prior.dimension()`."""
        return self._dimension

    def __call__(self, x):
        return scipy.stats.multivariate_normal.pdf(
            x, mean=self._mean, cov=self._cov)


class NormalPrior(Prior):
    """
    *Extends:* :class:`Prior`

    Defines a 1-d normal prior with a given mean and variance

    For example: ``p = NormalPrior(0,1)`` for a mean of ``0`` and variance
    of ``1``.
    """
    def __init__(self, mean, sigma):
        # Parse input arguments
        self._mean = mean
        self._sigma = sigma
        self._dimension = 1

        # Cache constants
        self._inv2cov = -1.0 / (2 * sigma ** 2)
        self._scale = 1 / np.sqrt(2 * np.pi * sigma ** 2)

    def dimension(self):
        """See :meth:`Prior.dimension()`."""
        return 1

    def __call__(self, x):
        return self._scale * np.exp(self._inv2cov * (x[0] - self._mean)**2)

    def random_sample(self, samples=1):
        """See :meth:`Prior.random_sample()`."""
        m_samples = np.zeros((samples, self._dimension))
        mu = self._mean
        sigma = self._sigma
        for i in range(0,self._dimension):
            m_samples[:, i] = np.random.normal(loc=mu, scale=sigma,
                size=samples)
        return m_samples


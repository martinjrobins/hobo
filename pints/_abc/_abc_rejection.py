#
# ABC Rejection method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np

import pints


class RejectionABC(pints.ABCSampler):
    """
    Rejection ABC sampler.

    See, for example, [1]_. In each iteration of the algorithm, the following
    steps occur::

        theta* ~ p(theta), i.e. sample parameters from prior distribution
        x ~ p(x|theta*), i.e. sample data from sampling distribution
        if s(x) < threshold:
            theta* added to list of samples

    References
    ----------
    .. [1] "Approximate Bayesian Computation (ABC) in practice". Katalin
           Csillery, Michael G.B.Blum, Oscar E. Gaggiotti, Olivier Francois
           (2010) Trends in Ecology & Evolution
           https://doi.org/10.1016/j.tree.2010.04.001
    """
    def __init__(self, log_prior):

        self._log_prior = log_prior
        self._threshold = 1
        self._xs = None
        self._ready_for_tell = False

    def name(self):
        """ See :meth:`pints.ABCSampler.name()`. """
        return 'Rejection ABC'

    def ask(self, n_samples):
        """ See :meth:`ABCSampler.ask()`. """
        if self._ready_for_tell:
            raise RuntimeError('ask called before tell.')
        self._xs = self._log_prior.sample(n_samples)

        self._ready_for_tell = True
        return self._xs

    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('tell called before ask.')
        self._ready_for_tell = False
        if isinstance(fx, list):
            accepted = [a < self._threshold for a in fx]
            if np.sum(accepted) == 0:
                return None
            else:
                return [self._xs[c].tolist() for c, x in
                        enumerate(accepted) if x]
        else:
            if fx < self._threshold:
                return self._xs
            else:
                return None

    def threshold(self):
        """
        Returns threshold error distance that determines if a sample is
        accepted (is error < threshold).
        """
        return self._threshold

    def set_threshold(self, threshold):
        """
        Sets threshold error distance that determines if a sample is accepted
        (``if error < threshold``).
        """
        x = float(threshold)
        if x <= 0:
            raise ValueError('Threshold must be positive.')
        self._threshold = threshold


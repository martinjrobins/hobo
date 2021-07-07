#
# Nested rejection sampler implementation.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints


class NestedRejectionSampler(pints.NestedSampler):
    """
    Creates a nested sampler that estimates the marginal likelihood and
    generates samples from the posterior.

    This is the simplest form of nested sampler and involves using rejection
    sampling from the prior as described in the algorithm on page 839 in [1]_
    to estimate the marginal likelihood and generate weights, preliminary
    samples (with their respective likelihoods), required to generate posterior
    samples.

    The posterior samples are generated as described in [1]_ on page 849 by
    randomly sampling the preliminary point, accounting for their weights and
    likelihoods.

    Initialise::

        Z = 0
        X_0 = 1

    Draw samples from prior::

        for i in 1:n_active_points:
            theta_i ~ p(theta), i.e. sample from the prior
            L_i = p(theta_i|X)
        endfor

    In each iteration of the algorithm (t)::

        L_min = min(L)
        indexmin = min_index(L)
        X_t = exp(-t / n_active_points)
        w_t = X_t - X_t-1
        Z = Z + L_min * w_t
        theta* ~ p(theta)
        while p(theta*|X) < L_min:
            theta* ~ p(theta)
        endwhile
        theta_indexmin = theta*
        L_indexmin = p(theta*|X)

    At the end of iterations, there is a final ``Z`` increment::

        Z = Z + (1 / n_active_points) * (L_1 + L_2 + ..., + L_n_active_points)

    The posterior samples are generated as described in [1]_ on page 849 by
    weighting each dropped sample in proportion to the volume of the
    posterior region it was sampled from. That is, the probability
    for drawing a given sample j is given by::

        p_j = L_j * w_j / Z

    where j = 1, ..., n_iterations.

    Extends :class:`NestedSampler`.

    References
    ----------
    .. [1] "Nested Sampling for General Bayesian Computation", John Skilling,
           Bayesian Analysis 1:4 (2006).
           https://doi.org/10.1214/06-BA127
    """
    def __init__(self, log_prior):
        super(NestedRejectionSampler, self).__init__(log_prior)

        self._needs_sensitivities = False

    def ask(self, n_points):
        """
        Proposes new point(s) by sampling from the prior.
        """
        if n_points > 1:
            self._proposed = self._log_prior.sample(n_points)
        else:
            self._proposed = self._log_prior.sample(n_points)[0]
        return self._proposed

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def set_hyper_parameters(self, x):
        """
        Hyper-parameter vector is: ``[active_points_rate]``

        Parameters
        ----------
        x
            An array of length ``n_hyper_parameters`` used to set the
            hyper-parameters
        """
        self.set_n_active_points(x[0])

    def name(self):
        """ See :meth:`pints.NestedSampler.name()`. """
        return 'Nested rejection sampler'

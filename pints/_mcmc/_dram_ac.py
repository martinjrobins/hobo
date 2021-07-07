#
# DRAM AC MC method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class DramACMC(pints.AdaptiveCovarianceMC):
    """
    DRAM (Delayed Rejection Adaptive Covariance) MCMC, as described in [1]_.

    In this method, rejections do not necessarily lead an iteration to end.
    Instead, if a rejection occurs, another point is proposed although
    typically from a narrower (i.e. more conservative) proposal kernel than was
    used for the first proposal.

    In this approach, in each iteration, the following steps return the next
    state of the Markov chain (assuming the current state is ``theta_0`` and
    that there are 2 proposal kernels)::

        theta_1 ~ N(theta_0, lambda * scale_1 * sigma)
        alpha_1(theta_0, theta_1) = min(1, p(theta_1|X) / p(theta_0|X))
        u_1 ~ uniform(0, 1)
        if alpha_1(theta_0, theta_1) > u_1:
            return theta_1
        theta_2 ~ N(theta_0, lambda * scale_2 * sigma0)
        alpha_2(theta_0, theta_1, theta_2) =
            min(1, p(theta_2|X) (1 - alpha_1(theta_2, theta_1)) /
                   (p(theta_0|X) (1 - alpha_1(theta_0, theta_1))))
        u_2 ~ uniform(0, 1)
        if alpha_2(theta_0, theta_1, theta_2) > u_2:
            return theta_2
        else:
            return theta_0

    At the end of each iterations, a 'base' proposal kernel is adapted::

        mu = (1 - gamma) mu + gamma theta
        sigma = (1 - gamma) sigma + gamma (theta - mu)(theta - mu)^t
        log_lambda = log_lambda + gamma (accepted - target_acceptance_rate)

    where ``gamma = adaptations^-eta``, ``theta`` is the current state of
    the Markov chain and ``accepted`` is a binary indicator for whether any of
    the series of proposals were accepted. The kernels for the 2 proposals
    are then adapted as ``[scale_1, scale_2] * sigma``, where the
    scale factors are set using ``set_sigma_scale``.

    *Extends:* :class:`GlobalAdaptiveCovarianceMC`

    References
    ----------
    .. [1] "DRAM: Efficient adaptive MCMC".
           H Haario, M Laine, A Mira, E Saksman, (2006) Statistical Computing
           https://doi.org/10.1007/s11222-006-9438-0
    """
    def __init__(self, x0, sigma0=None):
        super(DramACMC, self).__init__(x0, sigma0)

        self._log_lambda = 0
        self._n_kernels = 2
        self._proposal_count = 0
        self._sigma_base = np.copy(self._sigma)
        self._Y_log_pdf = np.zeros(self._n_kernels)
        self.set_sigma_scale([1, 0.01])  # scale used in [1]_ for experiments

    def _adapt_sigma(self):
        """
        Updates the covariance matrices of the 2 kernels being used according
        to adaptive Metropolis routine.
        """
        dsigm = np.reshape(self._current - self._mu, (self._n_parameters, 1))
        self._sigma_base = ((1 - self._gamma) * self._sigma_base +
                            self._gamma * np.dot(dsigm, dsigm.T))
        self._sigma = [self._sigma_scale[i] * self._sigma_base
                       for i in range(self._n_kernels)]

    def _alpha_1_log(self, fx, fy1):
        """
        Calculates probability of acceptance in stage 1 of DRAM (eq. 1 in
        [1]_).
        """
        alpha_log = fy1 - fx
        return min(0, alpha_log)

    def _alpha_2_log(self, fx, fy1, fy2):
        """
        Calculates probability of acceptance in stage 1 of DRAM (eq. 2 in
        [1]_).
        """
        alpha_log = (fy2 - fx +
                     np.log1p(np.exp(self._alpha_1_log(fy2, fy1))) -
                     np.log1p(np.exp(self._alpha_1_log(fx, fy1))))
        return min(0, alpha_log)

    def _generate_proposal(self):
        """ See :meth:`AdaptiveCovarianceMC._generate_proposal()`. """
        proposed = np.random.multivariate_normal(
            self._current, np.exp(self._log_lambda) *
            self._sigma[self._proposal_count])
        return proposed

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Delayed Rejection Adaptive Metropolis (Dram) MCMC'

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 2

    def _r_log(self):
        """
        Calculates value of logged acceptance ratio (eq. 3 in [1]_).
        """
        if self._proposal_count == 0:
            r_log = self._alpha_1_log(
                self._current_log_pdf, self._Y_log_pdf[0])
        else:
            r_log = self._alpha_2_log(
                self._current_log_pdf, self._Y_log_pdf[0],
                self._Y_log_pdf[1]
            )
        return r_log

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[eta, sigma_scale]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_eta(x[0])
        self.set_sigma_scale(x[1])

    def set_sigma_scale(self, scales):
        """
        Set the scale of the mulipliers for the two proposal kernel
        covariance matrices. Must be of the form ``[scale_1, scale_2]``.
        """
        if len(scales) != 2:
            raise ValueError("Scales must be of length 2.")
        for scale in scales:
            if scale <= 0:
                raise ValueError("Scales must be positive.")
        self._sigma_scale = [scales[0], scales[1]]
        self._sigma = [self._sigma_scale[i] * self._sigma_base
                       for i in range(self._n_kernels)]

    def sigma_scale(self):
        """
        Returns scale factors used to multiply a base covariance matrix,
        resulting in proposal matrices for each accept-reject step.
        """
        return self._sigma_scale

    def tell(self, fx):
        """
        If first proposal, then accept with ordinary Metropolis probability; if
        a later proposal, use probability determined by [1]_.
        """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        if self._proposal_count == 0:
            self._iterations += 1

        # Ensure fx is a float
        fx = float(fx)
        self._Y_log_pdf[self._proposal_count] = fx

        # First point?
        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            self._current = self._proposed
            self._current_log_pdf = fx
            self._proposed = None

            # Return first point for chain
            return self._current, self._current_log_pdf, True

        # Check if the proposed point can be accepted
        accept = 0
        if np.isfinite(fx):
            r_log = self._r_log()
            u = np.log(np.random.uniform(0, 1))
            if u < r_log:
                accept = 1
                self._acceptance_count += 1
                self._current = self._proposed
                self._current_log_pdf = fx
                self._proposal_count = 0
                self._Y_log_pdf = np.zeros(self._n_kernels)
        self._proposed = None

        if accept == 0:
            if self._proposal_count < (self._n_kernels - 1):
                self._proposal_count += 1
                return None
            else:
                self._proposal_count = 0

        # Update mu, covariance matrix and log lambda
        self._acceptance_rate = self._acceptance_count / self._iterations
        self._gamma = self._adaptations**-self._eta
        self._adaptations += 1
        self._adapt_mu()
        self._adapt_sigma()
        self._log_lambda += (self._gamma *
                             (accept - self._target_acceptance))
        return self._current, self._current_log_pdf, accept != 0

#
# Adaptive covariance MCMC method
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class AdaptiveCovarianceMCMC(pints.SingleChainAdaptiveMCMC):
    """
    *Extends:* :class:`SingleChainAdaptiveMCMC`

    Adaptive covariance MCMC, as described in [1, 2].

    The algorithm starts out as basic MCMC, but after

    TODO

    a certain number of iterations

    TODO

    starts tuning the covariance matrix, so that the acceptance rate
    of the MCMC steps converges to a user specified rate.

    [1] Uncertainty and variability in models of the cardiac action potential:
    Can we build trustworthy models?
    Johnstone, Chang, Bardenet, de Boer, Gavaghan, Pathmanathan, Clayton,
    Mirams (2015) Journal of Molecular and Cellular Cardiology

    [2] An adaptive Metropolis algorithm
    Heikki Haario, Eero Saksman, and Johanna Tamminen (2001) Bernoulli
    """
    def __init__(self, x0, sigma0=None):
        super(AdaptiveCovarianceMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False
        self._ready_for_tell = False
        self._need_first_point = True

        # Default settings
        self.set_target_acceptance_rate()

    def acceptance_rate(self):
        """
        Returns the current (measured) acceptance rate.
        """
        return self._acceptance

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # After this method we're ready for tell()
        self._ready_for_tell = True

        # Need evaluation of first point?
        if self._need_first_point:
            return self._current

        # Propose new point
        # Note: Normal distribution is symmetric
        #  N(x|y, sigma) = N(y|x, sigma) so that we can drop the proposal
        #  distribution term from the acceptance criterion
        self._proposed = np.random.multivariate_normal(
            self._current, np.exp(self._loga) * self._sigma)

        # Return proposed point
        self._proposed.setflags(write=False)
        return self._proposed

    def _initialise(self):
        """
        Initialises the routine before the first iteration.
        """
        if self._running:
            raise Exception('Already initialised.')

        # Set initial mu and sigma
        self._mu = np.array(self._x0, copy=True)
        self._sigma = np.array(self._sigma0, copy=True)

        # Set current sample
        self._current = self._x0
        self._current_log_pdf = float('inf')

        # Iteration counts (for acceptance rate)
        self._iterations = 0
        self._adaptations = 2

        # Initial acceptance rate
        self._loga = 0
        self._acceptance = 0

        # Update sampler state
        self._running = True

    def name(self):
        """See: :meth:`pints.MCMCSampler.name()`."""
        return 'Adaptive covariance MCMC'

    def tell(self, fx):
        """See: :meth:`pints.SingleChainMCMC.tell()`."""

        # First point?
        if self._need_first_point:
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')
            self._current_log_pdf = fx

            # Increase iteration count
            self._iterations += 1

            # Update state
            self._need_first_point = False
            self._ready_for_tell = False

            # Return first point for chain
            return self._current

        # Check if the proposed point can be accepted
        accepted = 0
        if np.isfinite(fx):
            u = np.log(np.random.rand())
            if u < fx - self._current_log_pdf:
                accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx

        # Adapt covariance matrix
        if self._adaptation:
            # Set gamma based on number of adaptive iterations
            gamma = self._adaptations ** -0.6
            self._adaptations += 1

            # Update mu, log acceptance rate, and covariance matrix
            self._mu = (1 - gamma) * self._mu + gamma * self._current
            self._loga += gamma * (accepted - self._target_acceptance)
            dsigm = np.reshape(self._current - self._mu, (self._dimension, 1))
            self._sigma = (
                (1 - gamma) * self._sigma + gamma * np.dot(dsigm, dsigm.T))

        # Update acceptance rate (only used for output!)
        self._acceptance = ((self._iterations * self._acceptance + accepted) /
                            (self._iterations + 1))

        # Increase iteration count
        self._iterations += 1

        # Update state
        self._ready_for_tell = False

        # Return new point for chain
        return self._current

    def set_target_acceptance_rate(self, rate=0.3):
        """
        Sets the target acceptance rate.
        """
        rate = float(rate)
        if rate <= 0:
            raise ValueError('Target acceptance rate must be greater than 0.')
        elif rate > 1:
            raise ValueError('Target acceptance rate cannot exceed 1.')
        self._target_acceptance = rate

    def target_acceptance_rate(self):
        """
        Returns the target acceptance rate.
        """
        return self._target_acceptance_rate


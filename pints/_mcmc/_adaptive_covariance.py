#
# Adaptive covariance MCMC method
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


class AdaptiveCovarianceMCMC(pints.SingleChainMCMC):
    """
    Adaptive covariance MCMC [1]_, [2]_.

    Using a covariance matrix, that is tuned so that the acceptance rate of the
    MCMC steps converges to a user specified value.

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] Johnstone, Chang, Bardenet, de Boer, Gavaghan, Pathmanathan,
           Clayton, Mirams (2015) "Uncertainty and variability in models of the
           cardiac action potential: Can we build trustworthy models?"
           Journal of Molecular and Cellular Cardiology.
           https://10.1016/j.yjmcc.2015.11.018

    .. [2] Haario, Saksman, Tamminen (2001) "An adaptive Metropolis algorithm"
           Bernoulli.
           https://doi.org/10.2307/3318737
    """
    def __init__(self, x0, sigma0=None):
        super(AdaptiveCovarianceMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False

        # Current point and proposed point
        self._current = None
        self._current_log_pdf = None
        self._proposed = None

        # Default settings
        self.set_target_acceptance_rate()

        # Adaptive mode: disabled during initial phase
        self._adaptive = False

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

        # Propose new point
        if self._proposed is None:

            # Note: Gaussian distribution is symmetric
            #  N(x|y, sigma) = N(y|x, sigma) so that we can drop the proposal
            #  distribution term from the acceptance criterion
            self._proposed = np.random.multivariate_normal(
                self._current, np.exp(self._loga) * self._sigma)

            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return self._current_log_pdf

    def _initialise(self):
        """
        Initialises the routine before the first iteration.
        """
        if self._running:
            raise RuntimeError('Already initialised.')

        # Propose x0 as first point
        self._current = None
        self._current_log_pdf = None
        self._proposed = self._x0

        # Set initial mu and sigma
        self._mu = np.array(self._x0, copy=True)
        self._sigma = np.array(self._sigma0, copy=True)

        # Adaptation
        self._loga = 0
        self._adaptations = 2

        # Acceptance rate monitoring
        self._iterations = 0
        self._acceptance = 0

        # Update sampler state
        self._running = True

    def in_initial_phase(self):
        """ See :meth:`pints.MCMCSampler.in_initial_phase()`. """
        return not self._adaptive

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._acceptance)

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Adaptive covariance MCMC'

    def needs_initial_phase(self):
        """ See :meth:`pints.MCMCSampler.needs_initial_phase()`. """
        return True

    def set_initial_phase(self, initial_phase):
        """ See :meth:`pints.MCMCSampler.set_initial_phase()`. """
        # No adaptation during initial phase
        self._adaptive = not bool(initial_phase)

    def tell(self, fx):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure fx is a float
        fx = float(fx)

        # First point?
        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Accept
            self._current = self._proposed
            self._current_log_pdf = fx

            # Increase iteration count
            self._iterations += 1

            # Clear proposal
            self._proposed = None

            # Return first point for chain
            return self._current

        # Check if the proposed point can be accepted
        accepted = 0
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < fx - self._current_log_pdf:
                accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx

        # Clear proposal
        self._proposed = None

        # Adapt covariance matrix
        if self._adaptive:
            # Set gamma based on number of adaptive iterations
            gamma = self._adaptations ** -0.6
            self._adaptations += 1

            # Update mu, log acceptance rate, and covariance matrix
            self._mu = (1 - gamma) * self._mu + gamma * self._current
            self._loga += gamma * (accepted - self._target_acceptance)
            dsigm = np.reshape(
                self._current - self._mu, (self._n_parameters, 1))
            self._sigma = (
                (1 - gamma) * self._sigma + gamma * np.dot(dsigm, dsigm.T))

        # Update acceptance rate (only used for output!)
        self._acceptance = ((self._iterations * self._acceptance + accepted) /
                            (self._iterations + 1))

        # Increase iteration count
        self._iterations += 1

        # Return new point for chain
        return self._current

    def replace(self, current, current_log_pdf, proposed=None):
        """ See :meth:`pints.SingleChainMCMC.replace()`. """

        # At least one round of ask-and-tell must have been run
        if (not self._running) or self._current_log_pdf is None:
            raise RuntimeError(
                'Replace can only be used when already running.')

        # Check values
        current = pints.vector(current)
        if len(current) != self._n_parameters:
            raise ValueError('Point `current` has the wrong dimensions.')
        current_log_pdf = float(current_log_pdf)
        if proposed is not None:
            proposed = pints.vector(proposed)
            if len(proposed) != self._n_parameters:
                raise ValueError('Point `proposed` has the wrong dimensions.')

        # Store
        self._current = current
        self._current_log_pdf = current_log_pdf
        if proposed is not None:
            self._proposed = proposed

    def set_target_acceptance_rate(self, rate=0.234):
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
        return self._target_acceptance


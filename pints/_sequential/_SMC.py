#
# Sequential Monte Carlo following Del Moral et al. 2006
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
from scipy.special import logsumexp


class SMC(pints.SMCSampler):
    """
    Samples from a density using sequential Monte Carlo sampling [1], although
    allows multiple MCMC steps per temperature, if desired.

    Algorithm 3.1.1 using equation (31) for ``w_tilde``.

    [1] "Sequential Monte Carlo Samplers", Del Moral et al. 2006,
    Journal of the Royal Statistical Society. Series B.
    """
    def __init__(self, log_prior, sigma0=None):
        super(SMC, self).__init__(log_prior, sigma0)

        # ESS threshold (default from Del Moral et al.)
        self._ess_threshold = None

        # Keep track of last ess
        self._last_ess = None

        # Determines whether to resample particles at end of steps 2 and 3 from
        # Del Moral et al. (2006)
        self._resample_end_2_3 = False

        # Current samples, their log pdfs, and weights
        self._samples = None
        self._log_pdfs = None
        self._samples_previous = None
        self._log_pdfs_previous = None
        self._weights = None

        # Proposed samples
        self._proposals = None

        # Internal mcmc chain: Uses only 1 chain for all samples!
        # - Must be a SingleChainMCMC method
        # - Must implement replace()
        self._method = pints.AdaptiveCovarianceMCMC
        self._chain = None

        # Iterations:
        #   i_temp (outer loop)
        #       i_mcmc (inner loop)
        #           i_batch (indice of first sample to update)
        self._i_temp = 1
        self._i_mcmc = 0
        self._i_batch = 0

        # Status
        self._expecting_ask = True
        self._finished = False

    def ask(self):
        """ See :meth:`SMCSampler.ask()`. """

        # Check ask/tell pattern
        if not self._expecting_ask:
            raise RuntimeError('Ask called when expecting tell.')
        self._expecting_ask = False

        # Too many steps?
        if self._i_temp >= len(self._schedule):
            raise RuntimeError(
                'SMC.ask() called after the maximum number of iterations was'
                ' reached.')

        # We're up and running now
        self._running = True

        # Initialise
        if self._samples is None:

            # Set default ESS if not user specified
            if self._ess_threshold is None:
                self._ess_threshold = self._n_particles / 2
            elif self._ess_threshold > self._n_particles:
                raise RuntimeError(
                    'ESS threshold must be lower than or equal to number of'
                    ' particles. Set to ' + str(self._ess_threshold) +
                    ', expecting <= ' + str(self._n_particles))

            # Sample from the prior
            self._proposals = self._log_prior.sample(self._n_particles)
            self._proposals.setflags(write=False)

            # Create and configure chain
            x0 = self._proposals[0]
            #TODO: COME UP WITH BETTER x0
            self._chain = self._method(x0, self._sigma0)
            # TODO: Ignoring initial phase for now
            if self._chain.needs_initial_phase():
                self._chain.set_initial_phase(False)

            # Call ask(), so that chain asks for f(initial value)
            self._chain.ask()

            # Get LogPDFs of all initial samples via ask/tell
            return self._proposals

        # Get beta, using next temperature
        beta = self._schedule[self._i_temp]

        # Start of an MCMC iteration?
        if self._i_batch == 0:
            # If ESS < threshold then resample to avoid degeneracies
            if self._last_ess < self._ess_threshold:
                print('Resampling ess threshold')
                self._resample()

            # Create new proposal
            self._proposals = np.zeros((self._n_particles, self._n_parameters))

            # Store previous samples and log pdfs
            self._samples_previous = np.copy(self._samples)
            self._log_pdfs_previous = np.copy(self._log_pdfs)

        # Set the proposals in batches
        lo = self._i_batch
        hi = min(lo + self._n_batch, self._n_particles)
        for i in range(lo, hi):
            # Repeatedly set chain to proposed point and its tempered PDF, get
            # new proposals
            self._chain.replace(self._samples[i], self._temper(
                self._log_pdfs[i], self._log_prior(self._samples[i]), beta))
            self._proposals[i] = self._chain.ask()

        batch = self._proposals[lo:hi]
        batch.setflags(write=False)
        return batch

    def ess(self):
        """
        Returns ess from last run of SMC.
        """
        return self._last_ess

    def name(self):
        """ See :meth:`SMCSampler.name()`. """
        return 'Sequential Monte Carlo'

    def tell(self, log_pdfs):
        """ See :meth:`SMCSampler.ask()`. """

        # Check ask/tell pattern
        if self._expecting_ask:
            raise RuntimeError('Tell called when expecting ask.')
        self._expecting_ask = True

        # First step?
        if self._samples is None:

            # Check returned size
            if len(log_pdfs) != len(self._proposals):
                raise ValueError(
                    'Number of evaluations passed to tell() does not match'
                    ' number requested by ask().')

            # Store current samples and logpdfs
            # (Copy proposals to remove read-only property - can't just unset
            #  flags as user still has reference too)
            self._samples = np.copy(self._proposals)
            self._log_pdfs = np.array(log_pdfs, copy=True)

            # Get next temperature
            beta = self._schedule[1]

            # Tell chain evaluation of its initial value
            initial_tempered = self._temper(
                log_pdfs[0], self._log_prior(self._samples[0]), beta)
            self._chain.tell(initial_tempered)

            # Set weights based on next temperature
            priors = np.array([self._log_prior(x) for x in self._samples])
            self._weights = beta * (self._log_pdfs - priors)
            self._weights = np.exp(self._weights - logsumexp(self._weights))

            # Update ess
            self._last_ess = 1 / np.sum(self._weights**2)

            # Store copy of proposals to return
            to_return = self._proposals

            # Clear proposals
            self._proposals = None

            # Return current samples
            return to_return

        # Normal iteration
        beta = self._schedule[self._i_temp]

        # Update batch of samples
        lo = self._i_batch
        hi = min(lo + self._n_batch, self._n_particles)
        if hi - lo != len(log_pdfs):
            raise ValueError(
                'Number of evaluations passed to tell() does not match number'
                ' requested by ask().')

        for i in range(lo, hi):
            # Update chain to (sample[i], f(sample)[i], proposal[i]), then call
            # tell() with tempered log pdf values
            proposed = self._proposals[i]

            # `Replace' chain position
            current_tempered = self._temper(
                self._log_pdfs[i], self._log_prior(self._samples[i]), beta)
            self._chain.replace(self._samples[i], current_tempered, proposed)

            # Tell, get updated sample
            proposed_tempered = self._temper(
                log_pdfs[i - lo], self._log_prior(proposed), beta)
            self._samples[i] = self._chain.tell(proposed_tempered)

            # Update log pdf (to untempered value)
            if np.all(self._samples[i] == proposed):  # TODO: use accepted()
                self._log_pdfs[i] = log_pdfs[i - lo]

        # Update i_batch
        self._i_batch += self._n_batch
        if self._i_batch < self._n_particles:
            return None
        self._i_batch = 0

        # Clear proposals
        self._proposals = None

        # Run more mcmc steps before continuing?
        self._i_mcmc += 1
        if self._i_mcmc < self._n_mcmc_steps:
            return None

        # Reset mcmc steps
        self._i_mcmc = 0

        # Update weights
        for j, w in enumerate(self._weights):
            self._weights[j] = np.log(w) + self._w_tilde(
                self._log_pdfs_previous[j],
                self._log_prior(self._samples_previous[j]),
                self._schedule[self._i_temp - 1],
                self._schedule[self._i_temp])
        self._weights = np.exp(self._weights - logsumexp(self._weights))

        # Store ess, before resampling
        self._last_ess = 1 / np.sum(self._weights**2)

        # Update temperature step
        self._i_temp += 1

        # Conditional resampling step
        if self._resample_end_2_3 and self._i_temp < len(self._schedule):
            print('Resampling end 2/3')
            self._resample()

        # Return copy of current samples
        return np.copy(self._samples)

    def set_ess_threshold(self, ess_threshold):
        """
        Sets the threshold effective sample size (ESS).
        Use ``None`` to reset it to a default value.
        """
        if ess_threshold is None:
            self._ess_threshold = None
            return

        ess_threshold = int(ess_threshold)
        if ess_threshold <= 0:
            raise ValueError('ESS must be greater than zero.')
        if ess_threshold > self._n_particles:
            raise ValueError(
                'ESS threshold must be lower than or equal to number of'
                ' particles.')
        self._ess_threshold = ess_threshold

    def set_resample_end_2_3(self, resample_end_2_3):
        """
        Determines whether a resampling step is performed at end of steps 2 and
        3 in Del Moral et al. Algorithm 3.1.1.
        """
        self._resample_end_2_3 = bool(resample_end_2_3)

    def weights(self):
        """
        Returns weights from last run of SMC.
        """
        if self._weights is None:
            return None
        return np.copy(self._weights)

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Temperature')
        logger.add_float('ESS')
        logger.add_float('Acc.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        # Called after tell() has updated i_temp!
        logger.log(1 - self._schedule[self._i_temp - 1])
        logger.log(self._last_ess)
        logger.log(self._chain.acceptance_rate())

    def _resample(self, update_weights=True):
        """
        Resamples (and updates the weights and log_pdfs) according to the
        weights vector from the multinomial distribution.
        """
        selected = np.random.multinomial(self._n_particles, self._weights)
        new_samples = np.zeros((self._n_particles, self._n_parameters))
        new_log_pdfs = np.zeros(self._n_particles)
        lo = hi = 0
        for i, n_selected in enumerate(selected):
            if n_selected:
                hi += n_selected
                new_samples[lo:hi, :] = self._samples[i]
                new_log_pdfs[lo:hi] = self._log_pdfs[i]
                lo = hi
        self._samples = new_samples
        self._log_pdfs = new_log_pdfs

        # Update weights
        self._weights = np.repeat(1 / self._n_particles, self._n_particles)

    def _temper(self, fx, f_prior, beta):
        """
        Returns beta * fx + (1-beta) * f_prior
        """
        return beta * fx + (1 - beta) * f_prior

    def _w_tilde(self, fx_old, f_prior_old, beta_old, beta_new):
        """
        Calculates the log unnormalised incremental weight as per eq. (31) in
        Del Moral.
        """
        return (
            self._temper(fx_old, f_prior_old, beta_new)
            - self._temper(fx_old, f_prior_old, beta_old)
        )


#
# Differential evolution MCMCs
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


class DifferentialEvolutionMCMC(pints.MCMC):
    """
    *Extends:* :class:`MCMC`

    Uses differential evolution MCMC as described in [1]
    to do posterior sampling from the posterior.

    In each step of the algorithm N chains are evolved
    using the evolution equation,

    x_proposed = x[i,r] + gamma * (X[i,r1] - x[i,r2]) + epsilon

    where r1 and r2 are random chain indices chosen (without
    replacement) from the N available chains, which must not
    equal i or each other, where i indicates the current time
    step. epsilon ~ N(0,b) in d dimensions (where d is the
    dimensionality of the parameter vector).

    If x_proposed / x[i,r] > u ~ U(0,1), then
    x[i+1,r] = x_proposed; otherwise, x[i+1,r] = x[i].

    [1] "A Markov Chain Monte Carlo version of the genetic
    algorithm Differential Evolution: easy Bayesian computing
    for real parameter spaces", 2006, Cajo J. F. Ter Braak,
    Statistical Computing.
    """
    def __init__(self, log_pdf, x0, sigma0=None):
        super(DifferentialEvolutionMCMC, self).__init__(log_pdf, x0, sigma0)

        # Total number of iterations
        self._iterations = self._dimension * 2000

        # Gamma
        self._gamma = 2.38 / np.sqrt(2 * self._dimension)

        # Normal proposal std.
        self._b = 0.01

        # Number of chains to evolve
        self._num_chains = 100

        # Number of iterations to discard as warm-up
        self._warm_up = int(0.5 * self._iterations)

        # Thinning: Store only one sample per X
        self._thinning_rate = 1

    def iterations(self):
        """
        Returns the total number of iterations that will be performed in the
        next run, including the non-adaptive and warm-up iterations.
        """
        return self._iterations

    def run(self):
        """See: :meth:`pints.MCMC.run()`."""

        # Report the current settings
        if self._verbose:
            print('Running differential evolution MCMC')
            print('gamma = ' + str(self._gamma))
            print('normal proposal std. = ' + str(self._b))
            print('Total number of iterations: ' + str(self._iterations))
            print(
                'Number of iterations to discard as warm-up: '
                + str(self._warm_up))
            print('Storing one sample per ' + str(self._thinning_rate))

        # Initial starting parameters
        mu = self._x0
        current = self._x0
        current_log_pdf = self._log_pdf(current)
        if not np.isfinite(current_log_pdf):
            raise ValueError(
                'Suggested starting position has a non-finite log-pdf.')

        # chains of stored samples
        chains = np.zeros(
            (self._iterations, self._num_chains, self._dimension))
        current_log_pdf = np.zeros(self._num_chains)

        # Set initial values
        for j in range(self._num_chains):
            chains[0, j, :] = np.random.normal(
                loc=mu, scale=mu / 100, size=len(mu))
            current_log_pdf[j] = self._log_pdf(chains[0, j, :])

        # Go!
        for i in range(1, self._iterations):
            for j in range(self._num_chains):
                r1, r2 = R_draw(j, self._num_chains)
                proposed = (
                    chains[i - 1, j, :]
                    + self._gamma * (
                        chains[i - 1, r1, :] - chains[i - 1, r2, :])
                    + np.random.normal(loc=0, scale=self._b * mu, size=len(mu))
                )
                u = np.log(np.random.rand())
                proposed_log_pdf = self._log_pdf(proposed)

                if u < proposed_log_pdf - current_log_pdf[j]:
                    chains[i, j, :] = proposed
                    current_log_pdf[j] = proposed_log_pdf
                else:
                    chains[i, j, :] = chains[i - 1, j, :]

            # Report
            if self._verbose and i % 50 == 0:
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))
                print('  In warm-up: ' + str(i < self._warm_up))

        non_warm_up = self._iterations - self._warm_up
        chains = chains[non_warm_up:, :, :]
        chains = chains[::self._thinning_rate, :, :]

        # Convert 3d array to list of lists
        samples = [chains[:, i, :] for i in range(0, self._num_chains)]

        # Return generated chain
        return samples

    def set_b(self, b):
        """
        Sets the normal scale coefficient used in updating the position of each
        chain.
        """
        if b < 0:
            raise ValueError('normal scale coefficient must be non-negative.')
        self._b = b

    def set_gamma(self, gamma):
        """
        Sets the gamma coefficient used in updating the position of each
        chain.
        """
        if gamma < 0:
            raise ValueError('Gamma must be non-negative.')
        self._gamma = gamma

    def set_iterations(self, iterations):
        """
        Sets the total number of iterations to be performed in the next run
        (including warm-up and non-adaptive iterations).
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

    def set_num_chains(self, num_chains):
        """
        Sets the number of chains to evolve
        """
        if num_chains < 10:
            raise ValueError('This method works best with many chains (>>10,'
                             + 'typically).')
        self._num_chains = num_chains

    def set_thinning_rate(self, thinning):
        """
        Sets the thinning rate. With a thinning rate of *n*, only every *n-th*
        sample will be stored.
        """
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError('Thinning rate must be greater than zero.')
        self._thinning_rate = thinning

    def set_warm_up(self, warm_up):
        """
        Sets the number of iterations to discard as warm-up (burn-in) in the
        next run.
        """
        warm_up = int(warm_up)
        if warm_up < 0:
            raise ValueError('Warm-up cannot be negative.')
        self._warm_up = warm_up

    def thinning_rate(self):
        """
        Returns the thinning rate that will be used in the next run. A thinning
        rate of *n* indicates that only every *n-th* sample will be stored.
        """
        return self._thinning_rate

    def warm_up(self):
        """
        Returns the number of iterations that will be discarded as warm-up
        (burn-in) in the next run.
        """
        return self._warm_up


def differential_evolution_mcmc(log_pdf, x0, sigma0=None):
    """
    Runs an differential evolution MCMC routine with the default parameters.
    """
    return DifferentialEvolutionMCMC(log_pdf, x0, sigma0).run()


class DreamMCMC(pints.MCMC):
    """
    *Extends:* :class:`MCMC`

    Uses differential evolution adaptive Metropolis (DREAM) MCMC as described
    in [1] to do posterior sampling from the posterior.

    In each step of the algorithm N chains are evolved using the following
    steps:

    1. Select proposal::

        x_proposed = x[i,r] + (1 + e) * gamma(delta, d, p_g) *
                     sum_j=1^delta (X[i,r1[j]] - x[i,r2[j]])
                     + epsilon

    where [r1[j], r2[j]] are random chain indices chosen (without replacement)
    from the ``N`` available chains, which must not equal each other or ``i``,
    where ``i`` indicates the current time step;
    ``delta ~ uniform_discrete(1,D)`` determines the number of terms to include
    in the summation::

        e ~ U(-b*, b*) in d dimensions;
        gamma(delta, d, p_g) =:
          if p_g < u1 ~ U(0,1):
            2.38 / sqrt(2 * delta * d)
          else:
            1

    ``epsilon ~ N(0,b)`` in ``d`` dimensions (where ``d`` is the dimensionality
    of the parameter vector).

    2. Modify random subsets of the proposal according to a crossover
    probability CR::

        for j in 1:N:
          if 1 - CR > u2 ~ U(0,1):
            x_proposed[j] = x[j],
          else:
            x_proposed[j] = x_proposed[j] from 1.

    If ``x_proposed / x[i,r] > u ~ U(0,1)``, then
    ``x[i+1,r] = x_proposed``; otherwise, ``x[i+1,r] = x[i]``.

    [1] "Accelerating Markov Chain Monte Carlo Simulation by Differential
    Evolution with Self-Adaptive Randomized Subspace Sampling ",
    2009, Vrugt et al.,
    International Journal of Nonlinear Sciences and Numerical Simulation.
    """
    def __init__(self, log_pdf, x0, sigma0=None):
        super(DreamMCMC, self).__init__(log_pdf, x0, sigma0)

        # Total number of iterations
        self._iterations = self._dimension * 2000

        # Normal proposal std.
        self._b = 0.01

        # b* distribution for e ~ U(-b*, b*)
        self._b_star = 0.01

        # Probability of longer gamma versus regular
        self._p_g = 0.2

        # Determines maximum delta to choose in sums
        self._D = 3

        # Constant crossover probability boolean
        self._constant_crossover = False

        # Crossover probability for variable CR case
        self._nCR = 3

        # Constant CR probability for constant CR probability
        self._CR = 0.5

        # Number of chains to evolve
        self._num_chains = 100

        # Number of iterations to discard as warm-up
        self._warm_up = int(0.5 * self._iterations)

        # Thinning: Store only one sample per X
        self._thinning_rate = 1

    def iterations(self):
        """
        Returns the total number of iterations that will be performed in the
        next run, including the non-adaptive and warm-up iterations.
        """
        return self._iterations

    def run(self):
        """See: :meth:`pints.MCMC.run()`."""

        # Report the current settings
        if self._verbose:
            print('Running differential evolution MCMC')
            print('gamma = ' + str(self._gamma))
            print('normal proposal std. = ' + str(self._b))
            print('Total number of iterations: ' + str(self._iterations))
            print(
                'Number of iterations to discard as warm-up: '
                + str(self._warm_up))
            print('Storing one sample per ' + str(self._thinning_rate))

        # Initial starting parameters
        mu = self._x0
        current = self._x0
        current_log_pdf = self._log_pdf(current)
        if not np.isfinite(current_log_pdf):
            raise ValueError(
                'Suggested starting position has a non-finite log-pdf.')

        # chains of stored samples
        chains = np.zeros(
            (self._iterations, self._num_chains, self._dimension))
        current_log_pdf = np.zeros(self._num_chains)

        # Set initial values
        for j in range(self._num_chains):
            chains[0, j, :] = np.random.normal(
                loc=mu, scale=mu / 100, size=len(mu))
            current_log_pdf[j] = self._log_pdf(chains[0, j, :])

        # Go!
        p = np.repeat(1 / self._nCR, self._nCR)
        L = np.zeros(self._nCR)
        Delta = np.zeros(self._nCR)
        after_warm_up_indicator = 0
        if self._constant_crossover is False:
            for i in range(1, self._iterations):
                # Warm-up
                if i < self._warm_up:
                    for j in range(self._num_chains):
                        # Step 1. Select (initial) proposal
                        delta = int(np.random.choice(self._D, 1)[0] + 1)
                        dX = 0
                        u1 = np.random.rand()
                        if self._p_g < u1:
                            gamma = 2.38 / np.sqrt(2 * delta * self._dimension)
                        else:
                            gamma = 1.0
                        e = np.random.uniform(low=-self._b_star * mu,
                                              high=self._b_star * mu)
                        for k in range(0, delta):
                            r1, r2 = R_draw(j, self._num_chains)
                            dX += (1 + e) * gamma * (
                                chains[i - 1, r1, :] - chains[i - 1, r2, :])
                        proposed = (
                            chains[i - 1, j, :]
                            + dX
                            + np.random.normal(
                                loc=0, scale=self._b * mu, size=len(mu))
                        )

                        # Select CR from multinomial distribution
                        m = np.nonzero(np.random.multinomial(self._nCR, p))
                        m = m[0][0]
                        CR = float(m + 1) / float(self._nCR)
                        L[m] += 1

                        # Step 2. Randomly set elements of proposal to original
                        for d in range(0, self._dimension):
                            u2 = np.random.rand()
                            if 1.0 - CR > u2:
                                proposed[d] = chains[i - 1, j, d]

                        # Accept/reject
                        u = np.log(np.random.rand())
                        proposed_log_pdf = self._log_pdf(proposed)

                        if u < proposed_log_pdf - current_log_pdf[j]:
                            chains[i, j, :] = proposed
                            current_log_pdf[j] = proposed_log_pdf
                        else:
                            chains[i, j, :] = chains[i - 1, j, :]

                        # Update CR distribution
                        for d in range(0, self._dimension):
                            Delta[m] += (
                                (chains[i, j, d] - chains[i - 1, j, d])**2
                                / np.var(chains[:, j, d])
                            )
                    for k in range(0, self._nCR):
                        p[k] = (
                            i * self._num_chains * (Delta[k] / float(L[k]))
                            / np.sum(Delta))
                    p = p / np.sum(p)

                # After warm-up
                else:
                    if after_warm_up_indicator == 0:
                        if self._verbose:
                            print('Finished warm-up...starting sampling')
                            print('Crossover probabilities = ' + str(p))
                        after_warm_up_indicator = 1
                    for j in range(self._num_chains):
                        # Step 1. Select (initial) proposal
                        delta = int(np.random.choice(self._D, 1)[0] + 1)
                        dX = 0
                        u1 = np.random.rand()
                        if self._p_g < u1:
                            gamma = 2.38 / np.sqrt(2 * delta * self._dimension)
                        else:
                            gamma = 1.0
                        e = np.random.uniform(low=-self._b_star * mu,
                                              high=self._b_star * mu)
                        for k in range(0, delta):
                            r1, r2 = R_draw(j, self._num_chains)
                            dX += (1 + e) * gamma * (chains[i - 1, r1, :] -
                                                     chains[i - 1, r2, :])
                        proposed = (
                            chains[i - 1, j, :]
                            + dX
                            + np.random.normal(
                                loc=0, scale=self._b * mu, size=len(mu))
                        )

                        # Step 2. Randomly set elements of proposal to original
                        # Select CR from multinomial distribution using tuned p
                        m = np.nonzero(np.random.multinomial(self._nCR, p))
                        m = m[0][0]
                        CR = float(m + 1) / float(self._nCR)
                        for d in range(0, self._dimension):
                            u2 = np.random.rand()
                            if 1.0 - CR > u2:
                                proposed[d] = chains[i - 1, j, d]

                        # Accept/reject
                        u = np.log(np.random.rand())
                        proposed_log_pdf = self._log_pdf(proposed)

                        if u < proposed_log_pdf - current_log_pdf[j]:
                            chains[i, j, :] = proposed
                            current_log_pdf[j] = proposed_log_pdf
                        else:
                            chains[i, j, :] = chains[i - 1, j, :]
                    # Report
                    if self._verbose and i % 50 == 0:
                        print('Iteration ' + str(i) + ' of '
                              + str(self._iterations))
                        print('  In warm-up: ' + str(i < self._warm_up))

        # Constant crossover probability
        else:
            CR = self._CR
            for j in range(self._num_chains):
                # Step 1. Select (initial) proposal
                delta = int(np.random.choice(self._D, 1)[0] + 1)
                dX = 0
                u1 = np.random.rand()
                if self._p_g < u1:
                    gamma = 2.38 / np.sqrt(2 * delta * self._dimension)
                else:
                    gamma = 1.0
                e = np.random.uniform(low=-self._b_star * mu,
                                      high=self._b_star * mu)
                for k in range(0, delta):
                    r1, r2 = R_draw(j, self._num_chains)
                    dX += (1 + e) * gamma * (chains[i - 1, r1, :] -
                                             chains[i - 1, r2, :])
                proposed = chains[i - 1, j, :] + dX \
                           + np.random.normal(loc=0, scale=self._b * mu, size=len(mu))  # NOQA

                # Step 2. Randomly set elements of proposal to original
                for d in range(0, self._dimension):
                    u2 = np.random.rand()
                    if 1.0 - CR > u2:
                        proposed[d] = chains[i - 1, j, d]

                # Accept/reject
                u = np.log(np.random.rand())
                proposed_log_pdf = self._log_pdf(proposed)

                if u < proposed_log_pdf - current_log_pdf[j]:
                    chains[i, j, :] = proposed
                    current_log_pdf[j] = proposed_log_pdf
                else:
                    chains[i, j, :] = chains[i - 1, j, :]

                # Report
                if self._verbose and i % 50 == 0:
                    print('Iteration ' + str(i) + ' of ' + str(self._iterations))  # NOQA
                    print('  In warm-up: ' + str(i < self._warm_up))

        non_warm_up = self._iterations - self._warm_up
        chains = chains[non_warm_up:, :, :]
        chains = chains[::self._thinning_rate, :, :]

        # Convert 3d array to list of lists
        samples = [chains[:, i, :] for i in range(0, self._num_chains)]

        # Return generated chain
        return samples

    def set_b(self, b):
        """
        Sets the normal scale coefficient used in updating the position of each
        chain.
        """
        if b < 0:
            raise ValueError('normal scale coefficient must be non-negative.')
        self._b = b

    def set_gamma(self, gamma):
        """
        Sets the gamma coefficient used in updating the position of each
        chain.
        """
        if gamma < 0:
            raise ValueError('Gamma must be non-negative.')
        self._gamma = gamma

    def set_iterations(self, iterations):
        """
        Sets the total number of iterations to be performed in the next run
        (including warm-up and non-adaptive iterations).
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

    def set_num_chains(self, num_chains):
        """
        Sets the number of chains to evolve
        """
        if num_chains < 10:
            raise ValueError('This method works best with many chains (>>10,'
                             + 'typically).')
        self._num_chains = num_chains

    def set_thinning_rate(self, thinning):
        """
        Sets the thinning rate. With a thinning rate of *n*, only every *n-th*
        sample will be stored.
        """
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError('Thinning rate must be greater than zero.')
        self._thinning_rate = thinning

    def set_warm_up(self, warm_up):
        """
        Sets the number of iterations to discard as warm-up in the next run.
        """
        warm_up = int(warm_up)
        if warm_up < 0:
            raise ValueError('Warm-up cannot be negative.')
        self._warm_up = warm_up

    def thinning_rate(self):
        """
        Returns the thinning rate that will be used in the next run. A thinning
        rate of *n* indicates that only every *n-th* sample will be stored.
        """
        return self._thinning_rate

    def warm_up(self):
        """
        Returns the number of iterations that will be discarded as warm-up in
        the next run.
        """
        return self._warm_up


def R_draw(i, num_chains):
    r_both = np.random.choice(num_chains, 2, replace=False)
    r1 = r_both[0]
    r2 = r_both[1]
    while(r1 == i or r2 == i or r1 == r2):
        r_both = np.random.choice(num_chains, 2, replace=False)
        r1 = r_both[0]
        r2 = r_both[1]
    return r1, r2

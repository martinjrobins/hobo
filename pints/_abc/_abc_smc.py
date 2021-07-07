#
# ABC SMC method
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


class ABCSMC(pints.ABCSampler):
    """
    ABC-SMC Algorithm  See, for example, [1]_. In each iteration of the
    algorithm, the following steps occur::

        theta* ~ p_(t-1)(theta), i.e. sample parameters from previous
            intermediate distribution
        theta** ~ K(theta|theta*), i.e. perturb theta* to obtain to new point
        x ~ p(x|theta**), i.e. sample data from sampling distribution
        if s(x) < threshold_(t):
            theta* added to list of samples[t]

    After we have obtained n_target samples, t is advanced, and weights
    are calculated for samples[t-1]. At the last value for threshold,
    samples are returned whenever they are accepted.

    References
    ----------
    .. [1] "Sisson SA, Fan Y and Tanaka MM. Sequential Monte Carlo without
            likelihoods. Proc Natl Acad Sci USA, 104(6):1760-5, 2007."
    """
    def __init__(self, log_prior, perturbation_kernel=None):

        self._log_prior = log_prior
        self._samples = [[]]
        self._accepted_count = 0
        self._weights = []
        self._threshold = 1
        self._e_schedule = [1]
        self._n_target = 100
        self._xs = None
        self._ready_for_tell = False
        self._t = 0

        dim = log_prior.n_parameters()

        if perturbation_kernel is None:
            self._perturbation_kernel = pints.MultivariateGaussianLogPrior(
                np.zeros(dim),
                0.001 * np.identity(dim))
        elif isinstance(perturbation_kernel, pints.LogPrior):
            self._perturbation_kernel = perturbation_kernel
        else:
            raise ValueError("Provided perturbation kernel must be an instance\
                of pints.LogPrior")

    def name(self):
        """ See :meth:`pints.ABCSampler.name()`. """
        return 'ABC-SMC'

    def ask(self, n_samples):
        """ See :meth:`ABCSampler.ask()`. """
        if self._ready_for_tell:
            raise RuntimeError('ask called before tell.')
        if self._t == 0:
            self._xs = self._log_prior.sample(n_samples)
        else:
            self._xs = []

            while len(self._xs) < n_samples:
                theta_s_s = None  # to appease the linter
                while (theta_s_s is None or
                       self._log_prior(theta_s_s) == -np.inf):
                    indices = np.random.choice(
                        range(len(self._samples[self._t - 1])),
                        p=self._weights[self._t - 1])
                    theta_s = self._samples[self._t - 1][indices]
                    # perturb using _K_t TODO: Allow this to adapt e.g. OLCM
                    theta_s_s = np.add(theta_s,
                                       self._perturbation_kernel.sample(1)[0])
                    # check if theta_s_s is possible under the prior
                    # sample again if not
                self._xs.append(theta_s_s)
        self._ready_for_tell = True
        return self._xs

    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('tell called before ask.')
        self._ready_for_tell = False
        if isinstance(fx, list):
            accepted = [a < self._threshold for a in fx]
            if sum(accepted) > 0:
                if self._t == len(self._e_schedule) - 1:
                    return [self._xs[c].tolist() for c, x in
                            enumerate(accepted) if x]

                self._accepted_count += sum(accepted)
                self._samples[self._t].extend(
                    [self._xs[c].tolist() for c, x in enumerate(accepted) if x]
                )

                if self._accepted_count >= self._n_target:
                    self._advance_time()
            return None
        else:
            if fx < self._threshold:
                self._accepted_count += 1
                if self._t == len(self._e_schedule) - 1:
                    return self._xs
                self._samples[self._t].append(self._xs)
                if self._accepted_count >= self._n_target:
                    self._advance_time()
            return None

    def _advance_time(self):
        t = self._t
        if t == 0:
            self._weights.append(
                np.full(self._accepted_count, 1 / self._accepted_count))
        else:
            unnorm_weights = self._calculate_weights(
                self._samples[t], self._samples[t - 1], self._weights[t - 1])
            # Normalise weights
            normal = sum(unnorm_weights)
            self._weights.append([w / normal for w in unnorm_weights])

        self._samples.append([])
        self._accepted_count = 0
        self._t += 1
        self._threshold = self._e_schedule[self._t]
        print(
            "Trying t=" + str(self._t)
            + ", threshold=" + str(self._threshold))

    def _calculate_weights(self, new_samples, old_samples, old_weights):
        new_weights = []
        for i in range(0, self._accepted_count):
            # Calculate weights according to the Toni algorithm
            prior_prob = np.exp(self._log_prior(new_samples[i]))

            # Don't know what the technical name is for this (O(n^2))
            mw = [old_weights[j] * np.exp(self._perturbation_kernel(
                np.subtract(new_samples[i], old_samples[j])))
                for j in range(len(old_samples))]

            w = prior_prob / sum(mw)
            new_weights.append(w)
        return new_weights

    def set_threshold_schedule(self, schedule):
        """
        Sets a schedule for the threshold error distance that determines if a
        sample is accepted (if error < threshold).

        Schedule should be a list of epsilon values
        """
        e_schedule = np.array(schedule)
        if any(e_schedule <= 0):
            raise ValueError('All threshold values must be positive.')
        self._e_schedule = e_schedule
        self._threshold = self._e_schedule[self._t]

    def set_intermediate_size(self, n):
        """
        Sets the size of the intermediate distributions, after we find n
        acceptable samples then we will progress to the next threshold values
        in the schedule
        """
        self._n_target = n

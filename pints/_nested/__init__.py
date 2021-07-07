#
# Sub-module containing nested samplers
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy.special

try:
    from scipy.special import logsumexp
except ImportError:     # pragma: no cover
    # Older versions
    from scipy.misc import logsumexp


class NestedSampler(pints.TunableMethod):
    """
    Abstract base class for nested samplers.

    Parameters
    ----------
    log_prior : pints.LogPrior
        A logprior to draw proposal samples from.

    """
    def __init__(self, log_prior):

        # Store logprior
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError('Given log_prior must extend pints.LogPrior')

        # prior accessed by subclasses to do prior sampling in ask() step
        self._log_prior = log_prior

        # Current value of the threshold log-likelihood value
        self._running_log_likelihood = -float('inf')
        self._proposed = None

        # Initialise active point containers
        self._n_active_points = 400
        self._n_parameters = self._log_prior.n_parameters()
        self._m_active = np.zeros((self._n_active_points,
                                   self._n_parameters + 1))
        self._min_index = None
        self._accept_count = 0
        self._n_evals = 0

        # multiple ellipsoid indicator
        self._multiple_ellipsoids = False
        self._ellipsoid_count = 0

    def active_points(self):
        """
        Returns the active points from nested sampling run.
        """
        return self._m_active

    def ask(self):
        """
        Proposes new point at which to evaluate log-likelihood.
        """
        raise NotImplementedError

    def _initialise_active_points(self, m_initial, v_fx):
        """
        Sets initial active points matrix.
        """
        for i, fx in enumerate(v_fx):
            self._m_active[i, self._n_parameters] = fx
        self._m_active[:, :-1] = m_initial
        self._min_index = np.argmin(self._m_active[:, self._n_parameters])
        self._set_running_log_likelihood(
            self._m_active[self._min_index, self._n_parameters])

    def in_initial_phase(self):
        """
        For methods that need an initial phase (see
        :meth:`needs_initial_phase()`), this method returns ``True`` if the
        method is currently configured to be in its initial phase. For other
        methods a ``NotImplementedError`` is returned.
        """
        raise NotImplementedError

    def min_index(self):
        """ Returns index of sample with lowest log-likelihood. """
        return self._min_index

    def n_active_points(self):
        """
        Returns the number of active points that will be used in next run.
        """
        return self._n_active_points

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        raise NotImplementedError

    def name(self):
        """ Name of sampler """
        raise NotImplementedError

    def needs_sensitivities(self):
        """
        Determines whether sampler uses sensitivities of the solution.
        """
        return self._needs_sensitivities

    def needs_initial_phase(self):
        """
        Returns ``True`` if this method needs an initial phase, for example
        ellipsoidal nested sampling has a period of running rejection
        sampling before it starts to fit ellipsoids to points.
        """
        return False

    def running_log_likelihood(self):
        """
        Returns current value of the threshold log-likelihood value.
        """
        return self._running_log_likelihood

    def set_n_active_points(self, active_points):
        """
        Sets the number of active points for the next run.
        """
        active_points = int(active_points)
        if active_points <= 5:
            raise ValueError('Number of active points must be greater than 5.')
        self._n_active_points = active_points
        self._m_active = np.zeros((self._n_active_points,
                                   self._n_parameters + 1))

    def set_hyper_parameters(self, x):
        """
        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        raise NotImplementedError

    def set_initial_phase(self, in_initial_phase):
        """
        For methods that need an initial phase (see
        :meth:`needs_initial_phase()`), this method toggles the initial phase
        algorithm. For other methods a ``NotImplementedError`` is returned.
        """
        raise NotImplementedError

    def _set_running_log_likelihood(self, running_log_likelihood):
        """
        Updates the current value of the threshold log-likelihood value.
        """
        self._running_log_likelihood = running_log_likelihood

    def tell(self, fx):
        """
        If a single evaluation is provided as arguments, a single point is
        accepted and returned if its likelihood exceeds the current threshold;
        otherwise None is returned.

        If multiple evaluations are provided as arguments (for example, if
        running the algorithm in parallel), None is returned if no points
        have likelihood exceeding threshold; if a single point passes the
        threshold, it is returned; if multiple points pass, one is selected
        uniformly at random and returned and the others are stored for later
        use.

        In all cases, two objects are returned: the proposed point (which may
        be None) and an array of other points that also pass the threshold
        (which is empty for single evaluation mode but may be non-empty for
        multiple evaluation mode).
        """

        # for serial evaluation just return point or None and an empty array
        if np.isscalar(fx):
            self._n_evals += 1
            if np.isnan(fx) or fx < self._running_log_likelihood:
                return None, np.array([[]])
            else:
                proposed = self._proposed
                fx_temp = fx
                winners = np.array([[]])

        # if running in parallel, then fx will be a sequence
        else:
            a_len = len(fx)
            self._n_evals += a_len
            results = []
            for i in range(a_len):
                if np.isnan(fx[i]) or fx[i] < self._running_log_likelihood:
                    results.append(None)
                else:
                    results.append(fx[i])
            n_non_none = sum(x is not None for x in results)

            # if none pass threshold return None and an empty array
            if n_non_none == 0:
                return None, np.array([[]])

            # if one passes then return it and an empty array
            elif n_non_none == 1:
                fx_temp = next(item for item in results if item is not None)
                index = results.index(fx_temp)
                proposed = self._proposed[index]
                winners = np.array([[]])

            # if more than a single point passes select at random from multiple
            # non-nones and return it and an array of the other points whose
            # likelihood exceeds threshold
            else:
                fx_short = [i for i in results if i]
                idex = [results.index(i) for i in fx_short]
                proposed_short = [self._proposed[i] for i in idex]
                fx_temp = np.random.choice(fx_short)
                index_temp = results.index(fx_temp)
                proposed = self._proposed[index_temp]
                index1 = fx_short.index(fx_temp)
                del proposed_short[index1]
                fx_short.remove(fx_temp)
                winners = np.transpose(
                    np.vstack([np.transpose(proposed_short), fx_short]))

        self._m_active[self._min_index, :] = np.concatenate(
            (proposed, np.array([fx_temp])))
        self._min_index = np.argmin(
            self._m_active[:, self._n_parameters])
        self._set_running_log_likelihood(
            np.min(self._m_active[:, self._n_parameters]))
        self._accept_count += 1
        return proposed, winners


class NestedController(object):
    """
    Uses nested sampling to sample from a posterior distribution.

    Parameters
    ----------
    log_likelihood : pints.LogPDF
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    log_prior : pints.LogPrior
        A :class:`LogPrior` function on the same parameter space.

    References
    ----------
    .. [1] "Nested Sampling for General Bayesian Computation", John Skilling,
           Bayesian Analysis 1:4 (2006).
           https://doi.org/10.1214/06-BA127

    .. [2] "Multimodal nested sampling: an efficient and robust alternative
            to Markov chain Monte Carlo methods for astronomical data analyses"
            F. Feroz and M. P. Hobson, 2008, Mon. Not. R. Astron. Soc.
    """

    def __init__(self, log_likelihood, log_prior, method=None):

        # Store log_likelihood and log_prior
        # if not isinstance(log_likelihood, pints.LogLikelihood):
        if not isinstance(log_likelihood, pints.LogPDF):
            raise ValueError(
                'Given log_likelihood must extend pints.LogLikelihood')
        self._log_likelihood = log_likelihood

        # Store function
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError('Given log_prior must extend pints.LogPrior')
        self._log_prior = log_prior

        # Get dimension
        self._n_parameters = self._log_likelihood.n_parameters()
        if self._n_parameters != self._log_prior.n_parameters():
            raise ValueError(
                'Given log_likelihood and log_prior must have same number of'
                ' parameters.')

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False

        # By default do serial evaluation
        self._parallel = False
        self._n_workers = 1
        self.set_parallel()

        # Parameters common to all routines
        # Total number of iterations
        self._iterations = 1000

        # Total number of posterior samples
        self._posterior_samples = 1000

        # Convergence criterion in log-evidence
        self._marginal_log_likelihood_threshold = 0.5

        # Initial marginal difference
        self._diff = np.float('-Inf')

        # By default use ellipsoidal sampling
        if method is None:
            method = pints.NestedEllipsoidSampler
        else:
            try:
                ok = issubclass(method, pints.NestedSampler)
            except TypeError:   # Not a class
                ok = False
            if not ok:
                raise ValueError(
                    'Given method must extend pints.NestedSampler.'
                )
        self._sampler = method(log_prior=self._log_prior)

        # Check if sensitivities are required
        self._needs_sensitivities = self._sampler.needs_sensitivities()

        # Performance metrics
        self._time = None

        # :meth:`run` can only be called once
        self._has_run = False

    def active_points(self):
        """
        Returns the active points from nested sampling.
        """
        return self._sampler.active_points()

    def _diff_marginal_likelihood(self, i, d):
        """
        Calculates difference in marginal likelihood between current and
        previous iterations.
        """
        v_temp = np.concatenate((
            self._v_log_Z[0:(i - 1)],
            [np.max(self._sampler._m_active[:, d])]
        ))
        w_temp = np.concatenate((self._w[0:(i - 1)], [self._X[i]]))
        self._diff = (
            + logsumexp(self._v_log_Z[0:(i - 1)], b=self._w[0:(i - 1)])
            - logsumexp(v_temp, b=w_temp)
        )

    def effective_sample_size(self):
        r"""
        Calculates the effective sample size of posterior samples from a
        nested sampling run using the formula:

        .. math::
            ESS = exp(-sum_{i=1}^{m} p_i log p_i),

        in other words, the information. Given by eqn. (39) in [1]_.
        """
        self._log_vP = (self._m_samples_all[:, self._n_parameters]
                        - self._log_Z + np.log(self._w))
        return np.exp(-np.sum(self._vP * self._log_vP))

    def inactive_points(self):
        """
        Returns the inactive points from nested sampling.
        """
        return self._m_inactive

    def _initialise_callable(self):
        """
        Initialises sensitivities if they are needed; otherwise, returns
        a callable log likelihood.
        """
        f = self._log_likelihood
        if self._needs_sensitivities:
            f = f.evaluateS1
        return f

    def _initialise_evaluator(self, f):
        """
        Initialises parallel runners, if desired.
        """
        # Create evaluator object
        if self._parallel:
            # Use at most n_workers workers
            n_workers = self._n_workers
            evaluator = pints.ParallelEvaluator(
                f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)
        return evaluator

    def _initialise_logger(self):
        """
        Initialises logger.
        """
        # Start logging
        self._logging = self._log_to_screen or self._log_filename
        if self._logging:

            if self._log_to_screen:
                # Show current settings
                print('Running ' + self._sampler.name())
                print('Number of active points: ' +
                      str(self._n_active_points))
                print('Total number of iterations: ' + str(self._iterations))
                print('Total number of posterior samples: ' + str(
                    self._posterior_samples))

            # Set up logger
            self._logger = pints.Logger()
            if not self._log_to_screen:
                self._logger.set_stream(None)
            if self._log_filename:
                self._logger.set_filename(
                    self._log_filename, csv=self._log_csv)

            # Add fields to log
            self._logger.add_counter('Iter.', max_value=self._iterations)
            self._logger.add_counter('Eval.', max_value=self._iterations * 10)
            self._logger.add_time('Time m:s')
            self._logger.add_float('Delta_log(z)')
            self._logger.add_float('Acceptance rate')
            if self._sampler._multiple_ellipsoids:
                self._logger.add_float('Ellipsoid count')

    def _initial_points(self):
        """
        Generates initial active points.
        """
        m_initial = self._log_prior.sample(self._n_active_points)
        v_fx = np.zeros(self._n_active_points)
        for i in range(0, self._n_active_points):
            # Calculate likelihood
            v_fx[i] = self._evaluator.evaluate([m_initial[i, :]])[0]
            self._sampler._n_evals += 1

            # Show progress
            if self._logging and i >= self._next_message:
                # Log state
                if not self._sampler._multiple_ellipsoids:
                    self._logger.log(0, self._sampler._n_evals,
                                     self._timer.time(), self._diff, 1.0)
                else:
                    self._logger.log(0, self._sampler._n_evals,
                                     self._timer.time(), self._diff, 1.0, 0.0)

                # Choose next logging point
                if i > self._message_warm_up:
                    self._next_message = self._message_interval * (
                        1 + i // self._message_interval)
        self._next_message = 0
        return v_fx, m_initial

    def iterations(self):
        """
        Returns the total number of iterations that will be performed in the
        next run.
        """
        return self._iterations

    def log_likelihood_vector(self):
        """
        Returns vector of log likelihoods for each of the stacked
        ``[m_active, m_inactive]`` points.
        """
        return self._m_samples_all[:, -1]

    def marginal_log_likelihood(self):
        """
        Calculates the marginal log likelihood of nested sampling run.
        """
        # Include active particles in sample
        m_active = self._sampler.active_points()
        self._v_log_Z[self._iterations] = logsumexp(m_active[:,
                                                    self._n_parameters])
        self._w[self._iterations:] = float(self._X[self._iterations]) / float(
            self._sampler.n_active_points())
        self._m_samples_all = np.vstack((self._m_inactive, m_active))

        # Determine log evidence
        log_Z = logsumexp(self._v_log_Z,
                          b=self._w[0:(self._iterations + 1)])
        self._log_Z_called = True
        return log_Z

    def marginal_log_likelihood_standard_deviation(self):
        """
        Calculates standard deviation in marginal log likelihood as in [2]_.
        """
        if not self._log_Z_called:
            self.marginal_log_likelihood()
        log_L_minus_Z = self._v_log_Z - self._log_Z
        log_Z_sd = logsumexp(log_L_minus_Z,
                             b=self._w[0:(self._iterations + 1)] *
                             log_L_minus_Z)
        log_Z_sd = np.sqrt(log_Z_sd / self._sampler.n_active_points())
        return log_Z_sd

    def marginal_log_likelihood_threshold(self):
        """
        Returns threshold for determining convergence in estimate of marginal
        log likelihood which leads to early termination of the algorithm.
        """
        return self._marginal_log_likelihood_threshold

    def n_posterior_samples(self):
        """
        Returns the number of posterior samples that will be returned (see
        :meth:`set_n_posterior_samples()`).
        """
        return self._posterior_samples

    def parallel(self):
        """
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False

    def posterior_samples(self):
        """
        Returns posterior samples generated during run of nested
        sampling object.
        """
        return self._m_posterior_samples

    def prior_space(self):
        """
        Returns a vector of X samples which approximates the proportion
        of prior space compressed.
        """
        return self._X

    def run(self):
        """
        Runs the nested sampling routine and returns a tuple of the posterior
        samples and an estimate of the marginal likelihood.
        """

        # Can only run once for each controller instance
        if self._has_run:
            raise RuntimeError("Controller is valid for single use only")
        self._has_run = True

        # Choose method to evaluate
        f = self._initialise_callable()

        # Set parallel
        self._evaluator = self._initialise_evaluator(f)

        # Set number of active points
        self._n_active_points = self._sampler.n_active_points()

        # Start timing
        self._timer = pints.Timer()

        # Set up progress reporting
        self._next_message = 0
        self._message_warm_up = 0
        self._message_interval = 20
        self._initialise_logger()

        d = self._n_parameters

        v_fx, m_initial = self._initial_points()
        self._sampler._initialise_active_points(m_initial, v_fx)

        # store all inactive points, along with their respective
        # log-likelihoods (hence, d+1)
        self._m_inactive = np.zeros((self._iterations, d + 1))

        # store weights
        self._w = np.zeros(self._n_active_points + self._iterations)

        # store X values (defined in [1])
        self._X = np.zeros(self._iterations + 1)
        self._X[0] = 1

        # log marginal likelihood holder
        self._v_log_Z = np.zeros(self._iterations + 1)

        # Run!
        self._X[0] = 1.0
        self._i_message = 0
        i_winners = 0
        m_previous_winners = []
        for i in range(0, self._iterations):
            i_iter_complete = 0
            self._i = i
            a_min_index = self._sampler.min_index()
            self._X[i + 1] = np.exp(-(i + 1) / self._n_active_points)
            if i > 0:
                self._w[i] = 0.5 * (self._X[i - 1] - self._X[i + 1])
            else:
                self._w[i] = self._X[i] - self._X[i + 1]
            self._v_log_Z[i] = self._sampler.running_log_likelihood()
            self._m_inactive[i, :] = self._sampler._m_active[a_min_index, :]

            # check whether previous winners exceed threshold
            if i_winners > 0:
                m_previous_winners = m_previous_winners[(
                    m_previous_winners[:, self._n_parameters] >
                    self._sampler.running_log_likelihood()), :]
                if m_previous_winners.shape[0] > 0:
                    index = np.random.choice(m_previous_winners.shape[0],
                                             1, replace=False)
                    proposed = m_previous_winners[index, :self._n_parameters]
                    fx_temp = m_previous_winners[index, self._n_parameters]
                    m_previous_winners = np.delete(m_previous_winners,
                                                   index, 0)
                    self._sampler._m_active[self._sampler._min_index, :] = (
                        np.concatenate((proposed[0], fx_temp))
                    )
                    self._sampler._min_index = np.argmin(
                        self._sampler._m_active[:, self._n_parameters])
                    self._sampler._set_running_log_likelihood(
                        np.min(self._sampler._m_active[:, self._n_parameters])
                    )
                    self._sampler._accept_count += 1
                    i_iter_complete = 1
            if i_iter_complete == 0:
                # Propose new samples
                proposed = self._sampler.ask(self._n_workers)
                # Evaluate their fit
                if self._n_workers > 1:
                    log_likelihood = self._evaluator.evaluate(proposed)
                else:
                    log_likelihood = self._evaluator.evaluate([proposed])[0]
                sample, winners = self._sampler.tell(log_likelihood)
                while sample is None:
                    proposed = self._sampler.ask(self._n_workers)
                    if self._n_workers > 1:
                        log_likelihood = (              # pragma: no cover
                            self._evaluator.evaluate(proposed))
                    else:
                        log_likelihood = self._evaluator.evaluate(
                            [proposed])[0]
                    sample, winners = self._sampler.tell(log_likelihood)
                if winners.size > 0:
                    if i_winners == 0:
                        m_previous_winners = winners
                        i_winners = 1
                    else:
                        m_previous_winners = [m_previous_winners, winners]
                        m_previous_winners = np.concatenate(m_previous_winners)

            # Check whether within convergence threshold
            if i > 2:
                self._diff_marginal_likelihood(i, d)
                if (np.abs(self._diff) <
                   self._marginal_log_likelihood_threshold):
                    if self._log_to_screen:
                        print(                              # pragma: no cover
                            'Convergence obtained with Delta_z = ' +
                            str(self._diff))

                    # shorten arrays according to current iteration
                    self._iterations = i
                    self._v_log_Z = self._v_log_Z[0:(self._iterations + 1)]
                    self._w = self._w[0:(
                        self._n_active_points + self._iterations)]
                    self._X = self._X[0:(self._iterations + 1)]
                    self._m_inactive = self._m_inactive[0:self._iterations, :]
                    break

            # Show progress
            self._update_logger()

        # Calculate log_evidence and uncertainty
        self._log_Z = self.marginal_log_likelihood()
        self._log_Z_sd = self.marginal_log_likelihood_standard_deviation()

        # Draw samples from posterior
        n = self._posterior_samples
        self._m_posterior_samples = self.sample_from_posterior(n)

        # Stop timer
        self._time = self._timer.time()

        return self._m_posterior_samples

    def sampler(self):
        """ Returns the underlying :class:`NestedSampler` object. """
        return self._sampler

    def sample_from_posterior(self, posterior_samples):
        """
        Draws posterior samples based on nested sampling run using importance
        sampling. This function is automatically called in
        ``NestedController.run()`` but can also be called afterwards to obtain
        new posterior samples.
        """
        if posterior_samples < 1:
            raise ValueError('Number of posterior samples must be positive.')

        # Calculate probabilities (can this be used to calculate effective
        # sample size as in importance sampling?) of each particle
        self._vP = np.exp(self._m_samples_all[:, self._n_parameters]
                          - self._log_Z) * self._w

        # Draw posterior samples
        m_theta = self._m_samples_all[:, :-1]
        vIndex = np.random.choice(
            range(0, self._iterations + self._sampler.n_active_points()),
            size=posterior_samples, p=self._vP)

        m_posterior_samples = m_theta[vIndex, :]
        return m_posterior_samples

    def set_iterations(self, iterations):
        """
        Sets the total number of iterations to be performed in the next run.
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

    def set_log_to_file(self, filename=None, csv=False):
        """
        Enables logging to file when a filename is passed in, disables it if
        ``filename`` is ``False`` or ``None``.

        The argument ``csv`` can be set to ``True`` to write the file in comma
        separated value (CSV) format. By default, the file contents will be
        similar to the output on screen.
        """
        if filename:
            self._log_filename = str(filename)
            self._log_csv = True if csv else False
        else:
            self._log_filename = None
            self._log_csv = False

    def set_log_to_screen(self, enabled):
        """
        Enables or disables logging to screen.
        """
        self._log_to_screen = True if enabled else False

    def set_marginal_log_likelihood_threshold(self, threshold):
        """
        Sets threshold for determining convergence in estimate of marginal
        log likelihood which leads to early termination of the algorithm.
        """
        if threshold <= 0:
            raise ValueError('Convergence threshold must be positive.')
        self._marginal_log_likelihood_threshold = threshold

    def set_parallel(self, parallel=False):
        """
        Enables/disables parallel evaluation.

        If ``parallel=True``, the method will run using a number of worker
        processes equal to the detected cpu core count. The number of workers
        can be set explicitly by setting ``parallel`` to an integer greater
        than 0.
        Parallelisation can be disabled by setting ``parallel`` to ``0`` or
        ``False``.
        """
        if parallel is True:
            self._parallel = True
            self._n_workers = pints.ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1

    def set_n_posterior_samples(self, posterior_samples):
        """
        Sets the number of posterior samples to generate from points proposed
        by the nested sampling algorithm.
        """
        posterior_samples = int(posterior_samples)
        if posterior_samples < 1:
            raise ValueError(
                'Number of posterior samples must be greater than zero.')
        self._posterior_samples = posterior_samples

    def time(self):
        """
        Returns the time needed for the last run, in seconds, or ``None`` if
        the controller hasn't run yet.
        """
        return self._time

    def _update_logger(self):
        """
        Updates logger if necessary.
        """
        # print(self._i_message)
        # print(self._next_message)
        if self._logging:
            self._i_message += 1
            if self._i_message >= self._next_message:
                # Log state
                if not self._sampler._multiple_ellipsoids:
                    self._logger.log(self._i_message, self._sampler._n_evals,
                                     self._timer.time(), self._diff,
                                     float(self._sampler._accept_count /
                                           (self._sampler._n_evals -
                                            self._sampler._n_active_points)))
                else:
                    self._logger.log(self._i_message, self._sampler._n_evals,
                                     self._timer.time(), self._diff,
                                     float(self._sampler._accept_count /
                                           (self._sampler._n_evals -
                                            self._sampler._n_active_points)),
                                     self._sampler._ellipsoid_count)

                # Choose next logging point
                if self._i_message > self._message_warm_up:
                    self._next_message = self._message_interval * (
                        1 + self._i_message // self._message_interval)


class Ellipsoid():
    """
    A class to represent N dimensional ellipsoids, which are needed by both
    ellipsoidal nested sampling and MultiNest.

    In "center form" the equation of an ellipsoid is given by:

    ``(x-c).T * A * (x-c) = 1``

    where ``A`` is a NxN dimensional positive definite symmetric matrix and
    ``c`` is a N dimensional vector indicating the center of the ellipsoid.

    Parameters
    ----------
    A : NxN dimensional positive definite symmetric matrix
        represents the orientation and size of ellipsoid
    c : N dimensional vector
        center of ellipsoid
    """
    def __init__(self, A, c):

        self._c = c
        self._n_parameters = len(c)

        if A.shape != (self._n_parameters, self._n_parameters):
            raise ValueError(
                'Sigma must have same dimension as mean, or be a square ' +
                'matrix with the same dimension as the center.')
        self._A = np.copy(A)

        # calculate useful quantities
        self._A_inv = np.linalg.inv(A)
        # don't calculate volume unless needed
        self._volume = None

        # don't cache points unless constructed using minimum_volume_ellipsoid
        self._points = None
        self._n_points = 0

    def set_points(self, points):
        """ Sets points contained within bounding ellipsoid. """
        self._points = points
        self._n_points = len(points)

    def centroid(self):
        """ Returns centroid of ellipsoid. """
        return self._c

    def enlarge(self, enlargement_factor):
        """ Enlarges ellipsoid by a factor. """
        self._A *= (1 / enlargement_factor)
        self._A_inv *= enlargement_factor
        self._volume = None

    @staticmethod
    def mahalanobis_distance(point, A, c):
        """
        Finds Mahalanobis distance between a point and the centroid of
        of an ellipsoid.
        """
        return np.matmul(np.matmul(point - c, A), point - c)

    @classmethod
    def minimum_volume_ellipsoid(cls, points):
        """
        Creates an approximate minimum bounding ellipsoid in "center form":
        ``(x-c).T * A * (x-c) = 1``.
        """
        cov = np.cov(np.transpose(points))
        cov_inv = np.linalg.inv(cov)
        c = np.mean(points, axis=0)
        dist = np.zeros(len(points))
        for i in range(len(points)):
            dist[i] = Ellipsoid.mahalanobis_distance(points[i], cov_inv, c)
        enlargement_factor = np.max(dist)
        A = (1.0 / enlargement_factor) * cov_inv
        obj = cls(A, c)
        obj.set_points(points)
        return obj

    def n_points(self):
        """ Returns number of points within bounding ellipsoid. """
        return self._n_points

    def points(self):
        """ Returns points within bounding ellipsoid. """
        return self._points

    def sample(self, npts, enlargement_factor=1):
        """
        Draws ``npts`` random uniform points from within the ellipsoid.

        Most of this functionality has been borrowed from:
        http://www.astro.gla.ac.uk/~matthew/blog/?p=368
        """
        ndims = self._n_parameters
        covmat = self._A_inv * enlargement_factor
        cent = self._c

        # calculate eigen_values (e) and eigen_vectors (v)
        eigen_values, eigen_vectors = np.linalg.eig(covmat)
        idx = (-eigen_values).argsort()[::-1][:ndims]
        e = eigen_values[idx]
        v = eigen_vectors[:, idx]
        e = np.diag(e)

        # generate radii of hyperspheres
        rs = np.random.uniform(0, 1, npts)

        # generate points
        pt = np.random.normal(0, 1, [npts, ndims])

        # get scalings for each point onto the surface of a unit
        # hypersphere
        fac = np.sum(pt**2, axis=1)

        # calculate scaling for each point to be within the unit
        # hypersphere with radii rs
        fac = (rs**(1 / ndims)) / np.sqrt(fac)
        pnts = np.zeros((npts, ndims))

        # scale points to the ellipsoid using the eigen_values and rotate
        # with the eigen_vectors and add centroid
        d = np.sqrt(np.diag(e))
        d.shape = (ndims, 1)

        for i in range(0, npts):
            # scale points to a uniform distribution within unit
            # hypersphere
            pnts[i, :] = fac[i] * pt[i, :]
            pnts[i, :] = np.dot(
                np.multiply(pnts[i, :], np.transpose(d)),
                np.transpose(v)
            ) + cent

        if npts > 1:
            return pnts
        else:
            return pnts[0]

    def volume(self):
        """
        Calculates volume of ellipsoid.
        See: https://math.stackexchange.com/questions/2751632/solve-for-volume-of-ellipsoid-mathbb-x-mathbf-mut-sigma-1-mathbb-x # noqa
        """
        if self._volume is None:
            d = self._n_parameters
            r = np.linalg.det(self._A_inv)
            vol = (
                (np.pi**(d / 2.0) / scipy.special.gamma((d / 2.0) + 1.0))
                * np.sqrt(r)
            )
            # cache volume calculation to avoid recomputation
            self._volume = vol
        return self._volume

    def weight_matrix(self):
        """ Returns weight matrix. """
        return self._A

    def within_ellipsoid(self, point):
        """ Determines if point is within ellipsoid. """
        return Ellipsoid.mahalanobis_distance(point,
                                              self.weight_matrix(),
                                              self.centroid()) <= 1

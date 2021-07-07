#
# Sub-module containing MCMC inference routines
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import os
import pints
import numpy as np


class MCMCSampler(pints.Loggable, pints.TunableMethod):
    """
    Abstract base class for (single or multi-chain) MCMC methods.

    All MCMC samplers implement the :class:`pints.Loggable` and
    :class:`pints.TunableMethod` interfaces.
    """

    def name(self):
        """
        Returns this method's full name.
        """
        raise NotImplementedError

    def in_initial_phase(self):
        """
        For methods that need an initial phase (see
        :meth:`needs_initial_phase()`), this method returns ``True`` if the
        method is currently configured to be in its initial phase. For other
        methods a ``NotImplementedError`` is returned.
        """
        raise NotImplementedError

    def needs_initial_phase(self):
        """
        Returns ``True`` if this method needs an initial phase, for example an
        adaptation-free period for adaptive covariance methods, or a warm-up
        phase for DREAM.
        """
        return False

    def needs_sensitivities(self):
        """
        Returns ``True`` if this methods needs sensitivities to be passed in to
        ``tell`` along with the evaluated logpdf.
        """
        return False

    def set_initial_phase(self, in_initial_phase):
        """
        For methods that need an initial phase (see
        :meth:`needs_initial_phase()`), this method toggles the initial phase
        algorithm. For other methods a ``NotImplementedError`` is returned.
        """
        raise NotImplementedError


class SingleChainMCMC(MCMCSampler):
    """
    Abstract base class for MCMC methods that generate a single markov chain,
    via an ask-and-tell interface.

    Extends :class:`MCMCSampler`.

    Parameters
    ----------
    x0
        An starting point in the parameter space.
    sigma0
        An optional (initial) covariance matrix, i.e., a guess of the
        covariance of the distribution to estimate, around ``x0``.
    """

    def __init__(self, x0, sigma0=None):

        # Check initial position
        self._x0 = pints.vector(x0)

        # Get number of parameters
        self._n_parameters = len(self._x0)

        # Check initial standard deviation
        if sigma0 is None:
            # Get representative parameter value for each parameter
            self._sigma0 = np.abs(self._x0)
            self._sigma0[self._sigma0 == 0] = 1
            # Use to create diagonal matrix
            self._sigma0 = np.diag(0.01 * self._sigma0)
        else:
            self._sigma0 = np.array(sigma0, copy=True)
            if np.product(self._sigma0.shape) == self._n_parameters:
                # Convert from 1d array
                self._sigma0 = self._sigma0.reshape((self._n_parameters,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                # Check if 2d matrix of correct size
                self._sigma0 = self._sigma0.reshape(
                    (self._n_parameters, self._n_parameters))

    def ask(self):
        """
        Returns a parameter vector to evaluate the LogPDF for.
        """
        raise NotImplementedError

    def tell(self, fx):
        """
        Performs an iteration of the MCMC algorithm, using the
        :class:`pints.LogPDF` evaluation ``fx`` of the point ``x`` specified by
        ``ask``.

        For methods that require sensitivities (see
        :meth:`MCMCSamper.needs_sensitivities`), ``fx`` should be a tuple
        ``(log_pdf, sensitivities)``, containing the values returned by
        :meth:`pints.LogPdf.evaluateS1()`.

        After a successful call, :meth:`tell()` returns a tuple
        ``(x, fx, accepted)``, where ``x`` contains the current position of the
        chain, ``fx`` contains the corresponding evaluation, and ``accepted``
        is a boolean indicating whether the last evaluated sample was added to
        the chain.

        Some methods may require multiple ask-tell calls per iteration. These
        methods can return ``None`` to indicate an iteration is still in
        progress.
        """
        raise NotImplementedError

    def replace(self, current, current_log_pdf, proposed=None):
        """
        Replaces the internal current position, current LogPDF, and proposed
        point (if any) by the user-specified values.

        This method can only be used once the initial position and LogPDF have
        been set (so after at least 1 round of ask-and-tell).

        This is an optional method, and some samplers may not support it.
        """
        raise NotImplementedError


class MultiChainMCMC(MCMCSampler):
    """
    Abstract base class for MCMC methods that generate multiple markov chains,
    via an ask-and-tell interface.

    Extends :class:`MCMCSampler`.

    Parameters
    ----------
    chains : int
        The number of MCMC chains to generate.
    x0
        A sequence of starting points. Can be a list of lists, a 2-dimensional
        array, or any other structure such that ``x0[i]`` is the starting point
        for chain ``i``.
    sigma0
        An optional initial covariance matrix, i.e., a guess of the covariance
        in ``logpdf`` around the points in ``x0`` (the same ``sigma0`` is used
        for each point in ``x0``).
        Can be specified as a ``(d, d)`` matrix (where ``d`` is the dimension
        of the parameterspace) or as a ``(d, )`` vector, in which case
        ``diag(sigma0)`` will be used.
    """

    def __init__(self, chains, x0, sigma0=None):

        # Check number of chains
        self._n_chains = int(chains)
        if self._n_chains < 1:
            raise ValueError('Number of chains must be at least 1.')

        # Check initial position(s)
        if len(x0) != chains:
            raise ValueError(
                'Number of initial positions must be equal to number of'
                ' chains.')
        self._n_parameters = len(x0[0])
        if not all([len(x) == self._n_parameters for x in x0[1:]]):
            raise ValueError('All initial points must have same dimension.')
        self._x0 = np.array([pints.vector(x) for x in x0])
        self._x0.setflags(write=False)

        # Check initial standard deviation
        if sigma0 is None:
            # Get representative parameter value for each parameter
            self._sigma0 = np.max(np.abs(self._x0), axis=0)
            self._sigma0[self._sigma0 == 0] = 1
            # Use to create diagonal matrix
            self._sigma0 = np.diag(0.01 * self._sigma0)
        else:
            self._sigma0 = np.array(sigma0, copy=True)
            if np.product(self._sigma0.shape) == self._n_parameters:
                # Convert from 1d array
                self._sigma0 = self._sigma0.reshape((self._n_parameters,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                # Check if 2d matrix of correct size
                self._sigma0 = self._sigma0.reshape(
                    (self._n_parameters, self._n_parameters))

    def ask(self):
        """
        Returns a sequence of parameter vectors to evaluate a LogPDF for.
        """
        raise NotImplementedError

    def current_log_pdfs(self):
        """
        Returns the log pdf values of the current points (i.e. of the most
        recent points returned by :meth:`tell()`).
        """
        raise NotImplementedError

    def tell(self, fxs):
        """
        Performs an iteration of the MCMC algorithm, using the
        :class:`pints.LogPDF` evaluations ``fxs`` of the points ``xs``
        specified by ``ask``.

        For methods that require sensitivities (see
        :meth:`MCMCSamper.needs_sensitivities`), each entry in ``fxs`` should
        be a tuple ``(log_pdf, sensitivities)``, containing the values returned
        by :meth:`pints.LogPdf.evaluateS1()`.

        After a successful call, :meth:`tell()` returns a tuple
        ``(xs, fxs, accepted)``, where ``x`` contains the current position of
        the chain, ``fx`` contains the corresponding evaluation, and
        ``accepted`` is an array of booleans indicating whether the last
        evaluated sample was added to the chain.

        Some methods may require multiple ask-tell calls per iteration. These
        methods can return ``None`` to indicate an iteration is still in
        progress.
        """
        raise NotImplementedError


class MCMCController(object):
    """
    Samples from a :class:`pints.LogPDF` using a Markov Chain Monte Carlo
    (MCMC) method.

    The method to use (either a :class:`SingleChainMCMC` class or a
    :class:`MultiChainMCMC` class) is specified at runtime. For example::

        mcmc = pints.MCMCController(
            log_pdf, 3, x0, method=pints.HaarioBardenetACMC)

    Properties related to the number if iterations, parallelisation, and
    logging can be set directly on the ``MCMCController`` object, e.g.::

        mcmc.set_max_iterations(1000)

    Sampler specific properties must be set on the internal samplers
    themselves, e.g.::

        for sampler in mcmc.samplers():
            sampler.set_target_acceptance_rate(0.2)

    Finally, to run an MCMC routine, call::

        chains = mcmc.run()

    By default, an MCMCController run will write regular progress updates to
    screen. This can be disabled using :meth:`set_log_to_screen()`. To write a
    similar progress log to a file, use :meth:`set_log_to_file()`. To store the
    chains and/or evaluations generated by :meth:`run()` to a file, use
    :meth:`set_chain_filename()` and :meth:`set_log_pdf_filename()`.

    Parameters
    ----------
    log_pdf : pints.LogPDF
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    chains : int
        The number of MCMC chains to generate.
    x0
        A sequence of starting points. Can be a list of lists, a 2-dimensional
        array, or any other structure such that ``x0[i]`` is the starting point
        for chain ``i``.
    sigma0
        An optional initial covariance matrix, i.e., a guess of the covariance
        in ``logpdf`` around the points in ``x0`` (the same ``sigma0`` is used
        for each point in ``x0``).
        Can be specified as a ``(d, d)`` matrix (where ``d`` is the dimension
        of the parameter space) or as a ``(d, )`` vector, in which case
        ``diag(sigma0)`` will be used.
    transform : pints.Transformation
        An optional :class:`pints.Transformation` to allow the sampler to work
        in a transformed parameter space. If used, points shown or returned to
        the user will first be detransformed back to the original space.
    method : class
        The class of :class:`MCMCSampler` to use. If no method is specified,
        :class:`HaarioBardenetACMC` is used.
    """

    def __init__(
            self, log_pdf, chains, x0, sigma0=None, transform=None,
            method=None):

        # Check function
        if not isinstance(log_pdf, pints.LogPDF):
            raise ValueError('Given function must extend pints.LogPDF')

        # Apply a transformation (if given). From this point onward the MCMC
        # sampler will see only the transformed search space and will know
        # nothing about the model parameter space.
        if transform is not None:
            # Convert log pdf
            log_pdf = transform.convert_log_pdf(log_pdf)

            # Convert initial positions
            x0 = [transform.to_search(x) for x in x0]

            # Convert sigma0, if provided
            if sigma0 is not None:
                sigma0 = np.asarray(sigma0)
                n_parameters = log_pdf.n_parameters()
                # Make sure sigma0 is a (covariance) matrix
                if np.product(sigma0.shape) == n_parameters:
                    # Convert from 1d array
                    sigma0 = sigma0.reshape((n_parameters,))
                    sigma0 = np.diag(sigma0)
                elif sigma0.shape != (n_parameters, n_parameters):
                    # Check if 2d matrix of correct size
                    raise ValueError(
                        'sigma0 must be either a (d, d) matrix or a (d, ) '
                        'vector, where d is the number of parameters.')
                sigma0 = transform.convert_covariance_matrix(sigma0, x0[0])

        # Store transform for later detransformation: if using a transform, any
        # parameters logged to the filesystem or printed to screen should be
        # detransformed first!
        self._transform = transform

        # Store function
        self._log_pdf = log_pdf

        # Get number of parameters
        self._n_parameters = self._log_pdf.n_parameters()

        # Check number of chains
        self._n_chains = int(chains)
        if self._n_chains < 1:
            raise ValueError('Number of chains must be at least 1.')

        # Check initial position(s): Most checking is done by samplers!
        if len(x0) != chains:
            raise ValueError(
                'Number of initial positions must be equal to number of'
                ' chains.')
        if not all([len(x) == self._n_parameters for x in x0]):
            raise ValueError(
                'All initial positions must have the same dimension as the'
                ' given LogPDF.')

        # Don't check initial standard deviation: done by samplers!

        # Set default method
        if method is None:
            method = pints.HaarioBardenetACMC
        else:
            try:
                ok = issubclass(method, pints.MCMCSampler)
            except TypeError:   # Not a class
                ok = False
            if not ok:
                raise ValueError('Given method must extend pints.MCMCSampler.')

        # Using single chain samplers?
        self._single_chain = issubclass(method, pints.SingleChainMCMC)

        # Create sampler(s)
        if self._single_chain:
            # Using n individual samplers (Note that it is possible to have
            # _single_chain=True and _n_samplers=1)
            self._n_samplers = self._n_chains
            self._samplers = [method(x, sigma0) for x in x0]
        else:
            # Using a single sampler that samples multiple chains
            self._n_samplers = 1
            self._samplers = [method(self._n_chains, x0, sigma0)]

        # Check if sensitivities are required
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()

        # Initial phase (needed for e.g. adaptive covariance)
        self._initial_phase_iterations = None
        self._needs_initial_phase = self._samplers[0].needs_initial_phase()
        if self._needs_initial_phase:
            self.set_initial_phase_iterations()

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_interval()

        # Storing chains and evaluations in memory
        self._chains_in_memory = True
        self._evaluations_in_memory = False
        self._samples = None
        self._evaluations = None
        self._n_evaluations = None
        self._time = None

        # Writing chains and evaluations to disk
        self._chain_files = None
        self._evaluation_files = None

        # Parallelisation
        self._parallel = False
        self._n_workers = 1
        self.set_parallel()

        # :meth:`run` can only be called once
        self._has_run = False

        #
        # Stopping criteria
        #

        # Maximum iterations
        self._max_iterations = None
        self.set_max_iterations()

        # TODO: Add more stopping criteria

    def chains(self):
        """
        Returns the chains generated by :meth:`run()`.

        The returned array has shape ``(n_chains, n_iterations,
        n_parameters)``.

        If the controller has not run yet, or if chain storage to memory is
        disabled, this method will return ``None``.
        """
        # Note: Not copying this, for efficiency. At this point we're done with
        # the chains, so nothing will go wrong if the user messes the array up.
        return self._samples

    def initial_phase_iterations(self):
        """
        For methods that require an initial phase (e.g. an adaptation-free
        phase for the adaptive covariance MCMC method), this returns the number
        of iterations that the initial phase will take.

        For methods that do not require an initial phase, a
        ``NotImplementedError`` is raised.
        """
        return self._initial_phase_iterations

    def log_pdfs(self):
        """
        Returns the :class:`LogPDF` evaluations generated by :meth:`run()`.

        If a :class:`LogPosterior` was used, the returned array will have shape
        ``(n_chains, n_iterations, 3)``, and for each sample the LogPDF,
        LogLikelihood, and LogPrior will be stored.
        For all other cases, only the full LogPDF evaluations are returned, in
        an array of shape ``(n_chains, n_iterations)``.

        If the controller has not run yet, or if storage of evaluations to
        memory is disabled (default), this method will return ``None``.
        """
        # Note: Not copying this, for efficiency. At this point we're done with
        # the chains, so nothing will go wrong if the user messes the array up.
        return self._evaluations

    def max_iterations(self):
        """
        Returns the maximum iterations if this stopping criterion is set, or
        ``None`` if it is not. See :meth:`set_max_iterations()`.
        """
        return self._max_iterations

    def method_needs_initial_phase(self):
        """
        Returns true if this sampler has been created with a method that has
        an initial phase (see :meth:`MCMCSampler.needs_initial_phase()`.)
        """
        return self._samplers[0].needs_initial_phase()

    def n_evaluations(self):
        """
        Returns the number of evaluations performed during the last run, or
        ``None`` if the controller hasn't run yet.
        """
        return self._n_evaluations

    def parallel(self):
        """
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False

    def run(self):
        """
        Runs the MCMC sampler(s) and returns the result.

        By default, this method returns an array of shape ``(n_chains,
        n_iterations, n_parameters)``.
        If storing chains to memory has been disabled with
        :meth:`set_chain_storage`, then ``None`` is returned instead.
        """

        # Can only run once for each controller instance
        if self._has_run:
            raise RuntimeError("Controller is valid for single use only")
        self._has_run = True

        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= (self._max_iterations is not None)
        if not has_stopping_criterion:
            raise ValueError('At least one stopping criterion must be set.')

        # Iteration and evaluation counting
        iteration = 0
        self._n_evaluations = 0

        # Choose method to evaluate
        f = self._log_pdf
        if self._needs_sensitivities:
            f = f.evaluateS1

        # Create evaluator object
        if self._parallel:
            # Use at most n_workers workers
            n_workers = min(self._n_workers, self._n_chains)
            evaluator = pints.ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Initial phase
        if self._needs_initial_phase:
            for sampler in self._samplers:
                sampler.set_initial_phase(True)

        # Storing evaluations to memory or disk
        prior = None
        store_evaluations = \
            self._evaluations_in_memory or self._evaluation_files
        if store_evaluations:
            # Bayesian inference on a log-posterior? Then separate out the
            # prior so we can calculate the loglikelihood
            if isinstance(self._log_pdf, pints.LogPosterior):
                prior = self._log_pdf.log_prior()

            # Store last accepted logpdf, per chain
            current_logpdf = np.zeros(self._n_chains)
            current_prior = np.zeros(self._n_chains)

        # Write chains to disk
        chain_loggers = []
        if self._chain_files:
            for filename in self._chain_files:
                cl = pints.Logger()
                cl.set_stream(None)
                cl.set_filename(filename, True)
                for k in range(self._n_parameters):
                    cl.add_float('p' + str(k))
                chain_loggers.append(cl)

        # Write evaluations to disk
        eval_loggers = []
        if self._evaluation_files:
            for filename in self._evaluation_files:
                cl = pints.Logger()
                cl.set_stream(None)
                cl.set_filename(filename, True)
                if prior:
                    # Logposterior in first column, to be consistent with the
                    # non-bayesian case
                    cl.add_float('logposterior')
                    cl.add_float('loglikelihood')
                    cl.add_float('logprior')
                else:
                    cl.add_float('logpdf')
                eval_loggers.append(cl)

        # Set up progress reporting
        next_message = 0

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            if self._log_to_screen:
                print('Using ' + str(self._samplers[0].name()))
                print('Generating ' + str(self._n_chains) + ' chains.')
                if self._parallel:
                    print('Running in parallel with ' + str(n_workers) +
                          ' worker processess.')
                else:
                    print('Running in sequential mode.')
                if self._chain_files:
                    print(
                        'Writing chains to ' + self._chain_files[0] + ' etc.')
                if self._evaluation_files:
                    print(
                        'Writing evaluations to ' + self._evaluation_files[0]
                        + ' etc.')

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            max_iter_guess = max(self._max_iterations or 0, 10000)
            max_eval_guess = max_iter_guess * self._n_chains
            logger.add_counter('Iter.', max_value=max_iter_guess)
            logger.add_counter('Eval.', max_value=max_eval_guess)
            for sampler in self._samplers:
                sampler._log_init(logger)
            logger.add_time('Time m:s')

        # Pre-allocate arrays for chain storage
        # Note: we store the inverse transformed (to model space) parameters
        # only if transform is provided.
        if self._chains_in_memory:
            # Store full chains
            samples = np.zeros(
                (self._n_chains, self._max_iterations, self._n_parameters))
        else:
            # Store only the current iteration
            samples = np.zeros((self._n_chains, self._n_parameters))

        # Pre-allocate arrays for evaluation storage
        if self._evaluations_in_memory:
            if prior:
                # Store posterior, likelihood, prior
                evaluations = np.zeros(
                    (self._n_chains, self._max_iterations, 3))
            else:
                # Store pdf
                evaluations = np.zeros((self._n_chains, self._max_iterations))

        # Some samplers need intermediate steps, where None is returned instead
        # of a sample. But samplers can run asynchronously, so that one returns
        # None while another returns a sample. To deal with this, we maintain a
        # list of 'active' samplers that have not reached `max_iterations` yet,
        # and we store the number of samples that we have in each chain.
        if self._single_chain:
            active = list(range(self._n_chains))
            n_samples = [0] * self._n_chains

        # Start sampling
        timer = pints.Timer()
        running = True
        while running:
            # Initial phase
            # Note: self._initial_phase_iterations is None when no initial
            # phase is needed
            if iteration == self._initial_phase_iterations:
                for sampler in self._samplers:
                    sampler.set_initial_phase(False)
                if self._log_to_screen:
                    print('Initial phase completed.')

            # Get points
            if self._single_chain:
                xs = [self._samplers[i].ask() for i in active]
            else:
                xs = self._samplers[0].ask()

            # Calculate logpdfs
            fxs = evaluator.evaluate(xs)

            # Update evaluation count
            self._n_evaluations += len(fxs)

            # Update chains
            if self._single_chain:
                # Single chain

                # Check and update the individual chains
                fxs_iterator = iter(fxs)
                for i in list(active):  # new list: active may be modified
                    reply = self._samplers[i].tell(next(fxs_iterator))

                    if reply is not None:
                        # Unpack reply into position, evaluation, and status
                        y, fy, accepted = reply

                        # Inverse transform to model space if transform is
                        # provided
                        if self._transform:
                            y_store = self._transform.to_model(y)
                        else:
                            y_store = y

                        # Store sample in memory
                        if self._chains_in_memory:
                            samples[i][n_samples[i]] = y_store
                        else:
                            samples[i] = y_store

                        # Update current evaluations
                        if store_evaluations:
                            # If accepted, update log_pdf and prior for logging
                            if accepted:
                                current_logpdf[i] = fy
                                if prior is not None:
                                    current_prior[i] = prior(y)

                            # Calculate evaluations to log
                            e = current_logpdf[i]
                            if prior is not None:
                                e = [e,
                                     current_logpdf[i] - current_prior[i],
                                     current_prior[i]]

                        # Store evaluations in memory
                        if self._evaluations_in_memory:
                            evaluations[i][n_samples[i]] = e

                        # Write evaluations to disk
                        if self._evaluation_files:
                            if prior is None:
                                eval_loggers[i].log(e)
                            else:
                                eval_loggers[i].log(*e)

                        # Stop adding samples if maximum number reached
                        n_samples[i] += 1
                        if n_samples[i] == self._max_iterations:
                            active.remove(i)

                # This is an intermediate step until the slowest sampler has
                # produced a new sample since the last `iteration`.
                intermediate_step = min(n_samples) <= iteration

            else:
                # Multi-chain methods

                # Get all chains samples at once
                reply = self._samplers[0].tell(fxs)
                intermediate_step = reply is None

                if not intermediate_step:
                    # Unpack reply into positions, evaluations, and status
                    ys, fys, accepted = reply

                    # Inverse transform to model space if transform is provided
                    if self._transform:
                        ys_store = np.zeros(ys.shape)
                        for i, y in enumerate(ys):
                            ys_store[i] = self._transform.to_model(y)
                    else:
                        ys_store = ys

                    # Store samples in memory
                    if self._chains_in_memory:
                        samples[:, iteration] = ys_store
                    else:
                        samples = ys_store

                    # Update current evaluations
                    if store_evaluations:
                        es = []
                        for i, y in enumerate(ys):
                            # Check if accepted, if so, update log_pdf and
                            # prior to be logged
                            if accepted[i]:
                                current_logpdf[i] = fys[i]
                                if prior is not None:
                                    current_prior[i] = prior(ys[i])

                            # Calculate evaluations to log
                            e = current_logpdf[i]
                            if prior is not None:
                                e = [e,
                                     current_logpdf[i] - current_prior[i],
                                     current_prior[i]]
                            es.append(e)

                    # Write evaluations to memory
                    if self._evaluations_in_memory:
                        for i, e in enumerate(es):
                            evaluations[i, iteration] = e

                    # Write evaluations to disk
                    if self._evaluation_files:
                        if prior is None:
                            for i, eval_logger in enumerate(eval_loggers):
                                eval_logger.log(es[i])
                        else:
                            for i, eval_logger in enumerate(eval_loggers):
                                eval_logger.log(*es[i])

            # If no new samples were added, then no MCMC iteration was
            # performed, and so the iteration count shouldn't be updated,
            # logging shouldn't be triggered, and stopping criteria shouldn't
            # be checked
            if intermediate_step:
                continue

            # Write samples to disk
            if self._chains_in_memory:
                for i, chain_logger in enumerate(chain_loggers):
                    chain_logger.log(*samples[i][iteration])
            else:
                for i, chain_logger in enumerate(chain_loggers):
                    chain_logger.log(*samples[i])

            # Show progress
            if logging and iteration >= next_message:
                # Log state
                logger.log(iteration, self._n_evaluations)
                for sampler in self._samplers:
                    sampler._log_write(logger)
                logger.log(timer.time())

                # Choose next logging point
                if iteration < self._message_warm_up:
                    next_message = iteration + 1
                else:
                    next_message = self._message_interval * (
                        1 + iteration // self._message_interval)

            # Update iteration count
            iteration += 1

            # Check requested number of samples
            if (self._max_iterations is not None and
                    iteration >= self._max_iterations):
                running = False
                halt_message = ('Halting: Maximum number of iterations ('
                                + str(iteration) + ') reached.')

        # Finished running
        self._time = timer.time()

        # Log final state and show halt message
        if logging:
            logger.log(iteration, self._n_evaluations)
            for sampler in self._samplers:
                sampler._log_write(logger)
            logger.log(self._time)
            if self._log_to_screen:
                print(halt_message)

        # Store evaluations in memory
        if self._evaluations_in_memory:
            self._evaluations = evaluations

        if self._chains_in_memory:
            # Store generated chains in memory
            self._samples = samples

        # Return generated chains
        return samples if self._chains_in_memory else None

    def sampler(self):
        """
        Returns the underlying :class:`MultiChainMCMC` object, or raises an
        error if :class:`SingleChainMCMC` objects are being used.

        See also: :meth:`samplers()`.
        """
        if self._single_chain:
            raise RuntimeError(
                'The method MCMCController.sampler() is only supported when a'
                ' MultiChainMCMC is selected. Please use'
                ' MCMCController.samplers() instead, to obtain a list of all'
                ' internal SingleChainMCMC instances.')
        return self._samplers[0]

    def samplers(self):
        """
        Returns a list containing the underlying sampler objects.

        If a :class:`SingleChainMCMC` method was selected, this will be a list
        containing as many :class:`SingleChainMCMC` objects as the number of
        chains. If a :class:`MultiChainMCMC` method was selected, this will be
        a list containing a single :class:`MultiChainMCMC` instance.
        """
        return self._samplers

    def set_chain_filename(self, chain_file):
        """
        Write chains to disk as they are generated.

        If a ``chain_file`` is specified, a CSV file will be created for each
        chain, to which samples will be written as they are accepted. To
        disable logging of chains, set ``chain_file=None``.

        Filenames for each chain file will be derived from ``chain_file``, e.g.
        if ``chain_file='chain.csv'`` and there are 2 chains, then the files
        ``chain_0.csv`` and ``chain_1.csv`` will be created. Each CSV file will
        start with a header (e.g. ``"p0","p1","p2",...``) and contain a sample
        on each subsequent line.
        """

        d = self._n_chains
        self._chain_files = None
        if chain_file:
            b, e = os.path.splitext(str(chain_file))
            self._chain_files = [b + '_' + str(i) + e for i in range(d)]

    def set_chain_storage(self, store_in_memory=True):
        """
        Store chains in memory as they are generated.

        By default, all generated chains are stored in memory as they are
        generated, and returned by :meth:`run()`. This method allows this
        behaviour to be disabled, which can be useful for very large chains
        which are already stored to disk (see :meth:`set_chain_filename()`).
        """
        self._chains_in_memory = bool(store_in_memory)

    def set_initial_phase_iterations(self, iterations=200):
        """
        For methods that require an initial phase (e.g. an adaptation-free
        phase for the adaptive covariance MCMC method), this sets the number of
        iterations that the initial phase will take.

        For methods that do not require an initial phase, a
        ``NotImplementedError`` is raised.
        """
        if not self._needs_initial_phase:
            raise NotImplementedError

        # Check input
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError(
                'Number of initial-phase iterations cannot be negative.')
        self._initial_phase_iterations = iterations

    def set_log_interval(self, iters=20, warm_up=3):
        """
        Changes the frequency with which messages are logged.

        Parameters
        ----------
        iters : int
            A log message will be shown every ``iters`` iterations.
        warm_up : int
            A log message will be shown every iteration, for the first
            ``warm_up`` iterations.
        """
        iters = int(iters)
        if iters < 1:
            raise ValueError('Interval must be greater than zero.')
        warm_up = max(0, int(warm_up))

        self._message_interval = iters
        self._message_warm_up = warm_up

    def set_log_pdf_filename(self, log_pdf_file):
        """
        Write :class:`LogPDF` evaluations to disk as they are generated.

        If an ``evaluation_file`` is specified, a CSV file will be created for
        each chain, to which :class:`LogPDF` evaluations will be written for
        every accepted sample. To disable this feature, set
        ``evaluation_file=None``. If the ``LogPDF`` being evaluated is a
        :class:`LogPosterior`, the individual likelihood and prior will also
        be stored.

        Filenames for each evaluation file will be derived from
        ``evaluation_file``, e.g. if ``evaluation_file='evals.csv'`` and there
        are 2 chains, then the files ``evals_0.csv`` and ``evals_1.csv`` will
        be created. Each CSV file will start with a header (e.g.
        ``"logposterior","loglikelihood","logprior"``) and contain the
        evaluations for i-th accepted sample on the i-th subsequent line.
        """

        d = self._n_chains
        self._evaluation_files = None
        if log_pdf_file:
            b, e = os.path.splitext(str(log_pdf_file))
            self._evaluation_files = [b + '_' + str(i) + e for i in range(d)]

    def set_log_pdf_storage(self, store_in_memory=False):
        """
        Store :class:`LogPDF` evaluations in memory as they are generated.

        By default, evaluations of the :class:`LogPDF` are not stored. This
        method can be used to enable storage of the evaluations for the
        accepted samples.
        After running, evaluations can be obtained using :meth:`evaluations()`.
        """
        self._evaluations_in_memory = bool(store_in_memory)

    def set_log_to_file(self, filename=None, csv=False):
        """
        Enables progress logging to file when a filename is passed in, disables
        it if ``filename`` is ``False`` or ``None``.

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
        Enables or disables progress logging to screen.
        """
        self._log_to_screen = True if enabled else False

    def set_max_iterations(self, iterations=10000):
        """
        Adds a stopping criterion, allowing the routine to halt after the
        given number of `iterations`.

        This criterion is enabled by default. To disable it, use
        `set_max_iterations(None)`.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError(
                    'Maximum number of iterations cannot be negative.')
        self._max_iterations = iterations

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

    def time(self):
        """
        Returns the time needed for the last run, in seconds, or ``None`` if
        the controller hasn't run yet.
        """
        return self._time


class MCMCSampling(MCMCController):
    """ Deprecated alias for :class:`MCMCController`. """

    def __init__(self, log_pdf, chains, x0, sigma0=None, method=None):
        # Deprecated on 2019-02-06
        import warnings
        warnings.warn(
            'The class `pints.MCMCSampling` is deprecated.'
            ' Please use `pints.MCMCController` instead.')
        super(MCMCSampling, self).__init__(log_pdf, chains, x0, sigma0,
                                           method=method)


def mcmc_sample(log_pdf, chains, x0, sigma0=None, method=None):
    """
    Sample from a :class:`pints.LogPDF` using a Markov Chain Monte Carlo
    (MCMC) method.

    Parameters
    ----------
    log_pdf : pints.LogPDF
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    chains : int
        The number of MCMC chains to generate.
    x0
        A sequence of starting points. Can be a list of lists, a 2-dimensional
        array, or any other structure such that ``x0[i]`` is the starting point
        for chain ``i``.
    sigma0
        An optional initial covariance matrix, i.e., a guess of the covariance
        in ``logpdf`` around the points in ``x0`` (the same ``sigma0`` is used
        for each point in ``x0``).
        Can be specified as a ``(d, d)`` matrix (where ``d`` is the dimension
        of the parameterspace) or as a ``(d, )`` vector, in which case
        ``diag(sigma0)`` will be used.
    method : class
        The class of :class:`MCMCSampler` to use. If no method is specified,
        :class:`HaarioBardenetACMC` is used.
    """
    return MCMCController(    # pragma: no cover
        log_pdf, chains, x0, sigma0, method=method).run()


def sample_initial_points(log_pdf, n_points, random_sampler=None,
                          max_tries=None, parallel=False, n_workers=None):
    """
    Draws parameter values from a given sampling distribution until either
    finite values for each of ``n_points`` have been generated or the total
    number of attempts exceeds ``max_tries``.

    If ``log_pdf`` is of :class:`LogPosterior`, then the
    ``log_pdf.log_prior().sample`` method is used for initialisation, although
    this is overruled by ``random_sampler`` if it is supplied.

    Parameters
    ----------
    log_pdf : pints.LogPDF
        A :class:`LogPDF` function that evaluates points in the parameter
        space. It is optional that ``log_pdf`` is a of type
        :class:`LogPosterior`.
    n_points : int
        The number of initial values to generate.
    random_sampler : stochastic function
        A function that when called returns draws from a probability
        distribution of the same dimensionality as ``log_pdf``. The only
        argument to this function should be an integer specifying the number of
        draws.
    max_tries : int
        Number of attempts to find a finite initial value across all
        ``n_points``. By default this is 50 x n_points.
    parallel : Boolean
        Whether to evaluate ``log_pdf`` in parallel (defaults to False).
    n_workers : int
        Number of workers on which to run parallel evaluation.
    """
    if random_sampler is not None and not callable(random_sampler):
        raise ValueError("random_sampler must be a callable function.")

    if random_sampler is None:
        if isinstance(log_pdf, pints.LogPosterior):
            random_sampler = log_pdf.log_prior().sample
        else:
            raise ValueError("If log_pdf not of class pints.LogPosterior " +
                             "then random_sampler must be supplied.")

    if n_points < 1:
        raise ValueError("Number of initial points must be 1 or more.")

    if max_tries is None:
        max_tries = 50 * n_points

    if parallel:
        n_workers = min(pints.ParallelEvaluator.cpu_count(), n_points)
        evaluator = pints.ParallelEvaluator(log_pdf, n_workers=n_workers)
    else:
        evaluator = pints.SequentialEvaluator(log_pdf)

    initialised_finite = False
    x0 = []
    n_tries = 0
    while not initialised_finite and n_tries < max_tries:
        xs = random_sampler(n_points)
        fxs = evaluator.evaluate(xs)
        xs_iterator = iter(xs)
        fxs_iterator = iter(fxs)
        for i in range(n_points):
            x = next(xs_iterator)
            fx = next(fxs_iterator)
            if np.isfinite(fx):
                x0.append(x)
            if len(x0) == n_points:
                initialised_finite = True
            n_tries += 1
    if not initialised_finite:
        raise RuntimeError(
            'Initialisation failed since log_pdf not finite at initial ' +
            'points after ' + str(max_tries) + ' attempts.')
    return x0

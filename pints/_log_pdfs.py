#
# Main Log PDF functions
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class LogPDF(object):
    """
    Represents the natural logarithm of a (not necessarily normalised)
    probability density function (PDF).

    All :class:`LogPDF` types are callable: when called with a vector argument
    ``p`` they return some value ``log(f(p))`` where ``f(p)`` is an
    unnormalised PDF. The size of the argument ``p`` is given by
    :meth:`n_parameters()`.
    """
    def __call__(self, x):
        raise NotImplementedError

    def evaluateS1(self, x):
        """
        Evaluates this LogPDF, and returns the result plus the partial
        derivatives of the result with respect to the parameters.

        The returned data is a tuple ``(L, L')`` where ``L`` is a scalar value
        and ``L'`` is a sequence of length ``n_parameters``.

        Note that the derivative returned is of the log-pdf, so
        ``L' = d/dp log(f(p))``, evaluated at ``p=x``.

        *This is an optional method that is not always implemented.*
        """
        raise NotImplementedError

    def n_parameters(self):
        """
        Returns the dimension of the space this :class:`LogPDF` is defined
        over.
        """
        raise NotImplementedError


class LogPrior(LogPDF):
    """
    Represents the natural logarithm ``log(f(theta))`` of a known probability
    density function ``f(theta)``.

    Priors are *usually* normalised (i.e. the integral ``f(theta)`` over all
    points ``theta`` in parameter space sums to 1), but this is not a strict
    requirement.

    Extends :class:`LogPDF`.
    """
    def cdf(self, x):
        """
        Returns the cumulative density function at point(s) ``x``.

        ``x`` should be an ``n x d`` array, where ``n`` is the number of input
        samples and ``d`` is the dimension of the parameter space.
        """
        raise NotImplementedError

    def convert_from_unit_cube(self, u):
        """
        Converts samples ``u`` uniformly drawn from the unit cube into those
        drawn from the prior space, typically by transforming using
        :meth:`LogPrior.icdf()`.

        ``u`` should be an ``n x d`` array, where ``n`` is the number of input
        samples and ``d`` is the dimension of the parameter space.
        """
        return self.icdf(u)

    def convert_to_unit_cube(self, x):
        """
        Converts samples from the prior ``x`` to be drawn uniformly from the
        unit cube, typically by transforming using :meth:`LogPrior.cdf()`.

        ``x`` should be an ``n x d`` array, where ``n`` is the number of input
        samples and ``d`` is the dimension of the parameter space.
        """
        return self.cdf(x)

    def icdf(self, p):
        """
        Returns the inverse cumulative density function at cumulative
        probability/probabilities ``p``.

        ``p`` should be an ``n x d`` array, where ``n`` is the number of input
        samples and ``d`` is the dimension of the parameter space.
        """
        raise NotImplementedError

    def mean(self):
        """
        Returns the analytical value of the expectation of a random variable
        distributed according to this :class:`LogPDF`.
        """
        raise NotImplementedError

    def sample(self, n=1):
        """
        Returns ``n`` random samples from the underlying prior distribution.

        The returned value is a NumPy array with shape ``(n, d)`` where ``n``
        is the requested number of samples, and ``d`` is the dimension of the
        prior.
        """
        raise NotImplementedError


class PooledLogPDF(LogPDF):
    r"""
    Combines :math:`m` :class:`LogPDFs<pints.LogPDF>`, each with :math:`n`
    parameters, into a single LogPDF where :math:`k` parameters are "pooled"
    (i.e. have the same value for each LogPDF), so that the resulting combined
    LogPDF has :math:`m (n - k) + k` independent parameters.

    This is useful for e.g. modelling the time-series of multiple individuals
    (each individual defines a separate :class:`LogPDF`), and some parameters
    are expected to be the same across individuals (for example, the noise
    parameter across different individuals within the same experiment).

    For two :class:`LogPDFs<pints.LogPDF>` :math:`L _1`  and
    :math:`L _2` with four parameters
    :math:`(\psi ^{(1)}_1, \psi ^{(1)}_2, \psi ^{(1)}_3, \psi ^{(1)}_4)`
    and
    :math:`(\psi ^{(2)}_1, \psi ^{(2)}_2, \psi ^{(2)}_3, \psi ^{(2)}_4)`
    respectively, a pooling of the second and third parameter
    :math:`\psi _2 := \psi ^{(1)}_2 = \psi ^{(2)}_2`,
    :math:`\psi _3 := \psi ^{(1)}_3 = \psi ^{(2)}_3` results in a pooled
    log-pdf of the form

    .. math::
        L(\psi ^{(1)}_1, \psi ^{(1)}_4, \psi ^{(2)}_1, \psi ^{(2)}_4, \psi _2,
                \psi _3 | D_1, D_2) =
            L _1(\psi ^{(1)}_1, \psi _2, \psi _3, \psi ^{(1)}_4 | D_1) +
            L _2(\psi ^{(2)}_1, \psi _2, \psi _3, \psi ^{(2)}_4 | D_2),

    :math:`D_i` is the measured time-series of individual :math:`i`. As
    :math:`k=2` parameters where pooled across the log-likelihoods, the
    pooled log-likelihood has six parameters in the following order:
    :math:`(\psi ^{(1)}_1, \psi ^{(1)}_4, \psi ^{(2)}_1, \psi ^{(2)}_4,
    \psi _2, \psi _3)`.

    Note that the input parameters of a :class:`PooledLogPDF` are not just a
    simple concatenation of the parameters of the individual
    :class:`LogPDFs<pints.LogPDF>`. The pooled parameters are only listed
    once and are moved to the end of the parameter list. This avoids inputting
    the value of the pooled parameters at mutliple positions. Otherwise the
    order of the parameters is determined firstly by the order of the
    likelihoods and then by the order of the parameters of those likelihoods.

    Extends :class:`LogPDF`.

    Parameters
    ----------
    log_pdfs
        A sequence of :class:`LogPDF` objects.
    pooled
        A sequence of booleans indicating which parameters across
        the likelihoods are pooled (``True``) or remain unpooled (``False``).

    Example
    -------
    ::

        pooled_log_likelihood = pints.PooledLogPDF(
            log_pdfs=[
                pints.GaussianLogLikelihood(problem1),
                pints.GaussianLogLikelihood(problem2)],
            pooled=[False, True])
    """
    def __init__(self, log_pdfs, pooled):
        super(PooledLogPDF, self).__init__()

        # Check input arguments
        if len(log_pdfs) < 2:
            raise ValueError(
                'PooledLogPDF requires at least two log-pdfs.')
        for index, pdf in enumerate(log_pdfs):
            if not isinstance(pdf, LogPDF):
                raise ValueError(
                    'All log-pdfs passed to PooledLogPDFs must be instances of'
                    ' pints.LogPDF (failed on argument '
                    + str(index) + ').')

        # Check parameter dimension across log-pdfs
        self._log_pdfs = log_pdfs
        n_parameters = self._log_pdfs[0].n_parameters()
        for pdf in self._log_pdfs:
            if pdf.n_parameters() != n_parameters:
                raise ValueError(
                    'All log-pdfs passed to PooledLogPDFs must have '
                    'same dimension.')

        # Check that pooled matches number of parameters
        self._pooled = np.asarray(pooled)
        if len(self._pooled) != n_parameters:
            raise ValueError(
                'The array-like input `pooled` needs to have the same length '
                'as the number of parameters of the individual log-pdfs.')

        # Check that pooled contains only booleans
        if self._pooled.dtype != np.bool:
            raise ValueError(
                'The array-like input `pooled` passed to PooledLogPDFs '
                'has to contain booleans exclusively.')

        # Get dimension of search space
        self._n_pooled = np.sum(self._pooled)
        n_individuals = len(self._log_pdfs)
        self._n_unpooled = np.sum(~self._pooled)
        self._n_parameters = \
            self._n_pooled + n_individuals * self._n_unpooled

    def __call__(self, parameters):
        # Get parameters of pooled log-pdf
        parameters = np.asarray(parameters)

        # Create container for parameters of individuals log-pdf and fill with
        # pooled parameters
        params_ind = np.empty(shape=self._n_unpooled + self._n_pooled)
        if self._n_pooled > 0:
            params_ind[self._pooled] = parameters[
                self._n_parameters - self._n_pooled:self._n_parameters]

        # Compute pdf score
        total = 0
        for idx, pdf in enumerate(self._log_pdfs):
            # Get unpooled parameters for individual
            params_ind[~self._pooled] = parameters[
                idx * self._n_unpooled: (idx + 1) * self._n_unpooled]

            # Compute pdf score contribution
            total += pdf(params_ind)
        return total

    def evaluateS1(self, parameters):
        r"""
        See :meth:`LogPDF.evaluateS1()`.

        The partial derivatives of the pooled log-likelihood with respect to
        unpooled parameters equals the partial derivative of the corresponding
        indiviudal log-likelihood.

        .. math::
            \frac{\partial L}{\partial \psi} =
            \frac{\partial L_i}{\partial \psi},

        where :math:`L` is the pooled log-likelihood, :math:`\psi` an unpooled
        parameter and :math:`L _i` the individual log-likelihood that depends
        on :math:`\psi`.

        For a pooled parameter :math:`\theta` the partial derivative of the
        pooled log-likelihood equals to the sum of partial derivatives of all
        individual log-likelihoods

        .. math::
            \frac{\partial L}{\partial \theta} =
            \sum _{i=1}^n\frac{\partial L_i}{\partial \theta}.

        Here :math:`n` is the number of individual log-likelihoods.

        *This method only works if all the underlying :class:`LogPDF` objects
        implement the optional method :meth:`LogPDF.evaluateS1()`!*
        """
        # Get parameters of pooled log-pdf
        parameters = np.asarray(parameters)

        # Create container for parameters of individuals log-pdf and fill with
        # pooled parameters
        params_ind = np.empty(shape=self._n_unpooled + self._n_pooled)
        if self._n_pooled > 0:
            params_ind[self._pooled] = parameters[
                self._n_parameters - self._n_pooled:self._n_parameters]

        # Compute pdf score and partials
        total = 0
        dtotal = np.zeros(shape=self._n_parameters)
        for idx, pdf in enumerate(self._log_pdfs):
            # Get unpooled parameters for individual
            params_ind[~self._pooled] = parameters[
                idx * self._n_unpooled: (idx + 1) * self._n_unpooled]

            # Compute pdf score and partials for individual
            score, partials = pdf.evaluateS1(params_ind)

            # Add contributions to score and partials.
            # NOTE: Partials of unpooled parameters equal partials of the
            # associated individual likelihood; Partials of pooled parameters
            # equals to the sum of partials from the individual likelihoods
            # with respect to that parameter.
            total += score
            dtotal[idx * self._n_unpooled: (idx + 1) * self._n_unpooled] = \
                partials[~self._pooled]
            if self._n_pooled > 0:
                dtotal[-self._n_pooled:] += partials[self._pooled]

        return total, dtotal

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters


class ProblemLogLikelihood(LogPDF):
    """
    Represents a log-likelihood on a problem's parameter space, used to
    indicate the likelihood of an observed (fixed) time-series given a
    particular parameter set (variable).

    Extends :class:`LogPDF`.

    Parameters
    ----------
    problem
        The time-series problem this log-likelihood is defined for.
    """
    def __init__(self, problem):
        super(ProblemLogLikelihood, self).__init__()
        self._problem = problem
        # Cache some problem variables
        self._values = problem.values()
        self._times = problem.times()
        self._n_parameters = problem.n_parameters()

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters

    def evaluate_pointwise_loglikelihoods(self, x):
        """
        Evaluates the Log-likelihood at each observation for the given
        parameters, x. Returns a numpy array of length no. timepoints if no.
        outputs = 1. Otherwise returns a 2d array of size no. timepoints by no.
        outputs.

        *This is an optional method that is not always implemented.*
        """

        raise NotImplementedError


class LogPosterior(LogPDF):
    """
    Represents the sum of a :class:`LogPDF` and a :class:`LogPrior` defined on
    the same parameter space.

    As an optimisation, if the :class:`LogPrior` evaluates as `-inf` for a
    particular point in parameter space, the corresponding :class:`LogPDF` will
    not be evaluated.

    Extends :class:`LogPDF`.

    Parameters
    ----------
    log_likelihood
        A :class:`LogPDF`, defined on the same parameter space.
    log_prior
        A :class:`LogPrior`, representing prior knowledge of the parameter
        space.
    """
    def __init__(self, log_likelihood, log_prior):
        super(LogPosterior, self).__init__()

        # Check arguments
        if not isinstance(log_prior, LogPrior):
            raise ValueError(
                'Given prior must extend pints.LogPrior.')
        if not isinstance(log_likelihood, LogPDF):
            raise ValueError(
                'Given log_likelihood must extend pints.LogPDF.')

        # Check dimensions
        self._n_parameters = log_prior.n_parameters()
        if log_likelihood.n_parameters() != self._n_parameters:
            raise ValueError(
                'Given log_prior and log_likelihood must have same dimension.')

        # Store prior and likelihood
        self._log_prior = log_prior
        self._log_likelihood = log_likelihood

        # Store -inf, for later use
        self._minf = -float('inf')

    def __call__(self, x):
        # Evaluate log-prior first, assuming this is very cheap
        log_prior = self._log_prior(x)
        if log_prior == self._minf:
            return self._minf
        return log_prior + self._log_likelihood(x)

    def evaluateS1(self, x):
        """
        Evaluates this LogPDF, and returns the result plus the partial
        derivatives of the result with respect to the parameters.

        The returned data has the shape ``(L, L')`` where ``L`` is a scalar
        value and ``L'`` is a sequence of length ``n_parameters``.

        *This method only works if the underlying :class:`LogPDF` and
        :class:`LogPrior` implement the optional method
        :meth:`LogPDF.evaluateS1()`!*
        """
        #TODO: Is there an optimisation to be made here?
        a, da = self._log_prior.evaluateS1(x)
        b, db = self._log_likelihood.evaluateS1(x)
        return a + b, da + db

    def log_likelihood(self):
        """ Returns the :class:`LogLikelihood` used by this posterior. """
        return self._log_likelihood

    def log_prior(self):
        """ Returns the :class:`LogPrior` used by this posterior. """
        return self._log_prior

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters


class SumOfIndependentLogPDFs(LogPDF):
    """
    Calculates a sum of :class:`LogPDF` objects, all defined on the same
    parameter space.

    This is useful for e.g. Bayesian inference using a single model evaluated
    on two **independent** data sets ``D`` and ``E``. In this case,

    .. math::
        f(\\theta|D,E) &= \\frac{f(D, E|\\theta)f(\\theta)}{f(D, E)} \\\\
                       &= \\frac{f(D|\\theta)f(E|\\theta)f(\\theta)}{f(D, E)}

    Extends :class:`LogPDF`.

    Parameters
    ----------
    log_likelihoods
        A sequence of :class:`LogPDF` objects.

    Example
    -------
    ::

        log_likelihood = pints.SumOfIndependentLogPDFs([
            pints.GaussianLogLikelihood(problem1),
            pints.GaussianLogLikelihood(problem2),
        ])
    """
    def __init__(self, log_likelihoods):
        super(SumOfIndependentLogPDFs, self).__init__()

        # Check input arguments
        if len(log_likelihoods) < 2:
            raise ValueError(
                'SumOfIndependentPdfs requires at least two log-pdfs.')
        for i, e in enumerate(log_likelihoods):
            if not isinstance(e, LogPDF):
                raise ValueError(
                    'All objects passed to SumOfIndependentLogPDFs must'
                    ' be instances of pints.LogPDF (failed on argument '
                    + str(i) + ').')
        self._log_likelihoods = list(log_likelihoods)

        # Get and check dimension
        i = iter(self._log_likelihoods)
        self._n_parameters = next(i).n_parameters()
        for e in i:
            if e.n_parameters() != self._n_parameters:
                raise ValueError(
                    'All log-likelihoods passed to'
                    ' SumOfIndependentLogPDFs must have same dimension.')

    def __call__(self, x):
        total = 0
        for e in self._log_likelihoods:
            total += e(x)
        return total

    def evaluateS1(self, x):
        """
        See :meth:`LogPDF.evaluateS1()`.

        *This method only works if all the underlying :class:`LogPDF` objects
        implement the optional method :meth:`LogPDF.evaluateS1()`!*
        """
        total = 0
        dtotal = np.zeros(self._n_parameters)
        for e in self._log_likelihoods:
            a, b = e.evaluateS1(x)
            total += a
            dtotal += np.asarray(b)
        return total, dtotal

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters

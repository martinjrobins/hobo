#
# Defines different prior distributions
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy
import scipy.special
import scipy.stats


class BetaLogPrior(pints.LogPrior):
    r"""
    Defines a beta (log) prior with given shape parameters ``a`` and ``b``,
    with pdf

    .. math::
        f(x|a,b) = \frac{x^{a-1} (1-x)^{b-1}}{\mathrm{B}(a,b)}

    where :math:`\mathrm{B}` is the Beta function. A random variable :math:`X`
    distributed according to this pdf has expectation

    .. math::
        \mathrm{E}(X)=\frac{a}{a+b}.

    For example, to create a prior with shape parameters ``a=5`` and ``b=1``,
    use::

        p = pints.BetaLogPrior(5, 1)

    Extends :class:`LogPrior`.
    """

    def __init__(self, a, b):
        # Parse input arguments
        self._a = float(a)
        self._b = float(b)

        # Validate inputs
        if self._a <= 0:
            raise ValueError('Shape parameter a must be positive')
        if self._b <= 0:
            raise ValueError('Shape parameter b must be positive')

        # Cache constant
        self._log_beta = scipy.special.betaln(self._a, self._b)

    def __call__(self, x):
        if x[0] < 0.0 or x[0] > 1.0:
            return -np.inf
        else:
            return scipy.special.xlogy(self._a - 1.0,
                                       x[0]) + scipy.special.xlog1py(
                self._b - 1.0, -x[0]) - self._log_beta

    def cdf(self, x):
        """ See :meth:`LogPrior.cdf()`. """
        return scipy.stats.beta.cdf(x, self._a, self._b)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        value = self(x)
        _x = x[0]

        # Account for pathological edges
        if _x == 0.0:
            _x = np.nextafter(0.0, 1.0)
        elif _x == 1.0:
            _x = np.nextafter(1.0, 0.0)

        if _x < 0.0 or _x > 1.0:
            return value, np.asarray([0.])
        else:
            # Use np.divide here to better handle possible v small denominators
            return value, np.asarray([np.divide(self._a - 1., _x) - np.divide(
                self._b - 1., 1. - _x)])

    def icdf(self, p):
        """ See :meth:`LogPrior.icdf()`. """
        return scipy.stats.beta.ppf(p, self._a, self._b)

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return self._a / (self._a + self._b)

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.beta(self._a, self._b, size=(n, 1))


class CauchyLogPrior(pints.LogPrior):
    r"""
    Defines a 1-d Cauchy (log) prior with a given ``location``, and ``scale``,
    with pdf

    .. math::
        f(x|\text{location}, \text{scale}) = \frac{1}{\pi\;\text{scale}
        \left[1 + \left(\frac{x-\text{location}}{\text{scale}}\right)^2
        \right]}.

    A random variable distributed according to this pdf has undefined
    expectation.

    For example, to create a prior centered around 0 and a scale of 5, use::

        p = pints.CauchyLogPrior(0, 5)

    Extends :class:`LogPrior`.

    Parameters
    ----------
    location
        The center of the distribution.
    scale
        The scale of the distribution.
    """

    def __init__(self, location, scale):
        # Test inputs
        if float(scale) <= 0:
            raise ValueError('Scale must be positive')

        self._location = location
        self._scale = scale

        # Cache constants
        self._pi_sig = np.pi * self._scale
        self._pi_on_sig = np.pi / self._scale

    def __call__(self, x):
        _x_sq = (x[0] - self._location) * (x[0] - self._location)
        return -np.log(self._pi_sig + self._pi_on_sig * _x_sq)

    def cdf(self, x):
        """ See :meth:`LogPrior.cdf()`. """
        return scipy.stats.cauchy.cdf(x, self._location, self._scale)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        value = self(x)
        loc_minus_x = self._location - x[0]
        return value, np.asarray([2 * loc_minus_x / (self._scale**2
                                  + loc_minus_x**2)])

    def icdf(self, p):
        """ See :meth:`LogPrior.icdf()`. """
        return scipy.stats.cauchy.ppf(p, self._location, self._scale)

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return np.nan

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return scipy.stats.cauchy.rvs(loc=self._location, scale=self._scale,
                                      size=(n, 1))


class ComposedLogPrior(pints.LogPrior):
    r"""
    N-dimensional :class:`LogPrior` composed of one or more other :math:`N_i`-
    dimensional LogPriors, such that :math:`\sum _i N_i = N`. The evaluation
    of the composed log-prior assumes the input log-priors are all independent
    from each other.

    For example, a composed log prior

        ``p = pints.ComposedLogPrior(log_prior1, log_prior2, log_prior3)``,

    where ``log_prior1``, ``log_prior2``, and ``log_prior3`` each have
    dimension 1, 2 and 1, will have dimension 4.

    The dimensionality of the individual priors does not have to be the same,
    i.e. :math:`N_i\neq N_j` is allowed.

    The input parameters of the :class:`ComposedLogPrior` have to be ordered in
    the same way as the individual priors. In the above example the prior may
    be evaluated by ``p(x)``, where:

        ``x = [parameter1_log_prior1, parameter1_log_prior2,
        parameter2_log_prior2, parameter1_log_prior3]``.

    Extends :class:`LogPrior`.
    """

    def __init__(self, *priors):
        # Check if sub-priors given
        if len(priors) < 1:
            raise ValueError('Must have at least one sub-prior')

        # Check if proper priors, count dimension
        self._n_parameters = 0
        for prior in priors:
            if not isinstance(prior, pints.LogPrior):
                raise ValueError('All sub-priors must extend pints.LogPrior.')
            self._n_parameters += prior.n_parameters()

        # Store
        self._priors = priors

    def __call__(self, x):
        output = 0
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            output += prior(x[lo:hi])
        return output

    def cdf(self, x):
        """
        See :meth:`LogPrior.cdf()`.

        *This method only works if the underlying :class:`LogPrior` classes all
        implement the optional method :class:`LogPDF.cdf().`.*
        """
        cdfs = []
        for i, prior in enumerate(self._priors):
            cdfs.append(prior.cdf(x[i]))
        return cdfs

    def evaluateS1(self, x):
        """
        See :meth:`LogPDF.evaluateS1()`.

        *This method only works if the underlying :class:`LogPrior` classes all
        implement the optional method :class:`LogPDF.evaluateS1().`.*
        """
        output = 0
        doutput = np.zeros(self._n_parameters)
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            p, dp = prior.evaluateS1(x[lo:hi])
            output += p
            doutput[lo:hi] = np.asarray(dp)
        return output, doutput

    def icdf(self, x):
        """
        See :meth:`LogPrior.icdf()`.

        *This method only works if the underlying :class:`LogPrior` classes all
        implement the optional method :class:`LogPDF.icdf().`.*
        """
        icdfs = []
        for i, prior in enumerate(self._priors):
            icdfs.append(prior.icdf(x[i]))
        return icdfs

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._n_parameters

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        output = np.zeros((n, self._n_parameters))
        lo = hi = 0
        for prior in self._priors:
            lo = hi
            hi += prior.n_parameters()
            output[:, lo:hi] = prior.sample(n)
        return output

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return [prior.mean() for prior in self._priors]


class ExponentialLogPrior(pints.LogPrior):
    r"""
    Defines an exponential (log) prior with given rate parameter ``rate`` with
    pdf

    .. math::
        f(x|\text{rate}) = \text{rate} \; e^{-\text{rate}\;x}.

    A random variable :math:`X` distributed according to this pdf has
    expectation

    .. math::
        \mathrm{E}(X)=\frac{1}{\text{rate}}.

    For example, to create a prior with ``rate=0.5`` use::

        p = pints.ExponentialLogPrior(0.5)

    Extends :class:`LogPrior`.
    """

    def __init__(self, rate):
        # Parse input arguments
        self._rate = float(rate)

        # Validate inputs
        if self._rate <= 0:
            raise ValueError('Rate parameter "scale" must be positive')

        # Cache constant
        self._log_scale = np.log(self._rate)

    def __call__(self, x):
        if x[0] < 0.0:
            return -np.inf
        else:
            return self._log_scale - self._rate * x[0]

    def cdf(self, x):
        """ See :meth:`LogPrior.cdf()`. """
        return scipy.stats.expon.cdf(x, loc=0, scale=1.0 / self._rate)

    def icdf(self, p):
        """ See :meth:`LogPrior.icdf()`. """
        return scipy.stats.expon.ppf(p, loc=0, scale=1.0 / self._rate)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        value = self(x)

        if x[0] < 0.0:
            return value, np.asarray([0.])
        else:
            return value, np.asarray([-self._rate])

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return 1 / self._rate

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.exponential(scale=1. / self._rate, size=(n, 1))


class GammaLogPrior(pints.LogPrior):
    r"""
    Defines a gamma (log) prior with given shape parameter ``a`` and rate
    parameter ``b``, with pdf

    .. math::
        f(x|a,b)=\frac{b^a x^{a-1} e^{-bx}}{\mathrm{\Gamma}(a)}.

    where :math:`\Gamma` is the Gamma function.  A random variable :math:`X`
    distributed according to this pdf has expectation

    .. math::
        \mathrm{E}(X)=\frac{a}{b}.

    For example, to create a prior with shape parameters ``a=5`` and ``b=1``,
    use::

        p = pints.GammaLogPrior(5, 1)

    Extends :class:`LogPrior`.
    """

    def __init__(self, a, b):
        # Parse input arguments
        self._a = float(a)
        self._b = float(b)

        # Validate inputs
        if self._a <= 0:
            raise ValueError('Shape parameter a must be positive')
        if self._b <= 0:
            raise ValueError('Rate parameter b must be positive')

        # Cache constant
        self._constant = scipy.special.xlogy(self._a,
                                             self._b) - scipy.special.gammaln(
            self._a)

    def __call__(self, x):
        if x[0] < 0.0:
            return -np.inf
        else:
            return self._constant + scipy.special.xlogy(self._a - 1.,
                                                        x[0]) - self._b * x[0]

    def cdf(self, x):
        """ See :meth:`LogPrior.cdf()`. """
        return scipy.stats.gamma.cdf(x, a=self._a, loc=0, scale=1.0 / self._b)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        value = self(x)

        _x = x[0]

        # Account for pathological edge
        if _x == 0.0:
            _x = np.nextafter(0.0, 1.0)

        if _x < 0.0:
            return value, np.asarray([0.])
        else:
            # Use np.divide here to better handle possible v small denominators
            return value, np.asarray([np.divide(self._a - 1., _x) - self._b])

    def icdf(self, p):
        """ See :meth:`LogPrior.icdf()`. """
        return scipy.stats.gamma.ppf(p, a=self._a, loc=0, scale=1.0 / self._b)

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return self._a / self._b

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.gamma(self._a, 1. / self._b, size=(n, 1))


class GaussianLogPrior(pints.LogPrior):
    r"""
    Defines a 1-d Gaussian (log) prior with a given ``mean`` and
    standard deviation ``sd``, with pdf

    .. math::
        f(x|\text{mean},\text{sd}) = \frac{1}{\text{sd}\sqrt{2\pi}}
        \exp\left(-\frac{(x-\text{mean})^2}{2\;\text{sd}^2}\right).

    A random variable :math:`X` distributed according to this pdf has
    expectation

    .. math::
        \mathrm{E}(X)=\text{mean}.

    For example, to create a prior with mean of ``0`` and a standard deviation
    of ``1``, use::

        p = pints.GaussianLogPrior(0, 1)

    Extends :class:`LogPrior`.
    """

    def __init__(self, mean, sd):
        # Parse input arguments
        self._mean = float(mean)

        if sd <= 0:
            raise ValueError('sd parameter must be positive')
        self._sd = float(sd)

        # Cache constants
        self._offset = np.log(1 / np.sqrt(2 * np.pi * self._sd ** 2))
        self._factor = 1 / (2 * self._sd ** 2)
        self._factor2 = 1 / self._sd**2

    def __call__(self, x):
        return self._offset - self._factor * (x[0] - self._mean)**2

    def cdf(self, x):
        """ See :meth:`LogPrior.cdf()`. """
        return scipy.stats.norm.cdf(x, self._mean, self._sd)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        return self(x), self._factor2 * (self._mean - np.asarray(x))

    def icdf(self, p):
        """ See :meth:`LogPrior.icdf()`. """
        return scipy.stats.norm.ppf(p, self._mean, self._sd)

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return self._mean

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return np.random.normal(self._mean, self._sd, size=(n, 1))


class HalfCauchyLogPrior(pints.LogPrior):
    r"""
    Defines a 1-d half-Cauchy (log) prior with a given ``location`` and
    ``scale``. This is a Cauchy distribution that has been truncated to lie in
    between :math:`(0,\infty)`, with pdf

    .. math::
        f(x|\text{location},\text{scale})=\begin{cases}\frac{1}{\pi\;
        \text{scale}\left(\frac{1}{\pi}\arctan\left(\frac{\text{location}}
        {\text{scale} }\right)+\frac{1}{2}\right)\left(\frac{(x-\text{location}
        )^2}{\text{scale}^2}+1\right)},&x>0\\0,&\text{otherwise.}\end{cases}

    A random variable distributed according to this pdf has undefined
    expectation.

    For example, to create a prior centered around 0 and a scale of 5, use::

        p = pints.HalfCauchyLogPrior(0, 5)

    Extends :class:`LogPrior`.

    Parameters
    ----------
    location
        The center of the distribution.
    scale
        The scale of the distribution.
    """

    def __init__(self, location, scale):
        # Test inputs
        if float(scale) <= 0:
            raise ValueError('Scale must be positive')

        self._location = location
        self._scale = scale
        self._cauchy = pints.CauchyLogPrior(location=self._location,
                                            scale=self._scale)

        # Cache constants
        self._norm_factor = -np.log(np.arctan(location / scale) / np.pi + 0.5)
        self._arctan = np.arctan(self._location / self._scale) / np.pi

    def __call__(self, x):
        if x[0] > 0:
            return self._norm_factor + self._cauchy(x)
        else:
            return -np.inf

    def cdf(self, x):
        """ See :meth:`LogPrior.cdf()`. """
        return (
            (self._arctan +
             np.arctan((-self._location + x) / self._scale) / np.pi) /
            (0.5 + self._arctan))

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        value = self(x)
        scale = self._scale
        loc = self._location
        loc_minus_x = loc - x[0]
        dp = 2 * loc_minus_x / (scale**2 + loc_minus_x**2)
        return value, np.asarray([dp])

    def icdf(self, p):
        """ See :meth:`LogPrior.icdf()`. """
        return (self._location +
                self._scale * np.tan(np.pi * (-self._arctan +
                                              p * (0.5 + self._arctan))))

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return np.nan

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """

        # use inverse transform sampling
        us = np.random.uniform(0, 1, n)
        samples = np.array([self.icdf(u) for u in us])

        # Samples have shape (n,). Output needs to be (n, 1)
        return np.expand_dims(a=samples, axis=1)


class InverseGammaLogPrior(pints.LogPrior):
    r"""
    Defines an inverse gamma (log) prior with given shape parameter ``a`` and
    scale parameter ``b``, with pdf

    .. math::
        f(x|a,b)=\begin{cases}\frac{b^a}{\Gamma(a)}x^{-a-1}\exp
        \left(-\frac{b}{x}\right),&x>0\\0,&\text{otherwise.}\end{cases}

    where :math:`\Gamma` is the Gamma function.  A random variable :math:`X`
    distributed according to this pdf has expectation

    .. math::
        \mathrm{E}(X)=\begin{cases}\frac{b}{a-1},&a>1\\
        \text{undefined},&\text{otherwise.}\end{cases}

    For example, to create a prior with shape parameter ``a=5`` and scale
    parameter ``b=1``, use::

        p = pints.InverseGammaLogPrior(5, 1)

    Extends :class:`LogPrior`.
    """

    def __init__(self, a, b):
        # Parse input arguments
        self._a = float(a)
        self._b = float(b)

        # Validate inputs
        if self._a <= 0:
            raise ValueError('Shape parameter a must be positive')
        if self._b <= 0:
            raise ValueError('Scale parameter b must be positive')

        # Cache constants
        self._k = self._a * np.log(self._b) - scipy.special.gammaln(self._a)
        self._ap1 = self._a + 1.

    def __call__(self, x):
        _x = float(x[0])

        if _x <= 0.0:
            return -np.inf
        else:
            return self._k - self._ap1 * np.log(_x) - np.divide(self._b, _x)

    def cdf(self, x):
        """ See :meth:`LogPrior.cdf()`. """
        return scipy.stats.invgamma.cdf(x, a=self._a, loc=0, scale=self._b)

    def icdf(self, p):
        """ See :meth:`LogPrior.icdf()`. """
        return scipy.stats.invgamma.ppf(p, a=self._a, loc=0, scale=self._b)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        val = self(x)

        _x = float(x[0])

        if _x < 0.0:
            return val, np.asarray([0.])
        else:
            return val, np.asarray(
                [np.divide(self._b - self._ap1 * _x, _x * _x)])

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return self._b / (self._a - 1.) if self._a > 1 else np.nan

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return scipy.stats.invgamma.rvs(a=self._a, scale=self._b, loc=0.,
                                        size=(n, 1))


class LogNormalLogPrior(pints.LogPrior):
    r"""
    Defines a log-normal (log) prior with a given ``log_mean`` and scale
    ``scale``. The ``log_mean`` parameter of a log-normal distribution is the
    mean of a normal distribution whose random samples, when exponentiated,
    yield samples from a log-normal distribution. This log-normal distribution
    has pdf

    .. math::
        f(x|\text{log_mean},\text{scale}) = \frac{1}{x\;\text{scale}
        \sqrt{2\pi}}\exp\left(-\frac{(\log x-\text{log_mean})^2}{2\;
        \text{scale}^2}\right).

    A random variable :math:`X` distributed according to this pdf has
    expectation

    .. math::
        \mathrm{E}(X)=\exp\left(\text{log_mean}+\frac{\text{scale}^2}{2}
        \right).

    For example, to create a prior with log_mean of ``0`` and a scale of ``1``,
    use::

        p = pints.LogNormalLogPrior(0, 1)

    Extends :class:`LogPrior`.
    """

    def __init__(self, log_mean, scale):
        # Parse input arguments
        self._log_mean = float(log_mean)
        self._scale = float(scale)

        if self._scale <= 0:
            raise ValueError('Scale must be positive')

        # Cache constants
        self._offset = -np.log(self._scale * np.sqrt(2. * np.pi))
        self._1on2sigsq = 1. / (2. * self._scale * self._scale)
        self._m1onsigsq = -1. / (self._scale * self._scale)
        self._sigsqmmu = self._scale * self._scale - self._log_mean

    def __call__(self, x):
        if x[0] <= 0.0:
            return -np.inf
        else:
            _lx = np.log(x[0])
            _shift = _lx - self._log_mean
            return self._offset - _lx - self._1on2sigsq * _shift * _shift

    def cdf(self, x):
        """ See :meth:`LogPrior.cdf()`. """
        return scipy.stats.lognorm.cdf(x, scale=np.exp(self._log_mean),
                                       s=self._scale)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        if x[0] < 0.0:
            return self(x), np.asarray([0.])
        else:
            _x = x[0]
            _lx = np.log(_x)
            return self(x), np.asarray(
                [self._m1onsigsq * np.divide(self._sigsqmmu + _lx, _x)])

    def icdf(self, p):
        """ See :meth:`LogPrior.icdf()`. """
        return scipy.stats.lognorm.ppf(p, scale=np.exp(self._log_mean),
                                       s=self._scale)

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return np.exp(self._log_mean + 0.5 * self._scale * self._scale)

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return scipy.stats.lognorm.rvs(scale=np.exp(self._log_mean),
                                       s=self._scale, size=(n, 1))


class MultivariateGaussianLogPrior(pints.LogPrior):
    r"""
    Defines a multivariate Gaussian (log) prior with a given ``mean`` and
    covariance matrix ``cov``, with pdf

    .. math::
        f(x|\text{mean},\text{cov}) = \frac{1}{(2\pi)^{d/2}|
        \text{cov}|^{1/2}} \exp\left(-\frac{1}{2}(x-\text{mean})'
        \text{cov}^{-1}(x-\text{mean})\right).

    A random variable :math:`X` distributed according to this pdf has
    expectation

    .. math::
        \mathrm{E}(X)=\text{mean}.

    For example, to create a prior with zero mean and identity covariance,
    use::

        p = pints.MultivariateGaussianLogPrior(
                np.array([0, 0]), np.array([[1, 0],[0, 1]]))

    Extends :class:`LogPrior`.
    """

    def __init__(self, mean, cov):
        # Check input
        mean = pints.vector(mean)
        cov = np.array(cov, copy=True)
        if cov.ndim != 2:
            raise ValueError('Given covariance must be a matrix.')
        if not (mean.shape[0] == cov.shape[0] == cov.shape[1]):
            raise ValueError('Sizes of mean and covariance do not match.')

        # Store
        self._mean = mean
        self._cov = cov
        self._n_parameters = mean.shape[0]
        self._cov_inverse = np.linalg.inv(self._cov)
        self._cholesky_L, self._cholesky_lower = scipy.linalg.cho_factor(
            self._cov)
        log_det_cov = 2 * np.sum(np.log(self._cholesky_L.diagonal()))
        self._const_factor = - 0.5 * log_det_cov \
                             - 0.5 * len(self._mean) * np.log(2 * np.pi)

        # Factors needed for pseudo-cdf calculation
        self._sigma12_sigma22_inv_l = []
        self._sigma_bar_l = []
        self._mu1 = []
        self._mu2 = []
        # note the below does not do anything for index 1 since the first
        # distribution is just a simple marginal
        for j in range(1, self._n_parameters):
            sigma = self._cov[0:(j + 1), 0:(j + 1)]
            dims = sigma.shape
            sigma11 = sigma[dims[0] - 1, dims[1] - 1]
            sigma22 = sigma[0:(dims[0] - 1), 0:(dims[0] - 1)]
            sigma12 = sigma[dims[0] - 1, 0:(dims[0] - 1)]
            sigma21 = sigma[0:(dims[0] - 1), dims[0] - 1]
            mean = self._mean[0:dims[0]]
            mu2 = mean[0:(dims[0] - 1)]
            mu1 = mean[dims[0] - 1]
            sigma12_sigma22_inv = np.matmul(sigma12,
                                            np.linalg.inv(sigma22))
            sigma_bar = np.sqrt(sigma11 - np.matmul(sigma12_sigma22_inv,
                                                    sigma21))
            self._sigma12_sigma22_inv_l.append(sigma12_sigma22_inv)
            self._sigma_bar_l.append(sigma_bar)
            self._mu1.append(mu1)
            self._mu2.append(mu2)

    def __call__(self, x):
        tmp = x - self._mean
        return self._const_factor \
            - 0.5 * tmp.dot(
                scipy.linalg.cho_solve(
                    (self._cholesky_L, self._cholesky_lower), tmp
                )
            )

    def convert_from_unit_cube(self, u):
        """
        Converts a sample ``u`` uniformly drawn from the unit cube into one
        drawn from the prior space, using
        :meth:`MultivariateGaussianLogPrior.pseudo_icdf()`.
        """
        return self.pseudo_icdf(u)

    def convert_to_unit_cube(self, x):
        """
        Converts a sample from the prior ``x`` to be drawn uniformly from the
        unit cube using :meth:`MultivariateGaussianLogPrior.pseudo_cdf()`.
        """
        return self.pseudo_cdf(x)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        return self(x), -scipy.linalg.cho_solve(
            (self._cholesky_L, self._cholesky_lower), x - self._mean
        )

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return self._mean

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._n_parameters

    def pseudo_cdf(self, xs):
        r"""
        Calculates a pseudo-cdf for a multivariate Gaussian as described in
        Feroz et al. (2009) ("Multnest..."). In this approach, a multivariate
        Gaussian is factorised:

        .. math::
            \pi(\theta_1,\theta_2,...,\theta_d) = \pi_1(\theta_1)
                \pi_2(\theta_2|\theta_1)...
                \pi_d(\theta_d|\theta_1, \theta_2,...,\theta_{d-1})

        The cdfs we report are then the values for each individual conditional.
        For example, for the second component, we calculate:

        .. math::
            u_2 = \int_{-\infty}^{\theta_2} \pi_2(\theta_2|\theta_1)d\theta_2

        So that we return a vector of cdfs (u_1,u_2,...,u_d).
        Note that, this function is mainly to facilitate MultiNest sampling
        since the distribution (u_1,u_2,...,u_d) is uniform within the unit
        cube.
        """
        if not isinstance(xs, np.ndarray):
            if not isinstance(xs, list):
                xs = [xs]
            if any(isinstance(a, list) for a in xs):
                xs = np.array(xs)
            else:
                xs = np.array([xs])
            n_samples = xs.shape[0]
            n_params = xs.shape[1]
        else:
            if len(xs.shape) == 1:
                n_params = xs.shape[0]
                n_samples = 1
                xs = np.reshape(xs, (n_samples, n_params))
        if n_params != self._n_parameters:
            raise ValueError(
                "Dimensions of samples must match prior dimensions.")
        cdfs = np.zeros((n_samples, n_params))
        for j in range(n_samples):
            for i in range(self._n_parameters):
                if i == 0:
                    mu = self._mean[0]
                    sigma = np.sqrt(self._cov[0, 0])
                else:
                    sigma = self._sigma_bar_l[i - 1]
                    mu = self._mu1[i - 1] + np.matmul(
                        self._sigma12_sigma22_inv_l[i - 1],
                        (xs[j, 0:i] - self._mu2[i - 1]))
                cdfs[j, i] = scipy.stats.norm.cdf(xs[j, i], mu, sigma)
        if n_samples == 1:
            return cdfs[0]
        else:
            return cdfs

    def pseudo_icdf(self, ps):
        r"""
        Calculates a pseudo-icdf for a multivariate Gaussian as described in
        Feroz et al. (2009) ("Multnest..."). In this approach, a multivariate
        Gaussian is factorised:

        .. math::
            \pi(\theta_1,\theta_2,...,\theta_d) = \pi_1(\theta_1)
                \pi_2(\theta_2|\theta_1)...
                \pi_d(\theta_d|\theta_1, \theta_2,...,\theta_{d-1})

        The icdfs we report are then the values for each individual
        conditional. For example, for the second component, we calculate the
        theta_2 value that satisfies:

        .. math::
            u_2 = \int_{-\infty}^{\theta_2} \pi_2(\theta_2|\theta_1)d\theta_2

        So that we return a vector of icdfs (theta_1,theta_2,...,theta_d)
        Note that, this function is mainly to facilitate MultiNest sampling
        since the distribution (u_1,u_2,...,u_d) is uniform within the unit
        cube.
        """
        if not isinstance(ps, np.ndarray):
            if not isinstance(ps, list):
                ps = [ps]
            if any(isinstance(a, list) for a in ps):
                ps = np.array(ps)
            else:
                ps = np.array([ps])
            n_samples = ps.shape[0]
            n_params = ps.shape[1]
        else:
            if len(ps.shape) == 1:
                n_params = ps.shape[0]
                n_samples = 1
                ps = np.reshape(ps, (n_samples, n_params))
        if n_params != self._n_parameters:
            raise ValueError(
                "Dimensions of samples must match prior dimensions.")
        icdfs = np.zeros((n_samples, n_params))
        for j in range(n_samples):
            for i in range(self._n_parameters):
                if i == 0:
                    mu = self._mean[0]
                    sigma = np.sqrt(self._cov[0, 0])
                else:
                    sigma = self._sigma_bar_l[i - 1]
                    mu = self._mu1[i - 1] + np.matmul(
                        self._sigma12_sigma22_inv_l[i - 1],
                        (np.array(icdfs[j, 0:i]) - self._mu2[i - 1]))
                icdfs[j, i] = scipy.stats.norm.ppf(ps[j, i], mu, sigma)
        if n_samples == 1:
            return icdfs[0]
        else:
            return icdfs

    def sample(self, n=1):
        """ See :meth:`LogPrior.call()`. """
        # Note: size=n returns shape (n, d)
        return np.random.multivariate_normal(
            self._mean, self._cov, size=n)


class NormalLogPrior(GaussianLogPrior):
    r""" Deprecated alias of :class:`GaussianLogPrior`. """

    def __init__(self, mean, standard_deviation):
        # Deprecated on 2019-02-06
        import warnings
        warnings.warn(
            'The class `pints.NormalLogPrior` is deprecated.'
            ' Please use `pints.GaussianLogPrior` instead.')
        super(NormalLogPrior, self).__init__(mean, standard_deviation)


class StudentTLogPrior(pints.LogPrior):
    r"""
    Defines a 1-d Student-t (log) prior with a given ``location``,
    degrees of freedom ``df``,  and ``scale`` with pdf

    .. math::
        f(x|\text{location},\text{scale},\text{df})=\frac{\left(\frac{
        \text{df}}{\text{df}+\frac{(x-\text{location})^2}{\text{scale}^2}}
        \right)^{\frac{\text{df}+1}{2}}}{\sqrt{\text{df}}\;\text{scale}
        \;\mathrm{B}\left(\frac{\text{df} }{2},\frac{1}{2}\right)}.

    where :math:`\mathrm{B}` is the Beta function. A random variable :math:`X`
    distributed according to this pdf has expectation

    .. math::
        \mathrm{E}(X)=\begin{cases}\text{location},&\text{df}>1\\\
        \text{undefined},&\text{otherwise.}\end{cases}

    For example, to create a prior centered around 0 with 3 degrees of freedom
    and a scale of 1, use::

        p = pints.StudentTLogPrior(0, 3, 1)

    Extends :class:`LogPrior`.

    Parameters
    ----------
    location
        The center of the distribution.
    df : int
        The number of degrees of freedom of the distribution.
    scale
        The scale of the distribution.
    """

    def __init__(self, location, df, scale):
        # Test inputs
        if float(df) <= 0:
            raise ValueError('Degrees of freedom must be positive')
        if float(scale) <= 0:
            raise ValueError('Scale must be positive')

        # Parse input arguments
        self._df = float(df)
        self._location = float(location)
        self._scale = float(scale)

        # Cache constants
        self._log_df = np.log(self._df)

        self._1_sig_sq = 1. / (self._scale * self._scale)

        self._first = 0.5 * (1.0 + self._df)

        self._samp_const = scipy.special.xlogy(-0.5, self._df) - np.log(
            self._scale) - scipy.special.betaln(0.5 * self._df, 0.5)

        self._deriv_const = (-1. - self._df) * self._1_sig_sq

    def __call__(self, x):
        return self._samp_const + self._first * (self._log_df - np.log(
            self._df + self._1_sig_sq * (x[0] - self._location) ** 2))

    def cdf(self, x):
        """ See :meth:`LogPrior.cdf()`. """
        return scipy.stats.t.cdf(x, self._df, self._location, self._scale)

    def icdf(self, p):
        """ See :meth:`LogPrior.icdf()`. """
        return scipy.stats.t.ppf(p, self._df, self._location, self._scale)

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        offset = x[0] - self._location
        return self(x), np.asarray([offset * self._deriv_const / (
            self._df + offset * offset * self._1_sig_sq)])

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return self._location if self._df > 1. else np.nan

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return scipy.stats.t.rvs(df=self._df, loc=self._location,
                                 scale=self._scale, size=(n, 1))


class TruncatedGaussianLogPrior(pints.LogPrior):
    r"""
    Defines a truncated Gaussian log prior.

    This distribution is also known as the truncated Normal distribution.

    The truncated Gaussian distribution is similar to the Gaussian
    distribution, but constrained to lie between two values.

    The parameters are the mean ``mean`` and standard deviation ``sd``, as in
    the Gaussian distribution, as well as a lower bound ``a`` and an upper
    bound ``b``.

    The pdf of the truncated Gaussian distribution is given by

    .. math::
        f(x|\mu, \sigma, a, b) = \frac{1}{\sigma\sqrt{2\pi}} \exp
        \left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \frac{1}
            {\Phi((b-\mu) / \sigma) - \Phi((a-\mu) / \sigma)}

    for :math:`x \in [a, b]`, where :math:`\mu` indicates the mean and
    :math:`\sigma` indicates the standard deviation, and :math:`\Phi` is the
    standard normal CDF.

    For example, to create a prior with mean of 0 and a standard deviation of
    1, bounded above at 3 and below at -2, use::

        p = pints.TruncatedGaussianLogPrior(0, 1, -2, 3)

    For a Gaussian distribution truncated on only one side, ``numpy.inf`` or
    ``-numpy.inf`` can be used for the unbounded side.

    Extends :class:`LogPrior`.
    """

    def __init__(self, mean, sd, a, b):
        # Parse input arguments
        self._mean = float(mean)
        self._sd = float(sd)
        self._a = float(a)
        self._b = float(b)
        if b <= a:
            raise ValueError('Upper bound must exceed lower bound.')

        # Convert the upper and lower truncation levels to the Scipy definition
        self._lower = (a - self._mean) / self._sd
        self._upper = (b - self._mean) / self._sd

        # Cache constants
        self._factor2 = 1 / self._sd**2

    def __call__(self, x):
        return scipy.stats.truncnorm.logpdf(
            x[0],
            self._lower,
            self._upper,
            loc=self._mean,
            scale=self._sd
        )

    def cdf(self, x):
        """ See :meth:`LogPrior.cdf()`. """
        return scipy.stats.truncnorm.cdf(
            x,
            self._lower,
            self._upper,
            loc=self._mean,
            scale=self._sd
        )

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        dp = self._factor2 * (self._mean - np.asarray(x))

        # Set values outside limits to nan
        dp[(np.asarray(x) < self._a) | (np.asarray(x) > self._b)] = np.nan

        return self(x), dp

    def icdf(self, p):
        """ See :meth:`LogPrior.icdf()`. """
        return scipy.stats.truncnorm.ppf(
            p,
            self._lower,
            self._upper,
            loc=self._mean,
            scale=self._sd
        )

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        return scipy.stats.truncnorm.stats(
            self._lower,
            self._upper,
            loc=self._mean,
            scale=self._sd,
            moments='m'
        )

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return 1

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return scipy.stats.truncnorm.rvs(
            self._lower,
            self._upper,
            loc=self._mean,
            scale=self._sd,
            size=(n, 1)
        )


class UniformLogPrior(pints.LogPrior):
    r"""
    Defines a uniform prior over a given range.

    The range includes the lower, but not the upper boundaries, so that any
    point ``x`` with a non-zero prior must have ``lower <= x < upper``.

    In 1D this has pdf

    .. math::
        f(x|\text{lower},\text{upper})=\begin{cases}0,&\text{if }x\not\in
        [\text{lower},\text{upper})\\\frac{1}{\text{upper}-\text{lower}}
        ,&\text{if }x\in[\text{lower},\text{upper})\end{cases}.

    A random variable :math:`X` distributed according to this pdf has
    expectation

    .. math::
        \mathrm{E}(X)=\frac{1}{2}(\text{lower}+\text{upper}).

    For example, to create a prior with :math:`x\in[0,4]`, :math:`y\in[1,5]`,
    and :math:`z\in[2,6]` use either::

        p = pints.UniformLogPrior([0, 1, 2], [4, 5, 6])

    or::

        p = pints.UniformLogPrior(RectangularBoundaries([0, 1, 2], [4, 5, 6]))

    Extends :class:`LogPrior`.
    """

    def __init__(self, lower_or_boundaries, upper=None):
        # Parse input arguments
        if upper is None:
            if not isinstance(lower_or_boundaries, pints.Boundaries):
                raise ValueError(
                    'UniformLogPrior requires a lower and an upper bound, or a'
                    ' single Boundaries object.')
            self._boundaries = lower_or_boundaries
        else:
            self._boundaries = pints.RectangularBoundaries(
                lower_or_boundaries, upper)

        # Cache dimension
        self._n_parameters = self._boundaries.n_parameters()

        # Maximum output value
        # Use normalised value (1/area) for rectangular boundaries,
        # otherwise just use 1.
        if isinstance(self._boundaries, pints.RectangularBoundaries):
            self._value = -np.log(np.product(self._boundaries.range()))
        else:
            self._value = 1

    def __call__(self, x):
        return self._value if self._boundaries.check(x) else -np.inf

    def cdf(self, xs):
        """ See :meth:`LogPrior.cdf()`. """
        if not isinstance(xs, np.ndarray):
            if not isinstance(xs, list):
                xs = [xs]
            if any(isinstance(a, list) for a in xs):
                xs = np.array(xs)
            else:
                xs = np.array([xs])
            n_samples = xs.shape[0]
            n_params = xs.shape[1]
        else:
            if len(xs.shape) == 1:
                n_params = xs.shape[0]
                n_samples = 1
                xs = np.reshape(xs, (n_samples, n_params))
        if n_params != self._n_parameters:
            raise ValueError(
                "Dimensions of samples must match prior dimensions.")
        cdfs = np.zeros((n_samples, n_params))
        for j in range(n_samples):
            for i in range(n_params):
                if xs[j, i] > self._boundaries.lower()[i] and (
                        xs[j, i] < self._boundaries.upper()[i]):
                    cdfs[j, i] = ((-self._boundaries.lower()[i] + xs[j, i]) /
                                  (-self._boundaries.lower()[i] +
                                   self._boundaries.upper()[i]))
                elif xs[j, i] >= self._boundaries.upper()[i]:
                    cdfs[j, i] = 1.0
                else:
                    cdfs[j, i] = 0.0
        if n_samples == 1:
            return cdfs[0]
        else:
            return cdfs

    def icdf(self, ps):
        """ See :meth:`LogPrior.icdf()`. """
        if not isinstance(ps, np.ndarray):
            if not isinstance(ps, list):
                ps = [ps]
            if any(isinstance(a, list) for a in ps):
                ps = np.array(ps)
            else:
                ps = np.array([ps])
            n_samples = ps.shape[0]
            n_params = ps.shape[1]
        else:
            if len(ps.shape) == 1:
                n_params = ps.shape[0]
                n_samples = 1
                ps = np.reshape(ps, (n_samples, n_params))
        if n_params != self._n_parameters:
            raise ValueError(
                "Dimensions of samples must match prior dimensions.")
        icdfs = np.zeros((n_samples, n_params))
        for j in range(n_samples):
            for i in range(n_params):
                if ps[j, i] > 0 and ps[j, i] < 1:
                    icdfs[j, i] = (
                        self._boundaries.lower()[i] * (1 - ps[j, i]) +
                        self._boundaries.upper()[i] * ps[j, i])
                elif ps[j, i] <= 0:
                    icdfs[j, i] = self._boundaries.lower()[i]
                else:
                    icdfs[j, i] = self._boundaries.upper()[i]
        if n_samples == 1:
            return icdfs[0]
        else:
            return icdfs

    def evaluateS1(self, x):
        """ See :meth:`LogPrior.evaluateS1()`. """
        # Ignoring points on the boundaries (i.e. on the surface of the
        # hypercube), because it's very unlikely and won't help the search
        # much...
        return self(x), np.zeros(self._n_parameters)

    def mean(self):
        """ See :meth:`LogPrior.mean()`. """
        if isinstance(self._boundaries, pints.RectangularBoundaries):
            return 0.5 * (self._boundaries.lower() + self._boundaries.upper())
        else:
            raise NotImplementedError

    def n_parameters(self):
        """ See :meth:`LogPrior.n_parameters()`. """
        return self._n_parameters

    def sample(self, n=1):
        """ See :meth:`LogPrior.sample()`. """
        return self._boundaries.sample(n)

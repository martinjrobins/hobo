#
# Transformation functions
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
from scipy.special import logit, expit


class Transform(object):
    """
    Abstract base class for objects that provide some convenience parameter
    transformation from the model parameter space to a search space.

    If ``t`` is an instance of a ``Transform`` class, you can apply
    the transformation of a parameter vector from the model space ``p`` to the
    search space ``x`` by using ``x = t.to_search(p)`` and the inverse by using
    ``p = t.to_model(x)``.
    """
    def apply_log_pdf(self, log_pdf):
        """
        Returns a transformed log-PDF class.
        """
        return TransformedLogPDF(log_pdf, self)

    def apply_log_prior(self, log_prior):
        """
        Returns a transformed log-prior class.
        """
        return TransformedLogPrior(log_prior, self)

    def apply_error_measure(self, error_measure):
        """
        Returns a transformed error measure class.
        """
        return TransformedErrorMeasure(error_measure, self)

    def apply_boundaries(self, boundaries):
        """
        Returns a transformed boundaries class.
        """
        return TransformedBoundaries(boundaries, self)

    def jacobian(self, x):
        """
        Returns the Jacobian for a parameter vector ``x`` in the search space.

        *This is an optional method. It is needed when `sigma0` is provided in
        :meth:`pints.OptimisationController` or :meth:`pints.MCMCController`,
        or when the method requires ``evaluateS1()``.*
        """
        raise NotImplementedError

    def log_jacobian_det(self, x):
        """
        Returns the logarithm of the absolute value of the Jacobian
        determinant for a parameter vector ``x`` in the search space.

        *This is an optional method. It is needed when transformation is
        performed on :class:`LogPDF` and/or that requires ``evaluateS1()``;
        e.g. not necessary if it's used for :class:`ErrorMeasure` without
        :meth:`ErrorMeasure.evaluateS1()`.*
        """
        return np.log(np.abs(np.linalg.det(self.jacobian(x))))

    def log_jacobian_det_S1(self, x):
        """
        Computes the logarithm of the absolute value of the determinant of the
        Jacobian, and returns the result plus the partial derivatives of the
        result with respect to the parameters.

        The returned data is a tuple ``(S, S')`` where ``S`` is a scalar value
        and ``S'`` is a sequence of length ``n_parameters``.

        Note that the derivative returned is of the log of the determinant of
        the Jacobian, so ``S' = d/dq log(|det(J(q))|)``, evaluated at ``q=x``.

        *This is an optional method.*
        """
        raise NotImplementedError

    def n_parameters(self):
        """
        Returns the dimension of the parameter space this transformation is
        defined over.
        """
        raise NotImplementedError

    def to_model(self, x):
        """
        Returns the inverse of transformation of a vector from the search space
        ``x`` to the model parameter space ``p``.
        """
        raise NotImplementedError

    def to_search(self, p):
        """
        Returns the forward transformation of a vector from the model parameter
        space ``p`` to the search space ``x``.
        """
        raise NotImplementedError


class TransformedLogPDF(pints.LogPDF):
    """
    A log-PDF that is transformed from the model space to the search space.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    log_pdf
        A :class:`pints.LogPDF`.
    transform
        A :class:`pints.Transform`.
    """
    def __init__(self, log_pdf, transform):
        self._log_pdf = log_pdf
        self._transform = transform
        self._n_parameters = self._log_pdf.n_parameters()
        if self._transform.n_parameters() != self._n_parameters:
            raise ValueError('Number of parameters for log_pdf and transform '
                             'must match.')

    def __call__(self, x):
        # Get parameters in the model space
        p = self._transform.to_model(x)

        # Compute LogPDF in the model space
        logpdf_nojac = self._log_pdf(p)

        # Calculate the PDF change of variable. From Wikipedia:
        # https://w.wiki/UsJ
        #
        # For some model parameter vector b and distribution p(b), consider a
        # transformation a = f(b), then
        #
        # p(a) = p(f^{-1}(a)) |det(J(f^{-1}(a)))|
        #
        # where J is the Jacobian matrix. For log PDF, it becomes
        #
        # log(p(a)) = log(p(f^{-1}(a))) + log(|det(J(f^{-1}(a)))|)
        #
        log_jacobian_det = self._transform.log_jacobian_det(x)
        return logpdf_nojac + log_jacobian_det

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """
        # Get parameters in the model space
        p = self._transform.to_model(x)

        # Compute evaluateS1 of LogPDF in the model space
        logpdf_nojac, dlogpdf_nojac = self._log_pdf.evaluateS1(p)

        # Compute log Jacobian and its derivatives
        logjacdet, dlogjacdet = self._transform.log_jacobian_det_S1(x)

        # Calculate the PDF change of variable, see self.__call__()
        logpdf = logpdf_nojac + logjacdet

        # Calculate the PDF S1 change of variable.
        #
        # For some transformation x = h(p) and its inverse p = g(x) applied
        # to some distribution pi(p) where pi(x) = pi(p) |det(J)|, then
        #
        # d/dx_i log(pi(x)) = d/dx_i log(|det(J)|) + d/dx_i log(pi(p))
        #
        # Applying the chain rule (Wikipedia: https://w.wiki/Us8) to the second
        # term d/dx_i log(pi(p))
        #
        # d/dx_i log(pi(p)) = \sum_l (d/dp_l log(pi(p))) * dp_l/dx_i
        # (d denotes partial derivative)
        #
        # Or in matrix form, for Jacobian matrix J and E = log(pi(p))
        #
        # (dEdx)^T = J_(E.g) = J_E(p) J_g = (dEdp)^T J_g
        #
        jacobian = self._transform.jacobian(x)
        dlogpdf = np.matmul(dlogpdf_nojac, jacobian)  # Jacobian must be second
        dlogpdf += pints.vector(dlogjacdet)

        return logpdf, dlogpdf

    def n_parameters(self):
        """ See :meth:`LogPDF.n_parameters()`. """
        return self._n_parameters


class TransformedLogPrior(TransformedLogPDF, pints.LogPrior):
    """
    A log-prior that is transformed from the model space to the search space.

    Extends :class:`pints.LogPrior`, :class:`pints.TransformedLogPDF`.
    """
    def sample(self, n):
        """ See :meth:`LogPrior.sample()`. """
        ps = self._log_pdf.sample(n)
        xs = np.zeros(ps.shape)
        for i, p in enumerate(ps):
            xs[i, :] = self._transform.to_search(p)
        return xs


class TransformedErrorMeasure(pints.ErrorMeasure):
    """
    An error measure that is transformed from the model space to the search
    space.

    Extends :class:`pints.ErrorMeasure`.

    Parameters
    ----------
    error
        A :class:`pints.ErrorMeasure`.
    transform
        A :class:`pints.Transform`.
    """
    def __init__(self, error, transform):
        self._error = error
        self._transform = transform
        self._n_parameters = self._error.n_parameters()
        if self._transform.n_parameters() != self._n_parameters:
            raise ValueError('Number of parameters for error and transform '
                             'must match.')

    def __call__(self, x):
        # Get parameters in the model space
        p = self._transform.to_model(x)
        # Compute ErrorMeasure in the model space
        return self._error(p)

    def evaluateS1(self, x):
        """ See :meth:`ErrorMeasure.evaluateS1()`. """

        # Get parameters in the model space
        p = self._transform.to_model(x)

        # Compute evaluateS1 of ErrorMeasure in the model space
        e, de_nojac = self._error.evaluateS1(p)

        # Calculate the S1 change of variable. From Chain Rule Wikipedia:
        # https://w.wiki/Us8
        #
        # For some transformation x = h(p) and its inverse p = g(x) applied
        # to some function E
        #
        # dE/dx_i = \sum_l dE/dp_l * dp_l/dx_i  (d denotes partial derivative)
        #
        # Or in matrix form, for Jacobian matrix J
        #
        # (dEdx)^T = J_(E.g) = J_E(p) J_g = (dEdp)^T J_g
        #
        jacobian = self._transform.jacobian(x)
        de = np.matmul(de_nojac, jacobian)  # Jacobian must be the second term

        return e, de

    def n_parameters(self):
        """ See :meth:`ErrorMeasure.n_parameters()`. """
        return self._n_parameters


class TransformedBoundaries(pints.Boundaries):
    """
    Boundaries that are transformed from the model space to the search space.

    Extends :class:`pints.Boundaries`.

    Parameters
    ----------
    boundaries
        A :class:`pints.Boundaries`.
    transform
        A :class:`pints.Transform`.
    """
    def __init__(self, boundaries, transform):
        self._boundaries = boundaries
        self._transform = transform
        self._n_parameters = self._boundaries.n_parameters()

        if self._transform.n_parameters() != self._n_parameters:
            raise ValueError('Number of parameters for boundaries and '
                             'transform must match.')

    def check(self, x):
        """ See :meth:`Boundaries.check()`. """
        # Get parameters in the model space
        p = self._transform.to_model(x)
        # Check Boundaries in the model space
        return self._boundaries.check(p)

    def n_parameters(self):
        """ See :meth:`Boundaries.n_parameters()`. """
        return self._n_parameters

    def range(self):
        """
        Returns the size of the search space (i.e. ``upper - lower``).
        """
        upper = self._transform.to_search(self._boundaries.upper())
        lower = self._transform.to_search(self._boundaries.lower())
        return upper - lower


class LogTransform(Transform):
    r"""
    Logarithm transformation of the model parameters:

    .. math::
        x = \log(p),

    where :math:`p` is the model parameter vector and :math:`x` is the
    search space vector.

    The Jacobian adjustment of the log transformation is given by

    .. math::
        |\frac{d}{dx} \exp(x)| = \exp(x).

    The first order derivative of the log determinant of the Jacobian is

    .. math::
        \frac{d}{dx} \log(|J(x)|) = 1.

    Extends :class:`Transform`.

    Parameters
    ----------
    n_parameters
        Number of model parameters this transformation is defined over.
    """
    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        x = pints.vector(x)
        return np.diag(np.exp(x))

    def log_jacobian_det(self, x):
        """ See :meth:`Transform.log_jacobian_det()`. """
        x = pints.vector(x)
        return np.sum(x)

    def log_jacobian_det_S1(self, x):
        """ See :meth:`Transform.log_jacobian_det_S1()`. """
        x = pints.vector(x)
        logjacdet = self.log_jacobian_det(x)
        dlogjacdet = np.ones(self._n_parameters)
        return logjacdet, dlogjacdet

    def n_parameters(self):
        """ See :meth:`Transform.n_parameters()`. """
        return self._n_parameters

    def to_model(self, x):
        """ See :meth:`Transform.to_model()`. """
        x = pints.vector(x)
        return np.exp(x)

    def to_search(self, p):
        """ See :meth:`Transform.to_search()`. """
        p = pints.vector(p)
        return np.log(p)


class LogitTransform(Transform):
    r"""
    Logit (or log-odds) transformation of the model parameters:

    .. math::
        x = \text{logit}(p) = \log(\frac{p}{1 - p}),

    where :math:`p` is the model parameter vector and :math:`x` is the
    search space vector.

    The Jacobian adjustment of the logit transformation is given by

    .. math::
        |\frac{d}{dx} \text{logit}^{-1}(x)| = \text{logit}^{-1}(x) \times
        (1 - \text{logit}^{-1}(x)).

    The first order derivative of the log determinant of the Jacobian is

    .. math::
        \frac{d}{dx} \log(|J(x)|) = 2 \exp(-x) \text{logit}^{-1}(x) - 1.

    Extends :class:`Transform`.

    Parameters
    ----------
    n_parameters
        Number of model parameters this transformation is defined over.
    """
    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        x = pints.vector(x)
        return np.diag(expit(x) * (1. - expit(x)))

    def log_jacobian_det(self, x):
        """ See :meth:`Transform.log_jacobian_det()`. """
        x = pints.vector(x)
        return np.sum(np.log(expit(x)) + np.log(1. - expit(x)))

    def log_jacobian_det_S1(self, x):
        """ See :meth:`Transform.log_jacobian_det_S1()`. """
        x = pints.vector(x)
        logjacdet = self.log_jacobian_det(x)
        dlogjacdet = 2. * np.exp(-x) * expit(x) - 1.
        return logjacdet, dlogjacdet

    def n_parameters(self):
        """ See :meth:`Transform.n_parameters()`. """
        return self._n_parameters

    def to_model(self, x):
        """ See :meth:`Transform.to_model()`. """
        x = pints.vector(x)
        return expit(x)

    def to_search(self, p):
        """ See :meth:`Transform.to_search()`. """
        p = pints.vector(p)
        return logit(p)


class RectangularBoundariesTransform(Transform):
    r"""
    A generalised version of logit transformation for the model parameters,
    which transform an interval or ractangular boundaries :math:`[a, b)` to
    all real number:

    .. math::
        x = f(p) = \log(p - a) - \log(b - p),

    where :math:`p` is the model parameter vector and :math:`x` is the
    search space vector. The range includes the lower (:math:`a`), but not the
    upper (:math:`b`) boundaries. Note that :class:`LogitTransform` is a
    special case where :math:`a = 0` and :math:`b = 1`.

    The Jacobian adjustment of the transformation is given by

    .. math::
        |\frac{d}{dx} f^{-1}(x)| = \frac{b - a}{\exp(x) (1 + \exp(-x)) ^ 2}
        \log|\frac{d}{dx} f^{-1}(x)| = \log(b - a) - 2 \log(1 + \exp(-x)) - x

    The first order derivative of the log determinant of the Jacobian is

    .. math::
        \frac{d}{dx} \log(|J(x)|) = 2 \exp(-x) \text{logit}^{-1}(x) - 1.

    For example, to create a transform with :math:`p_1 \in [0, 4)`,
    :math:`p_2 \in [1, 5)`, and :math:`p_3 \in [2, 6)` use either::

        transform = pints.IntervalTransform([0, 1, 2], [4, 5, 6])

    or::

        boundaries = pints.RectangularBoundaries([0, 1, 2], [4, 5, 6])
        transform = pints.IntervalTransform(boundaries)

    Extends :class:`Transform`.
    """
    def __init__(self, lower_or_boundaries, upper=None):
        # Parse input arguments
        if upper is None:
            if not isinstance(lower_or_boundaries,
                              pints.RectangularBoundaries):
                raise ValueError(
                    'IntervalTransform requires a lower and an upper bound, '
                    'or a single RectangularBoundaries object.')
            boundaries = lower_or_boundaries
        else:
            # Create RectangularBoundaries for all the input checks
            boundaries = pints.RectangularBoundaries(lower_or_boundaries,
                                                     upper)

        self._a = boundaries.lower()
        self._b = boundaries.upper()

        # Cache dimension
        self._n_parameters = boundaries.n_parameters()
        del(boundaries)

    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        x = pints.vector(x)
        diag = (self._b - self._a) / (np.exp(x) * (1. + np.exp(-x)) ** 2)
        return np.diag(diag)

    def log_jacobian_det(self, x):
        """ See :meth:`Transform.log_jacobian_det()`. """
        x = pints.vector(x)
        s = self._softplus(-x)
        return np.sum(np.log(self._b - self._a) - 2. * s - x)

    def log_jacobian_det_S1(self, x):
        """ See :meth:`Transform.log_jacobian_det_S1()`. """
        x = pints.vector(x)
        logjacdet = self.log_jacobian_det(x)
        dlogjacdet = 2. * np.exp(-x) * expit(x) - 1.
        return logjacdet, dlogjacdet

    def n_parameters(self):
        """ See :meth:`Transform.n_parameters()`. """
        return self._n_parameters

    def to_model(self, x):
        """ See :meth:`Transform.to_model()`. """
        x = pints.vector(x)
        return (self._b - self._a) * expit(x) + self._a

    def to_search(self, p):
        p = pints.vector(p)
        """ See :meth:`Transform.to_search()`. """
        return np.log(p - self._a) - np.log(self._b - p)

    def _softplus(self, x):
        """ Returns the softplus function. """
        return np.log(1. + np.exp(x))


class IdentityTransform(Transform):
    """
    Identity transformation does nothing to the input parameters, i.e. the
    search space under this transformation is the same as the model space.
    And its Jacobian matrix is the identity matrix.

    Extends :class:`Transform`.

    Parameters
    ----------
    n_parameters
        Number of model parameters this transformation is defined over.
    """
    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        return np.eye(self._n_parameters)

    def log_jacobian_det_S1(self, x):
        """ See :meth:`Transform.log_jacobian_det_S1()`. """
        return self.log_jacobian_det(x), np.zeros(self._n_parameters)

    def n_parameters(self):
        """ See :meth:`Transform.n_parameters()`. """
        return self._n_parameters

    def to_model(self, x):
        """ See :meth:`Transform.to_model()`. """
        return pints.vector(x)

    def to_search(self, p):
        """ See :meth:`Transform.to_search()`. """
        return pints.vector(p)


class ComposedTransform(Transform):
    r"""
    N-dimensional :class:`Transform` composed of one or more other
    :math:`N_i`-dimensional ``Transform``, such that :math:`\sum _i N_i = N`.
    The evaluation and transformation of the composed transformations assume
    the input transformations are all independent from each other.

    For example, a composed transform

        ``t = pints.ComposedTransform(transform1, transform2, transform3)``,

    where ``transform1``, ``transform2``, and ``transform3`` each have
    dimension 1, 2 and 1, will have dimension 4.

    The dimensionality of the individual priors does not have to be the same,
    i.e. :math:`N_i\neq N_j` is allowed.

    The input parameters of the :class:`ComposedTransform` have to be ordered
    in the same way as the individual tranforms for the parameter vector. In
    the above example the transform may be performed by ``t.to_search(p)``,
    where:

        ``p = [parameter1_transform1, parameter1_transform2,
        parameter2_transform2, parameter1_transform3]``.

    Extends :class:`Transform`.
    """
    def __init__(self, *transforms):
        # Check if sub-transforms given
        if len(transforms) < 1:
            raise ValueError('Must have at least one sub-transform.')

        # Check if proper transform, count dimension
        self._n_parameters = 0
        for transform in transforms:
            if not isinstance(transform, pints.Transform):
                raise ValueError('All sub-transforms must extend '
                                 'pints.Transform.')
            self._n_parameters += transform.n_parameters()

        # Store
        self._transforms = transforms

    def jacobian(self, x):
        """ See :meth:`Transform.jacobian()`. """
        x = pints.vector(x)
        lo, hi = 0, self._transforms[0].n_parameters()
        output = self._transforms[0].jacobian(x[lo:hi])
        for transform in self._transforms[1:]:
            lo = hi
            hi += transform.n_parameters()
            jaco = transform.jacobian(x[lo:hi])
            pack = np.zeros((output.shape[0], jaco.shape[1]))
            output = np.block([[output, pack], [pack.T, jaco]])
        return output

    def log_jacobian_det(self, x):
        """ See :meth:`Transform.log_jacobian_det()`. """
        x = pints.vector(x)
        output = 0
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            output += transform.log_jacobian_det(x[lo:hi])
        return output

    def log_jacobian_det_S1(self, x):
        """ See :meth:`Transform.log_jacobian_det_S1()`. """
        x = pints.vector(x)
        output = 0
        output_s1 = np.zeros(x.shape)
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            j, js1 = transform.log_jacobian_det_S1(x[lo:hi])
            output += j
            output_s1[lo:hi] = np.asarray(js1)
        return output, output_s1

    def to_model(self, x):
        """ See :meth:`Transform.to_model()`. """
        x = pints.vector(x)
        output = np.zeros(x.shape)
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            output[lo:hi] = np.asarray(transform.to_model(x[lo:hi]))
        return output

    def to_search(self, p):
        """ See :meth:`Transform.to_search()`. """
        p = pints.vector(p)
        output = np.zeros(p.shape)
        lo = hi = 0
        for transform in self._transforms:
            lo = hi
            hi += transform.n_parameters()
            output[lo:hi] = np.asarray(transform.to_search(p[lo:hi]))
        return output

    def n_parameters(self):
        """ See :meth:`Transform.n_parameters()`. """
        return self._n_parameters
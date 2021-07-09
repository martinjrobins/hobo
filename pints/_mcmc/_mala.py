#
# Metropolis-adjusted Langevin algorithm
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy.stats


class MALAMCMC(pints.SingleChainMCMC):
    r"""
    Metropolis-Adjusted Langevin Algorithm (MALA), an MCMC sampler as described
    in [1]_.

    This method involves simulating Langevin diffusion such that the solution
    to the time evolution equation (the Fokker-Planck PDE) is a stationary
    distribution that equals the target density (in Bayesian problems, the
    posterior distribution). The stochastic differential equation (SDE) given
    below ensures that if :math:`u(\theta, 0) = \pi(\theta)`,
    then :math:`\partial u / \partial t = 0`,

    .. math::
        \mathrm{d}\Theta_t = 1/2 \nabla \; \text{log} \pi(\Theta_t)
            \mathrm{d}t + \mathrm{d}W_t

    where :math:`\pi(\theta)` is the target density and :math:`W` is
    a standard multivariate Wiener process.

    In general, the above SDE cannot be solved exactly and the below
    first-order Euler discretisation is used instead,

    .. math::
        \theta^* = \theta_t + \epsilon^2 1/2 \nabla \;
            \text{log} \pi(\theta_t) + \epsilon z

    where :math:`z \sim \mathcal{N}(0, I)` resulting in a mean
    :math:`\mu(\theta^*) = \theta_t + \epsilon^2 1/2 \nabla \;
    \text{log} \pi(\theta_t)`.

    To correct for first-order integration error that is introduced from
    discretisation, a Metropolis-Hastings acceptance probability is calculated
    after a step,

    .. math::
        \alpha = \frac{\pi(\theta^*)q(\theta_t|\theta^*)}{\pi(\theta_t)
            q(\theta^*|\theta_t)}

    where :math:`q(\theta_2|\theta_1) =
    \mathcal{N}(\theta_2|\mu(\theta_1), \epsilon I)` and
    :math:`\theta^*` is accepted with probability
    :math:`\text{min}(1, \alpha)`.

    Here we consider a slight variant of the above method discussed in [1]_,
    which is to use a preconditioning matrix :math:`M` to allow differing
    degrees of freedom in each dimension.

    .. math::
        \theta^* = \theta_t + \epsilon'^2 1/2 \nabla \;
            \text{log} \pi(\theta_t) + \epsilon' z

    leading to :math:`q(\theta_2|\theta_1) =
    \mathcal{N}(\theta_2|\mu(\theta_1), \epsilon')`.

    where :math:`\epsilon' = \epsilon \sqrt{M}` is given by the initial value
    of ``sigma0``.

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] Girolami, M. and Calderhead, B., 2011. Riemann manifold langevin and
           hamiltonian monte carlo methods. Journal of the Royal Statistical
           Society: Series B (Statistical Methodology), 73(2), pp.123-214.
           https://doi.org/10.1111/j.1467-9868.2010.00765.x
    """

    def __init__(self, x0, sigma0=None):
        super(MALAMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Current point and proposed point
        self._current = None
        self._current_log_pdf = None
        self._current_gradient = None
        self._proposed = None

        # hyper parameters
        self._epsilon = None
        self._step_size = 0.2
        self._scale_vector = np.diag(self._sigma0)

        # Acceptance rate monitoring
        self._iterations = 0
        self._acceptance = 0

        # Step size
        self._forward_mu = None
        self._backward_mu = None
        self._forward_q = None
        self._backward_q = None

        # initialise step size
        self.set_epsilon()

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

        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')

        # Propose new point
        if self._proposed is None:

            self._forward_mu = self._current + (self._epsilon**2 / 2.0) * (
                self._current_gradient)

            self._proposed = np.random.multivariate_normal(
                self._forward_mu,
                self._epsilon**2 * np.diag(np.ones(self._n_parameters)))

            self._forward_q = scipy.stats.multivariate_normal.logpdf(
                self._proposed, self._forward_mu,
                self._epsilon**2 * np.diag(np.ones(self._n_parameters))
            )

            # Set as read-only
            self._proposed.setflags(write=False)

        self._ready_for_tell = True

        # Return proposed point
        return self._proposed

    def epsilon(self):
        """
        Returns ``epsilon`` which is the effective step size used in proposals.
        """
        return self._epsilon

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
        self._proposed.setflags(write=False)

        # Update sampler state
        self._running = True

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._acceptance)

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Metropolis-Adjusted Langevin Algorithm (MALA)'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def set_epsilon(self, epsilon=None):
        """
        Sets epsilon, which is the effective step size used in proposals.
        If epsilon not specified, then ``epsilon = 0.2 * diag(sigma0)``
        will be used.
        """
        if epsilon is None:
            self._epsilon = self._step_size * self._scale_vector
        else:
            a = np.atleast_1d(epsilon)
            if not len(a) == self._n_parameters:
                raise ValueError('Dimensions of epsilon must be same as ' +
                                 'number of parameters.')
            for element in epsilon:
                if element <= 0:
                    raise ValueError('Elements of epsilon must exceed 0.')
            self._epsilon = np.array(epsilon)

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[epsilon]``.

        The effective step size (``epsilon``) is ``step_size * scale_vector``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_epsilon(x[0])

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """

        # Check if we had a proposal
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Unpack reply
        fx, log_gradient = reply

        # Check reply, copy gradient
        fx = float(fx)
        log_gradient = pints.vector(log_gradient)
        assert(log_gradient.shape == (self._n_parameters, ))

        # First point?
        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite log_pdf.')

            # Accept
            self._current = self._proposed
            self._current_log_pdf = fx
            self._current_gradient = log_gradient

            # Mark current state as read-only for safe returning (proposed is
            # already set)
            self._current_gradient.setflags(write=False)

            # Increase iteration count
            self._iterations += 1

            # Clear proposal
            self._proposed = None

            # Return first point for chain
            return (
                self._current,
                (self._current_log_pdf, self._current_gradient),
                True
            )

        # Calculate alpha
        proposed_gradient = log_gradient
        self._backward_mu = self._proposed + (
            0.5 * self._epsilon**2 * proposed_gradient)
        self._backward_q = scipy.stats.multivariate_normal.logpdf(
            self._current, self._backward_mu,
            self._epsilon**2 * (np.diag(np.ones(self._n_parameters))))
        alpha = fx + self._backward_q - (
            self._current_log_pdf + self._forward_q)

        # Check if the proposed point can be accepted
        accepted = 0
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if alpha > u:
                accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx
                self._current_gradient = log_gradient
                self._current_gradient.setflags(write=False)

        # Clear proposal
        self._proposed = None

        # Update acceptance rate (only used for output!)
        self._acceptance = ((self._iterations * self._acceptance + accepted) /
                            (self._iterations + 1))

        # Increase iteration count
        self._iterations += 1

        # Return next point for chain
        return (
            self._current,
            (self._current_log_pdf, self._current_gradient),
            accepted > 0
        )

#
# Horowitz Langenvin MCMC method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class NealLangevinMCMC(pints.SingleChainMCMC):
    r"""
    Implements the Neal Langevin MCMC algorithm as described in [1]_.

    Similar to MALA and HMC, this method uses a physical analogy of a particle
    moving across a landscape under Hamiltonian dynamics to aid efficient
    exploration of parameter space. The key differences are a persistent
    momentum after Horowitz [2]_, and clustered sequences of proposal
    rejections, which leads to better ergodicity properties.

    It introduces an auxilary variable -- the momentum :math:`p_i` of a
    particle moving in dimension :math:`i` of negative log posterior space --
    which supplements the position :math:`q_i` of the particle in parameter
    space. The particle's motion is dictated by solutions to Hamilton's
    equations,

    .. math::
        dq_i/dt &=   \partial H/\partial p_i\\
        dp_i/dt &= - \partial H/\partial q_i.

    The Hamiltonian is given by,

    .. math::
        H(q,p) &=       U(q)       +        KE(p)\\
               &= -\text{log}(p(q|X)p(q)) + \Sigma_{i=1}^{d} p_i^2/2m_i,

    where :math:`d` is the dimensionality of the model and :math:`m_i` is the
    'mass' assigned to each particle (often chosen to be 1 as default).

    To numerically integrate Hamilton's equations, it is essential to use a
    sympletic discretisation routine, of which the most typical approach is
    the leapfrog method,

    .. math::
        p_i(t + \epsilon/2) &= p_i(t) - (\epsilon/2) d U(q_i(t))/dq_i\\
        q_i(t + \epsilon) &= q_i(t) + \epsilon p_i(t + \epsilon/2) / m_i\\
        p_i(t + \epsilon) &= p_i(t + \epsilon/2) -
                             (\epsilon/2) d U(q_i(t + \epsilon))/dq_i

    In this method each iteration performs exactly one integrational step and
    is therefore in this regard equivalent to MALA or HMC with only one
    leapfrog step.

    Proposals :math:`(q', p')=(q(t + \epsilon), p(t + \epsilon))` are accepted
    if

    .. math::
        u < \text{exp}(-H(q', p')) / \text{exp}(-H(q, p)),

    where :math:`\text{p}(q,p)\propto \text{exp}(-H(q', p'))` is the
    probability of the phase space position :math:`(q, p)`. Here
    :math:`u=|v|` is updated at each MCMC iteration by an increment
    :math:`\delta \sim \mathcal{N}(\bar \delta, \sigma _{\delta})` and
    reflected off at the boundaries 1 and -1, such that :math:`u\in [0,1]`.
    This in contrast to MALA and HMC, where :math:`u` is uniformly drawn from
    :math:`[0, 1]` at each iteration. The gradual updating of :math:`u` leads
    to a clustering of rejections, which overall improves the ergodicity of the
    sampler. See [1]_ for more details.

    If the proposal is rejected, the :math:`(q,p)` is set to the last sampled
    values and the momentum negated. If the proposal is accepted
    :math:`(q,p) \leftarrow (q',p')` and the momentum is NOT negated.

    At the beginning of each MCMC iteration the current momentum is updated by
    a random variable
    :math:`\Delta p \sim \mathcal{N}(0, \mathbb{1})`

    .. math::
        p \leftarrow \alpha ^ 2p + \sqrt{1-\alpha ^ 2}\Delta p.

    This leads to a persistance of the momentum for accepted proposals,
    which avoids Random Walk behaviour in heavily peaked regions of the
    landscape.

    Setting :math:`\alpha = 0` turns the persistance of the momentum off.

    See references

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] "Non-reversibly updating a uniform [0,1] value for Metropolis
           accept/reject decisions". Radford M. Neal, 2020
           https://arxiv.org/abs/2001.11950v1.
    .. [2] "A generalized guided Monte Carlo algorithm", Alan M. Horowitz,
           Physics Letters B, Volume 268, Issue 2, 1991.
    """
    def __init__(self, x0, sigma0=None):
        super(NealLangevinMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Current point in the Markov chain
        self._current = None            # Aka current_q in the chapter
        self._current_energy = None     # Aka U(current_q) = -log_pdf
        self._current_gradient = None
        self._current_momentum = None   # Aka current_p

        # Current point in the leapfrog iterations
        self._momentum = None       # Aka p in the chapter
        self._position = None       # Aka q in the chapter
        self._gradient = None       # Aka grad_U(q) in the chapter

        # Iterations, acceptance monitoring, and leapfrog iterations
        self._mcmc_iteration = 0
        self._mcmc_acceptance = 0

        # Default integration step size for leapfrog algorithm
        self._epsilon = 0.1
        self._step_size = None
        self.set_leapfrog_step_size(np.diag(self._sigma0))

        # Default weighting of momentum update
        self._alpha = 0.9  # Default: high persistance of momentum

        # Default acceptance ratio parameters
        self._v = None
        self._delta = 0.05  # Default: slow updating of v
        self._sigma_delta = self._delta * 0.1  # Default: moderate noise

        # Divergence checking
        # Create a vector of divergent iterations
        self._divergent = np.asarray([], dtype='int')

        # Default threshold for Hamiltonian divergences
        # (currently set to match Stan)
        self._hamiltonian_threshold = 10**3

    def alpha(self):
        r"""
        Returns the weight of the momentum updates.

        Momentum updates before integration are performed according to
        :math:`p' = \alpha ^2 p_i - \sqrt{1 - \alpha ^2}\Delta p`,
        where :math:`\Delta p \sim \mathcal{N}(0,\mathbb{1})`.
        """
        return self._alpha

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        # Check ask/tell pattern
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')

        # Initialise on first call
        if not self._running:
            self._running = True

        # Notes:
        #  Ask is responsible for updating the position, which is the point
        #   returned to the user
        #  Tell is then responsible for updating the momentum, which uses the
        #   gradient at this new point
        #  The MCMC step happens in tell, and does not require any new
        #   information (it uses the log_pdf and gradient of the final point
        #   in the leapfrog run).

        # Very first iteration
        if self._current is None:

            # Ask for the pdf and gradient of x0
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)

        # First leapfrog step in chain
        if self._mcmc_iteration == 1:

            # Sample random initial momentum using identity cov
            self._current_momentum = np.random.multivariate_normal(
                np.zeros(self._n_parameters), np.eye(self._n_parameters))

        # Step 1 in ref [1] p. 5
        # Sample random adjustment of current momentum using identity cov
        delta_momentum = np.random.multivariate_normal(
            np.zeros(self._n_parameters), np.eye(self._n_parameters))

        # Compute new momentum with a weighted update
        self._current_momentum = self._alpha ** 2 * self._current_momentum \
            + np.sqrt(1 - self._alpha ** 2) * delta_momentum

        # Starting leapfrog position is the current sample in the chain
        self._position = np.array(self._current, copy=True)
        self._gradient = np.array(self._current_gradient, copy=True)
        self._momentum = np.array(self._current_momentum, copy=True)

        # Start of step 2 in ref [1] p. 5. (ends in tell)
        # Perform first half of leapfrog step for the momentum
        self._momentum -= self._scaled_epsilon * self._gradient * 0.5

        # Perform a leapfrog step for the position
        self._position += self._scaled_epsilon * self._momentum

        # Ask for the pdf and gradient of the current leapfrog position
        # Using this, the leapfrog step for the momentum is performed in tell()
        self._ready_for_tell = True
        return np.array(self._position, copy=True)

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return -self._current_energy

    def divergent_iterations(self):
        """
        Returns the iteration number of any divergent iterations
        """
        return self._divergent

    def delta(self):
        r"""
        Returns the mean :math:`\bar \delta` and standard deviation
        :math:`\sigma_{\delta}` of the updates
        :math:`\delta \sim \mathcal{N(\bar \delta, \sigma_{\delta})}` of
        the acceptance ratio :math:`u`.
        """
        return [self._delta, self._sigma_delta]

    def epsilon(self):
        """
        Returns epsilon used in leapfrog algorithm
        """
        return self._epsilon

    def hamiltonian_threshold(self):
        """
        Returns threshold difference in Hamiltonian value from one iteration to
        next which determines whether an iteration is divergent.
        """
        return self._hamiltonian_threshold

    def leapfrog_step_size(self):
        """
        Returns the step size for the leapfrog algorithm.
        """
        return self._step_size

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._mcmc_acceptance)

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 4

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Neal Langevin MCMC'

    def needs_sensitivities(self):
        """ See :meth:`pints.MCMCSampler.needs_sensitivities()`. """
        return True

    def scaled_epsilon(self):
        """
        Returns scaled epsilon used in leapfrog algorithm
        """
        return self._scaled_epsilon

    def set_alpha(self, alpha):
        r"""
        Sets the weight of the momentum updates.

        Momentum updates before integration are performed according to
        :math:`p' = \alpha ^2 p_i - \sqrt{1 - \alpha ^2}\Delta p`,
        where :math:`\Delta p \sim \mathcal{N}(0,\mathbb{1})`.
        """
        if alpha < 0 or alpha > 1:
            raise ValueError('Alpha must lie in the interval [0,1].')
        self._alpha = alpha

    def set_delta(self, mean, sigma=None):
        r"""
        Sets the mean :math:`\bar \delta` and standard deviation
        :math:`\sigma_{\delta}` of the updates
        :math:`\delta \sim \mathcal{N(\bar \delta, \sigma_{\delta})}` of
        the acceptance ratio :math:`u`. If ``sigma`` is ``None`` or ``0``, the
        updates will be deterministic. If no value for ``sigma`` is provided,
        it is set to ``None``.
        """
        self._delta = mean

        if sigma:
            if sigma <= 0:
                raise ValueError(
                    'The standard deviation of delta can only take non-'
                    'negative values.')
            if sigma > 0:
                self._sigma_delta = sigma
        else:
            self._sigma_delta = None

    def _set_scaled_epsilon(self):
        """
        Rescales epsilon along the dimensions of step_size
        """
        self._scaled_epsilon = np.zeros(self._n_parameters)
        for i in range(self._n_parameters):
            self._scaled_epsilon[i] = self._epsilon * self._step_size[i]

    def set_epsilon(self, epsilon):
        """
        Sets epsilon for the leapfrog algorithm
        """
        epsilon = float(epsilon)
        if epsilon <= 0:
            raise ValueError('epsilon must be positive for leapfrog algorithm')
        self._epsilon = epsilon
        self._set_scaled_epsilon()

    def set_hamiltonian_threshold(self, hamiltonian_threshold):
        """
        Sets threshold difference in Hamiltonian value from one iteration to
        next which determines whether an iteration is divergent.
        """
        if hamiltonian_threshold < 0:
            raise ValueError('Threshold for divergent iterations must be ' +
                             'non-negative.')
        self._hamiltonian_threshold = hamiltonian_threshold

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[alpha, step_size, mean_delta,
        std_delta]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_alpha(x[0])
        self.set_leapfrog_step_size(x[1])
        if len(x) < 4:
            self.set_delta(mean=x[2])
        else:
            self.set_delta(mean=x[2], sigma=x[3])

    def set_leapfrog_step_size(self, step_size):
        """
        Sets the step size for the leapfrog algorithm.
        """
        a = np.atleast_1d(step_size)
        if len(a[a < 0]) > 0:
            raise ValueError(
                'Step size for leapfrog algorithm must' +
                'be greater than zero.'
            )
        if len(a) == 1:
            step_size = np.repeat(step_size, self._n_parameters)
        elif not len(step_size) == self._n_parameters:
            raise ValueError(
                'Step size should either be of length 1 or equal to the' +
                'number of parameters'
            )
        self._step_size = step_size
        self._set_scaled_epsilon()

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Unpack reply
        energy, gradient = reply

        # Check reply, copy gradient
        energy = float(energy)
        gradient = pints.vector(gradient)
        assert(gradient.shape == (self._n_parameters, ))

        # Energy = -log_pdf, so flip both signs!
        energy = -energy
        gradient = -gradient

        # Very first call
        if self._current is None:

            # Check first point is somewhere sensible
            if not np.isfinite(energy):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Set current sample, energy, and gradient
            self._current = self._x0
            self._current_energy = energy
            self._current_gradient = gradient

            # Increase iteration count
            self._mcmc_iteration += 1

            # Set rejection threshold (no default in paper)
            self._v = 0.5

            # Mark current as read-only, so it can be safely returned
            self._current.setflags(write=False)

            # Return first point in chain
            return self._current

        # Set gradient of current leapfrog position
        self._gradient = gradient

        # Perform second (final) half of leapfrog step for the momentum
        self._momentum -= self._scaled_epsilon * self._gradient * 0.5

        # End of step 2 in ref [1] p. 5
        # Negation of momentum
        self._momentum *= -1

        # Before starting accept/reject procedure, check if the leapfrog
        # procedure has led to a finite momentum and logpdf. If not, reject.
        accept = 0
        if np.isfinite(energy) and np.all(np.isfinite(self._momentum)):

            # Evaluate potential and kinetic energies at start and end of
            # leapfrog trajectory
            current_U = self._current_energy
            current_K = np.sum(self._current_momentum**2 / 2)
            proposed_U = energy
            proposed_K = np.sum(self._momentum**2 / 2)

            # Check for divergent iterations by testing whether the
            # Hamiltonian difference is above a threshold
            div = proposed_U + proposed_K - (self._current_energy + current_K)
            if np.abs(div) > self._hamiltonian_threshold:  # pragma: no cover
                self._divergent = np.append(
                    self._divergent, self._mcmc_iteration)
                self._momentum = self._position = self._gradient = None

                # Step 4 in ref [1] p. 5 (dealt with as if rejected in step 3)
                # Negate current momentum
                self._current_momentum *= -1

                # Update MCMC iteration count
                self._mcmc_iteration += 1

                # Update acceptance rate (only used for output!)
                self._mcmc_acceptance = (
                    (self._mcmc_iteration * self._mcmc_acceptance + accept) /
                    (self._mcmc_iteration + 1))
                self._current.setflags(write=False)
                return self._current

            # Accept/reject
            else:
                # Step 3 in ref [1] p. 5
                r = np.exp(current_U - proposed_U + current_K - proposed_K)

                # Eq. (1) in ref [1] p. 2
                # Update rejection threshold # TODO: which noise?
                noise = 0
                if self._sigma_delta:
                    noise = np.random.normal(0, self._sigma_delta)

                self._v += self._delta + noise

                # Make sure theta v is in [-1, 1]
                while self._v > 1:
                    self._v -= 2
                while self._v < -1:
                    self._v += 2

                if abs(self._v) < r:
                    accept = 1
                    self._current = self._position
                    self._current_momentum = self._momentum
                    self._current_energy = energy
                    self._current_gradient = gradient

                    # Mark current as read-only, so it can be safely returned
                    self._current.setflags(write=False)

        # Reset leapfrog mechanism
        self._momentum = self._position = self._gradient = None

        # Step 4 in ref [1] p. 5
        # Negate current momentum
        self._current_momentum *= -1

        # Update MCMC iteration count
        self._mcmc_iteration += 1

        # Update acceptance rate (only used for output!)
        self._mcmc_acceptance = (
            (self._mcmc_iteration * self._mcmc_acceptance + accept) /
            (self._mcmc_iteration + 1))

        # Return current position as next sample in the chain
        return self._current

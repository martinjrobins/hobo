#
# Nested ellipsoidal sampler implementation.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
from pints._nested.__init__ import Ellipsoid


class NestedEllipsoidSampler(pints.NestedSampler):
    r"""
    Creates a nested sampler that estimates the marginal likelihood
    and generates samples from the posterior.

    This is the form of nested sampler described in [1]_, where an ellipsoid is
    drawn around surviving particles (typically with an enlargement factor to
    avoid missing prior mass), and then random samples are drawn from within
    the bounds of the ellipsoid. By sampling in the space of surviving
    particles, the efficiency of this algorithm aims to improve upon simple
    rejection sampling. This algorithm has the following steps:

    Initialise::

        Z_0 = 0
        X_0 = 1

    Draw samples from prior::

        for i in 1:n_active_points:
            theta_i ~ p(theta), i.e. sample from the prior
            L_i = p(theta_i|X)
        endfor
        L_min = min(L)
        indexmin = min_index(L)

    Run rejection sampling for ``n_rejection_samples`` to generate
    an initial sample, along with updated values of ``L_min`` and
    ``indexmin``.

    Fit active points using a minimum volume bounding ellipse. In our approach,
    we do this with the following procedure (which we term
    ``minimum_volume_ellipsoid`` in what follows) that returns the positive
    definite matrix A with centre c that define the ellipsoid
    by :math:`(x - c)^t A (x - c) = 1`::

        cov = covariance(transpose(active_points))
        cov_inv = inv(cov)
        c = mean(points)
        for i in n_active_points:
            dist[i] = (points[i] - c) * cov_inv * (points[i] - c)
        endfor
        enlargement_factor = max(dist)
        A = (1.0 / enlargement_factor) * cov_inv
        return A, c

    From then on, in each iteration (t), the following occurs::

        if mod(t, ellipsoid_update_gap) == 0:
            A, c = minimum_volume_ellipsoid(active_points)
        else:
            if dynamic_enlargement_factor:
                enlargement_factor *= (
                    exp(-(t + 1) / n_active_points)**alpha
                )
            endif
        endif
        L_min = min(L)
        indexmin = min_index(L)
        theta* = ellipsoid_sample(enlargement_factor, A, c)
        while p(theta*|X) < L_min:
            theta* = ellipsoid_sample(enlargement_factor, A, c)
        endwhile
        X_t = exp(-t / n_active_points)
        w_t = X_t - X_t-1
        Z = Z + L_min * w_t
        theta_indexmin = theta*
        L_indexmin = p(theta*|X)


    If the parameter ``dynamic_enlargement_factor`` is true, the enlargement
    factor is shrunk as the sampler runs, to avoid inefficiencies in later
    iterations. By default, the enlargement factor begins at 1.1.

    In ``ellipsoid_sample``, a point is drawn uniformly from within the minimum
    volume ellipsoid, whose volume is increased by a factor
    ``enlargement_factor``.

    At the end of iterations, there is a final ``Z`` increment::

        Z = Z + (1 / n_active_points) * (L_1 + L_2 + ..., + L_n_active_points)

    The posterior samples are generated as described in [2]_ on page 849 by
    weighting each dropped sample in proportion to the volume of the
    posterior region it was sampled from. That is, the probability
    for drawing a given sample j is given by::

        p_j = L_j * w_j / Z

    where j = 1, ..., n_iterations.

    Extends :class:`NestedSampler`.

    References
    ----------
    .. [1] "A nested sampling algorithm for cosmological model selection",
           Pia Mukherjee, David Parkinson, Andrew R. Liddle, 2008.
           arXiv: arXiv:astro-ph/0508461v2 11 Jan 2006
           https://doi.org/10.1086/501068
    .. [2] "Nested Sampling for General Bayesian Computation", John Skilling,
           Bayesian Analysis 1:4 (2006).
           https://doi.org/10.1214/06-BA127
    """

    def __init__(self, log_prior):
        super(NestedEllipsoidSampler, self).__init__(log_prior)

        # Gaps between updating ellipsoid
        self.set_ellipsoid_update_gap()

        # Enlargement factor for ellipsoid
        self.set_enlargement_factor()
        self._f0 = self._enlargement_factor - 1

        # Initial phase of rejection sampling
        # Number of nested rejection samples before starting ellipsoidal
        # sampling
        self.set_n_rejection_samples()
        self.set_initial_phase(True)

        self._needs_sensitivities = False

        # Dynamically vary the enlargement factor
        self._dynamic_enlargement_factor = False
        self._alpha = 0.2
        self._ellipsoid = None

    def ellipsoid(self):
        """ Returns ellipsoid used in sampling. """
        return self._ellipsoid

    def set_dynamic_enlargement_factor(self, dynamic_enlargement_factor):
        """
        Sets dynamic enlargement factor
        """
        self._dynamic_enlargement_factor = bool(dynamic_enlargement_factor)

    def dynamic_enlargement_factor(self):
        """
        Returns dynamic enlargement factor.
        """
        return self._dynamic_enlargement_factor

    def set_alpha(self, alpha):
        """
        Sets alpha which controls rate of decline of enlargement factor
        with iteration  (when `dynamic_enlargement_factor` is true).
        """
        if alpha < 0 or alpha > 1:
            raise ValueError('alpha must be between 0 and 1.')
        self._alpha = alpha

    def alpha(self):
        """
        Returns alpha which controls rate of decline of enlargement factor
        with iteration (when `dynamic_enlargement_factor` is true).
        """
        return self._alpha

    def set_initial_phase(self, in_initial_phase):
        """ See :meth:`pints.NestedSampler.set_initial_phase()`. """
        self._rejection_phase = bool(in_initial_phase)

    def needs_initial_phase(self):
        """ See :meth:`pints.NestedSampler.needs_initial_phase()`. """
        return True

    def in_initial_phase(self):
        """ See :meth:`pints.NestedSampler.in_initial_phase()`. """
        return self._rejection_phase

    def ellipsoid_update_gap(self):
        """
        Returns the ellipsoid update gap used in the algorithm (see
        :meth:`set_ellipsoid_update_gap()`).
        """
        return self._ellipsoid_update_gap

    def enlargement_factor(self):
        """
        Returns the enlargement factor used in the algorithm (see
        :meth:`set_enlargement_factor()`).
        """
        return self._enlargement_factor

    def n_rejection_samples(self):
        """
        Returns the number of rejection sample used in the algorithm (see
        :meth:`set_n_rejection_samples()`).
        """
        return self._n_rejection_samples

    def ask(self, n_points):
        """
        If in initial phase, then uses rejection sampling. Afterwards,
        points are drawn from within an ellipse (needs to be in uniform
        sampling regime).
        """
        i = self._accept_count
        if self._rejection_phase and (i + 1) > self._n_rejection_samples:
            self._rejection_phase = False
            # determine bounding ellipsoid
            self._ellipsoid = Ellipsoid.minimum_volume_ellipsoid(
                self._m_active[:, :self._n_parameters]
            )

        if self._rejection_phase:
            if n_points > 1:
                self._proposed = self._log_prior.sample(n_points)
            else:
                self._proposed = self._log_prior.sample(n_points)[0]
        else:
            # update bounding ellipsoid if sufficient samples taken
            if ((i + 1 - self._n_rejection_samples)
                    % self._ellipsoid_update_gap == 0):
                self._ellipsoid = Ellipsoid.minimum_volume_ellipsoid(
                    self._m_active[:, :self._n_parameters])
            # From Feroz-Hobson (2008) below eq. (14)
            if self._dynamic_enlargement_factor:
                f = (
                    self._f0 *
                    np.exp(-(i + 1) / self._n_active_points)**self._alpha
                )
                self._enlargement_factor = 1 + f
            # propose by sampling within ellipsoid
            self._proposed = self._ellipsoid.sample(
                n_points, self._enlargement_factor)
        return self._proposed

    def set_enlargement_factor(self, enlargement_factor=1.1):
        """
        Sets the factor (>1) by which to increase the minimal volume
        ellipsoidal in rejection sampling.

        A higher value means it is less likely that areas of high probability
        mass will be missed. A low value means that rejection sampling is more
        efficient.
        """
        if enlargement_factor <= 1:
            raise ValueError('Enlargement factor must exceed 1.')
        self._enlargement_factor = enlargement_factor

    def set_n_rejection_samples(self, rejection_samples=200):
        """
        Sets the number of rejection samples to take, which will be assigned
        weights and ultimately produce a set of posterior samples.
        """
        if rejection_samples < 0:
            raise ValueError('Must have non-negative rejection samples.')
        self._n_rejection_samples = rejection_samples

    def set_ellipsoid_update_gap(self, ellipsoid_update_gap=100):
        """
        Sets the frequency with which the minimum volume ellipsoid is
        re-estimated as part of the nested rejection sampling algorithm.

        A higher rate of this parameter means each sample will be more
        efficiently produced, yet the cost of re-computing the ellipsoid
        may mean it is better to update this not each iteration -- instead,
        with gaps of ``ellipsoid_update_gap`` between each update. By default,
        the ellipse is updated every 100 iterations.
        """
        ellipsoid_update_gap = int(ellipsoid_update_gap)
        if ellipsoid_update_gap <= 1:
            raise ValueError('Ellipsoid update gap must exceed 1.')
        self._ellipsoid_update_gap = ellipsoid_update_gap

    def name(self):
        """ See :meth:`pints.NestedSampler.name()`. """
        return 'Nested ellipsoidal sampler'

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 6

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[# active points, # rejection samples,
        enlargement factor, ellipsoid update gap, dynamic enlargement factor,
        alpha]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_n_active_points(x[0])
        self.set_n_rejection_samples(x[1])
        self.set_enlargement_factor(x[2])
        self.set_ellipsoid_update_gap(x[3])
        self.set_dynamic_enlargement_factor(x[4])
        self.set_alpha(x[5])

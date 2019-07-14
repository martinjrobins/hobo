#
# Slice Sampling with Doubling MCMC Method
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


class SliceDoublingMCMC(pints.SingleChainMCMC):
    """
    *Extends:* :class:`SingleChainMCMC`

    Implements Slice Sampling with Doubling, as described in [1]. This is a
    univariate method, which is applied in a Slice-Sampling-within-Gibbs
    framework to allow MCMC sampling from multivariate models.

    Generates samples by sampling uniformly from the volume underneath the
    posterior (``f``). It does so by introducing an auxiliary variable (``y``)
    and by definying a Markov chain.

    If the distribution is univariate, sampling follows:

    1) Calculate the pdf (``f(x0)``) of the current sample (``x0``).
    2) Draw a real value (``y``) uniformly from (0, f(x0)), defining a
    horizontal “slice”: S = {x: y < f (x)}. Note that ``x0`` is
    always within S.
    3) Find an interval (``I = (L, R)``) around ``x0`` that contains all,
    or much, of the slice.
    4) Draw a new point (``x1``) from the part of the slice
    within this interval.

    If the distribution is multivariate, we apply the univariate algorithm to
    each variable in turn, where the other variables are set at their
    current values.

    This implementation uses the ``Doubling`` method to estimate the interval
    ``I = (L, R)``, as described in [1] Fig. 4. pp.715 and consists of the
    following steps:

    1. ``U \sim uniform(0, 1)``
    2. ``L = x_0 - wU``
    3. ``R = L + w``
    4. ``K = p``
    5. while ``K > 0`` and ``{y < f(L) or y < f(R)}``:
        a. ``V \sim uniform(0, 1)``
        b. if ``V < 0.5``, ``L = L - (R - L)``
           else, ``R = R + (R - L)``
    6. ``K = K - 1``

    Intuitively, the interval ``I`` is estimated by expanding the initial
    interval by producing a sequence of intervals, each twice the size
    of the previous one, until an interval is found with both ends outside
    the slice, or until a pre-determined limit is reached. The parameters
    ``p`` (an integer, which determines the limit of slice size) and
    ``w`` (the estimate of typical slice width) are hyperparameters.

    To sample from the interval ``I = (L, R)``, such that the sample
    ``x`` satisfies ``y < f(x)``, we use the ``Shrinkage`` procedure, which
    reduces the size of the interval after rejecting a trial point,
    as defined in [1] Fig. 5. pp.716. This algorithm consists of the
    following steps:

    1. ``\bar{L} = L`` and ``\bar{R} = R``
        2. Repeat:
            a. ``U \sim uniform(0, 1)``
            b. ``x_1 = \bar{L} + U (\bar{R} - \bar{L})``
            c. if ``y < f(x_1)`` and ``Accept(x_1)``, exit loop
            else
                if ``x_1 < x_0``, ``\bar{L} = x_1``
                else ``\bar{R} = x_1``

    Intuitively, we uniformly sample a trial point from the interval ``I``,
    and subsequently shrink the interval each time a trial point is rejected.

    The ``Accept(x_1)`` check is required to guarantee detailed balance. We
    shall refer to this check as the ``Acceptance Check``. Intuitively, it
    tests  whether starting the doubling expansion at ``x_1`` leads to an
    earlier termination compared to starting it from the current state ``x_0``.
    The procedure works backward through the intervals that the doubling
    expansion would pass through to arrive at ``I`` when starting from ``x_1``,
    checking that none of them has both ends outside the slice. The algorithm
    is described in [1] Fig. 6. pp.717 and it consists of the following steps:

    1. ``\hat{L} = L`` and ``\hat{R} = R`` and ``D = False``
        2. while ``\hat{R} - \hat{L} > 1.1w``:
            a. M = ``(\hat{L} + \hat{R})/2``
            b. if {``x_0 < M`` and ``x_1 >= M``} or
                  {``x_0 >= M`` and `` x_1 < M``}, ``D = True``
            c. if ``x_1 < M``, `\hat{R} = M``
               else, `\hat{L} = M``
            d. if ``D`` and ``y >= f(\hat{L})`` and ``y >= f(\hat{R})``,
                reject proposal
        3. If the proposal is not rejected in the previous loop, accept it

    The multiplication by ``1.1`` in the ``while`` condition in Step 2 guards
    against possible round-off errors. The variable ``D`` tracks whether the
    intervals that would be generated from ``x_1`` differ from those leading
    to ``x_0``: when they don't, time is saved by omitting the subsequent
    check.

    To avoid floating-point underflow, we implement the suggestion advanced
    in [1] pp.712. We use the log pdf of the un-normalised posterior
    (``g(x) = log(f(x))``) instead of ``f(x)``. In doing so, we use an
    auxiliary variable ``z = log(y) = g(x0) − \epsilon``, where
    ``\epsilon \sim \text{exp}(1)`` and define the slice as
    S = {x : z < g(x)}.

    [1] Neal, R.M., 2003. Slice sampling. The annals of statistics, 31(3),
    pp.705-767.
    """

    def __init__(self, x0, sigma0=None):
        super(SliceDoublingMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._x0 = np.asarray(x0, dtype=float)
        self._running = False
        self._ready_for_tell = False

        # Iterations monitoring
        self._mcmc_iteration = 0

        # Current sample, log_pdf of the current sample
        self._current = None
        self._current_log_pdf = None

        # Current log_y used to define the slice
        self._current_log_y = None

        # Current proposed sample, log_pdf of the proposed sample
        self._proposed = None
        self._proposed_pdf = None

        # Flag used to store the log_pdf of the proposed sample
        self._sent_proposal = False

        # Default initial interval width w used in the Doubling procedure
        # to expand the interval
        self._w = np.ones(len(self._x0))

        # Default integer ``p`` limiting the size of the interval to
        # ``(2^p) * w``.
        # Integer ``k``` is used to count the interval expansion steps
        self._p = 10
        self._k = 0

        # Flag to initialise the expansion of the interval ``I=(L,R)``
        self._first_expansion = False

        # Flag indicating whether the interval expansion is concluded
        self._interval_found = False

        # Edges of the interval ``I=(L,R)``
        self._l = None
        self._r = None

        # Multi-dimensional points used to calculate the log_pdf of the
        # edges ``l,r```
        self._temp_l = None
        self._temp_r = None

        # Log_pdf of interval edge points ``l,r```
        self._fx_l = None
        self._fx_r = None

        # Edges used for the "Acceptance Check"
        self._l_hat = None
        self._r_hat = None

        # Multi-dimensional points used to calculate the log_pdf of the
        # "Acceptance Check" edges l_hat ,r_hat
        self._temp_l_hat = None
        self._temp_r_hat = None

        # Log_pdf of the "Acceptance Check" interval edge points
        # ``l_hat,r_hat``
        self._fx_l_hat = None
        self._fx_r_hat = None

        # Variable used in the "Acceptance Check" procedure to track whether
        # the intervals that would be generated from the new trial point
        # differ from those leading to the current point.
        self._d = False

        # Flag to initialise the "Acceptance Check"
        self._init_check = False

        # Flag to control the "Acceptance Check"
        self._continue_check = False

        # Mid point between l_hat and r_hat used in the "Acceptance Check"
        self._m = 0

        # Index of parameter "xi" we are updating of the sample
        # "x = (x1,...,xn)"
        self._i = 0

        # Stochastic variables as instance variables for testing
        self._u = 0
        self._v = 0
        self._e = 0

        # Flags used to calculate log_pdf of initial interval edges ``l,r``
        self._init_left = False
        self._init_right = False

        # Flags used to calculate log_pdf of initial ``Acceptance Check``
        # edges ``l_hat,r_hat``
        self._init_left_hat = False
        self._init_right_hat = False

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """

        # Check ask/tell pattern
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')

        # Initialise on first call
        if not self._running:
            self._running = True

        # Very first iteration
        if self._current is None:

            # Ask for the log pdf of x0
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)

        # If the flag is True, we initialise the expansion of interval
        # ``I=(l,r)``
        if self._first_expansion:

            # Set limit to the number to expansion steps
            self._k = self._p

            # Set initial values for l and r
            self._u = np.random.uniform()
            self._l = self._proposed[self._i] - self._w[self._i] * self._u
            self._r = self._l + self._w[self._i]

            # Initialise arrays used for calculating the log_pdf of the
            # edges l,r
            self._temp_l = np.array(self._proposed, copy=True)
            self._temp_r = np.array(self._proposed, copy=True)
            self._temp_l[self._i] = self._l
            self._temp_r[self._i] = self._r

            # We have initialised the expansion, so we set the flag to false
            self._first_expansion = False

            # Set flags to calculate log_pdf of ``l,r``
            self._init_left = True
            self._init_right = True

        # Ask for log_pdf of initial edges ``l,r```
        if self._init_left:

            # Ask for log_pdf of initial left edge
            self._ready_for_tell = True
            return np.array(self._temp_l, copy=True)

        if self._init_right:

            # Ask for log pdf of initial right edge
            self._ready_for_tell = True
            return np.array(self._temp_r, copy=True)

        # Expand the interval ``I``` until edges ``l,r`` are outside the slice
        # or we have reached limit of expansion steps
        if self._k > 0 and (self._current_log_y < self._fx_l or
                            self._current_log_y < self._fx_r):

            # Decrease number of allowed interval expansion steps
            self._k -= 1

            # Use ``Doubling`` expansion procedure as described in
            # [1] Fig. 4. pp.715
            self._v = np.random.uniform()

            # Return new expanded interval edges
            self._ready_for_tell = True

            if self._v < .5:
                self._l = self._l - (self._r - self._l)
                self._temp_l[self._i] = self._l
                return np.array(self._temp_l, copy=True)

            else:
                self._r = self._r + (self._r - self._l)
                self._temp_r[self._i] = self._r
                return np.array(self._temp_r, copy=True)

        # If previous "if" condition fails, the interval has been found
        self._interval_found = True

        # If we have proposed a new point, we initialise the
        # ``Acceptance Check``
        if self._init_check:

            # Initialise edges for the ``Acceptance Check``
            self._l_hat = self._l
            self._r_hat = self._r
            self._temp_l_hat = np.array(self._temp_l, copy=True)
            self._temp_r_hat = np.array(self._temp_r, copy=True)

            # Put flag to False so that we don't repeat the process while
            # being in the "Acceptance Check" loop
            self._init_check = False

            # Set flags to calculate log_pdf of ``l_hat,r_hat``
            self._init_left_hat = True
            self._init_right_hat = True

        # Ask for log_pdf of initial edges ``l_hat,r_hat```
        if self._init_left_hat:

            # Ask for log_pdf of initial left edge
            self._ready_for_tell = True
            return np.array(self._temp_l_hat, copy=True)

        if self._init_right_hat:

            # Ask for log pdf of initial right edge
            self._ready_for_tell = True
            return np.array(self._temp_r_hat, copy=True)

        # After having initialised the ``Acceptance Check`` procedure, we
        # continue with the checking loop
        if self._continue_check:

            # Work backward through the intervals that the doubling procedure
            # would pass through to arrive at the interval ``I`` when starting
            # from the new trial point. The float 1.1 guards against round-off
            # error, as described in [1] Fig.6 pp.717
            if (self._r_hat - self._l_hat) > 1.1 * self._w[self._i]:

                # Calculate interval ``A=(l_hat, r_hat)`` mid point
                self._m = (self._l_hat + self._r_hat) / 2

                # Boolean d tracks tracks whether the intervals that would be
                # generated from the new point differ from those leading to
                # the current point
                if ((self._current[self._i] < self._m and
                     self._proposed[self._i] >= self._m) or
                    (self._current[self._i] >= self._m and
                     self._proposed[self._i] < self._m)):
                    self._d = True

                # Send the new edges of interval ``A=(l_hat, r_hat) to
                # continue the ``Acceptance Check`` loop
                self._ready_for_tell = True

                # Work backward through the doubling procedure starting from
                # the trial point
                if self._proposed[self._i] < self._m:
                    self._r_hat = self._m
                    self._temp_r_hat[self._i] = self._r_hat
                    return np.array(self._temp_r_hat, copy=True)

                if self._proposed[self._i] >= self._m:
                    self._l_hat = self._m
                    self._temp_l_hat[self._i] = self._l_hat
                    return np.array(self._temp_l_hat, copy=True)

            # Now that (r_hat - l_hat) <= 1.1*w, the ``Acceptance Check`` loop
            # is over and we accept the trial point.

            # Reset log_pdf of initial interval edges for the
            # ``Acceptance Check``
            self._fx_r_hat = None
            self._fx_l_hat = None

            # Reset variable d to be correctly initalised for the next
            # ``Acceptance Check`` loop
            self._d = False

            # Reset ``Acceptance Check``
            self._continue_check = False

            # Send the accepted trial point
            self._ready_for_tell = True
            return np.array(self._proposed, copy=True)

        # Sample new parameter by sampling uniformly from the interval
        # ``I=(l,r)``
        self._u = np.random.uniform()
        self._proposed[self._i] = self._l + self._u * (self._r - self._l)

        # Set "Acceptance Check" flags for the proposal point
        self._init_check = True
        self._continue_check = True

        # Set flag indicating we have created a new proposal. This is used to
        # store the value of the log_pdf of the proposed point in the tell()
        # method
        self._sent_proposal = True

        # Send new point for ``Threshold Check```
        self._ready_for_tell = True
        return np.array(self._proposed, copy=True)

    def tell(self, reply):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """

        # Check ask/tell pattern
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False

        # Unpack reply
        fx = np.asarray(reply, dtype=float)

        # If this is the log_pdf of a new point, save the value and use it for
        # the "Threshold Check"
        if self._sent_proposal:
            self._proposed_pdf = fx
            self._sent_proposal = False

        # Very first call
        if self._current is None:

            # Check first point is somewhere sensible
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite logpdf.')

            # Set current sample, log pdf of current sample and initialise
            # proposed sample for next iteration
            self._current = np.array(self._x0, copy=True)
            self._current_log_pdf = fx
            self._proposed = np.array(self._current, copy=True)

            # Sample height of the slice log_y for the next iteration
            self._e = np.random.exponential(1)
            self._current_log_y = self._current_log_pdf - self._e

            # Increase number of samples found
            self._mcmc_iteration += 1

            # Set flag to true as we need to initialise the interval expansion
            # for the next iteration
            self._first_expansion = True

            # Return first point in chain, which is x0
            return np.array(self._current, copy=True)

        # While we expand the interval ``I=(l,r)``, we return None
        if not self._interval_found:

            # Set the log_pdf of the current interval ``I``` edges ``l,r``
            if self._init_left:
                self._fx_l = fx
                self._init_left = False
            elif self._init_right:
                self._fx_r = fx
                self._init_right = False
            else:
                if self._v < .5:
                    self._fx_l = fx
                else:
                    self._fx_r = fx
            return None

        # Do ``Threshold Check`` to check if the proposed point is within the
        # slice
        if self._current_log_y < self._proposed_pdf:

            # If the point passes the ``Threshold Check``, we start the
            # ``Acceptance Check``
            if self._continue_check:

                if self._init_left_hat:
                    self._fx_l_hat = fx
                    self._init_left_hat = False
                elif self._init_right_hat:
                    self._fx_r_hat = fx
                    self._init_right_hat = False
                elif not self._init_check:
                    if self._proposed[self._i] < self._m:
                        self._fx_r_hat = fx
                    else:
                        self._fx_l_hat = fx

                # If the condition is met, the point fails the
                # ``Acceptance Check`` and is rejected
                if (self._d and self._current_log_y >= self._fx_l_hat and
                   self._current_log_y >= self._fx_r_hat):

                    # If the proposed point is rejected, shrink the interval
                    #  ``I=(l,r)``
                    if self._proposed[self._i] < self._current[self._i]:
                        self._l = self._proposed[self._i]
                        self._temp_l[self._i] = self._l
                    else:
                        self._r = self._proposed[self._i]
                        self._temp_r[self._i] = self._r

                    # If the point has been rejected in the
                    # ``Acceptance Check``, reset the flag so that we wait for
                    # a new trial point before starting the
                    # ``Acceptance Check`` again
                    self._continue_check = False

                    # Reset d
                    self._d = False

                    # Reset log_pdf of initial interval edges for the
                    # ``Acceptance Check``
                    self._fx_r_hat = None
                    self._fx_l_hat = None

                    # Since we reject the trial point, we return None and wait
                    # for a new trial point from ask()
                    return None

                # If the rejection condition is not met, we continue the
                # ``Acceptance Check`` while (r_hat - l_hat ) > 1.1*w
                return None

            # We have updated successfully the parameter of the new sample!

            # Reset flag to true as we need to initialise the interval for the
            # update of the next parameter
            self._first_expansion = True

            # Reset flag to false since we still need to estimate the interval
            # for the next parameter
            self._interval_found = False

            # If we have updated all the parameters of the sample, start
            # constructing next sample
            if self._i == len(self._proposed) - 1:

                # Reset index to 0, so that we start from updating parameter
                # x0 of the next sample
                self._i = 0

                # The accepted sample becomes the new current sample
                self._current = np.array(self._proposed, copy=True)

                # The log_pdf of the accepted sample is used to construct the
                # new slice
                self._current_log_pdf = fx

                # Update number of mcmc iterations
                self._mcmc_iteration += 1

                # Sample new log_y used to define the next slice
                self._e = np.random.exponential(1)
                self._current_log_y = self._current_log_pdf - self._e

                # Return the accepted sample
                return np.array(self._proposed, copy=True)

            # If there are still parameters to update to generate the sample,
            # move to next parameter
            else:
                self._i += 1
                return None

        # If the trial point is rejected in the ``Threshold Check``, shrink
        # the interval
        if self._proposed[self._i] < self._current[self._i]:
            self._l = self._proposed[self._i]
            self._temp_l[self._i] = self._l
        else:
            self._r = self._proposed[self._i]
            self._temp_r[self._i] = self._r

        # If the point has been rejected, set flag for ``Acceptance Check``
        # check to False
        # so that we can sample a new point
        self._init_check = False
        self._continue_check = False

        return None

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Slice Sampling - Doubling'

    def set_w(self, w):
        """
        Sets width w for generating the interval.
        """
        if type(w) == int or float:
            w = np.full((len(self._x0)), w)
        else:
            w = np.asarray(w)
        if any(n < 0 for n in w):
            raise ValueError("""Width w must be positive
                            for interval expansion.""")
        self._w = w

    def set_p(self, m):
        """
        Set integer m for limiting interval expansion.
        """
        p = int(m)
        if p <= 0:
            raise ValueError("""Integer m must be positive to limit the
                             interval size to "(2^p)*w".""")
        self._p = p

    def get_w(self):
        """
        Returns width w used for generating the interval.
        """
        return self._w

    def get_p(self):
        """
        Returns integer p used for limiting interval expansion.
        """
        return self._p

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return self._current_log_pdf

    def current_slice_height(self):
        """
        Returns current log_y used to define the current slice.
        """
        return self._current_log_y

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 2

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[w, p]``.
        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_w(x[0])
        self.set_p(x[1])

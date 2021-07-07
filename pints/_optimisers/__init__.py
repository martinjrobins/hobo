#
# Sub-module containing several optimisation routines
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
from numpy.linalg import norm


class Optimiser(pints.Loggable, pints.TunableMethod):
    """
    Base class for optimisers implementing an ask-and-tell interface.

    This interface provides fine-grained control. Users seeking to simply run
    an optimisation may wish to use the :class:`OptimisationController`
    instead.

    Optimisation using "ask-and-tell" proceed by the user repeatedly "asking"
    the optimiser for points, and then "telling" it the function evaluations at
    those points. This allows a user to have fine-grained control over an
    optimisation, and implement custom parallelisation, logging, stopping
    criteria etc. Users who don't need this functionality can use optimisers
    via the :class:`OptimisationController` class instead.

    All PINTS optimisers are _minimisers_. To maximise a function simply pass
    in the negative of its evaluations to :meth:`tell()` (this is handled
    automatically by the :class:`OptimisationController`).

    All optimisers implement the :class:`pints.Loggable` and
    :class:`pints.TunableMethod` interfaces.

    Parameters
    ----------
    x0
        A starting point for searches in the parameter space. This value may be
        used directly (for example as the initial position of a particle in
        :class:`PSO`) or indirectly (for example as the center of a
        distribution in :class:`XNES`).
    sigma0
        An optional initial standard deviation around ``x0``. Can be specified
        either as a scalar value (one standard deviation for all coordinates)
        or as an array with one entry per dimension. Not all methods will use
        this information.
    boundaries
        An optional set of boundaries on the parameter space.

    Example
    -------
    An optimisation with ask-and-tell, proceeds roughly as follows::

        optimiser = MyOptimiser()
        running = True
        while running:
            # Ask for points to evaluate
            xs = optimiser.ask()

            # Evaluate the score function or pdf at these points
            # At this point, code to parallelise evaluation can be added in
            fs = [f(x) for x in xs]

            # Tell the optimiser the evaluations; allowing it to update its
            # internal state.
            optimiser.tell(fs)

            # Check stopping criteria
            # At this point, custom stopping criteria can be added in
            if optimiser.fbest() < threshold:
                running = False

            # Check for optimiser issues
            if optimiser.stop():
                running = False

            # At this point, code to visualise or benchmark optimiser behaviour
            # could be added in, for example by plotting `xs` in the parameter
            # space.
    """

    def __init__(self, x0, sigma0=None, boundaries=None):

        # Convert and store initial position
        self._x0 = pints.vector(x0)

        # Get dimension
        self._n_parameters = len(self._x0)
        if self._n_parameters < 1:
            raise ValueError('Problem dimension must be greater than zero.')

        # Store boundaries
        self._boundaries = boundaries
        if self._boundaries:
            if self._boundaries.n_parameters() != self._n_parameters:
                raise ValueError(
                    'Boundaries must have same dimension as starting point.')

        # Check initial position is within boundaries
        if self._boundaries:
            if not self._boundaries.check(self._x0):
                raise ValueError(
                    'Initial position must lie within given boundaries.')

        # Check initial standard deviation
        if sigma0 is None:
            # Set a standard deviation

            # Try and use boundaries to guess
            try:
                self._sigma0 = (1 / 6) * self._boundaries.range()
            except AttributeError:
                # No boundaries set, or boundaries don't support range()
                # Use initial position to guess at parameter scaling
                self._sigma0 = (1 / 3) * np.abs(self._x0)
                # But add 1 for any initial value that's zero
                self._sigma0 += (self._sigma0 == 0)

            self._sigma0.setflags(write=False)

        elif np.isscalar(sigma0):
            # Single number given, convert to vector
            sigma0 = float(sigma0)
            if sigma0 <= 0:
                raise ValueError(
                    'Initial standard deviation must be greater than zero.')
            self._sigma0 = np.ones(self._n_parameters) * sigma0
            self._sigma0.setflags(write=False)

        else:
            # Vector given
            self._sigma0 = pints.vector(sigma0)
            if len(self._sigma0) != self._n_parameters:
                raise ValueError(
                    'Initial standard deviation must be None, scalar, or have'
                    ' dimension ' + str(self._n_parameters) + '.')
            if np.any(self._sigma0 <= 0):
                raise ValueError(
                    'Initial standard deviations must be greater than zero.')

    def ask(self):
        """
        Returns a list of positions in the search space to evaluate.
        """
        raise NotImplementedError

    def fbest(self):
        """
        Returns the objective function evaluated at the current best position.
        """
        raise NotImplementedError

    def name(self):
        """
        Returns this method's full name.
        """
        raise NotImplementedError

    def needs_sensitivities(self):
        """
        Returns ``True`` if this methods needs sensitivities to be passed in to
        ``tell`` along with the evaluated error.
        """
        return False

    def running(self):
        """
        Returns ``True`` if this an optimisation is in progress.
        """
        raise NotImplementedError

    def stop(self):
        """
        Checks if this method has run into trouble and should terminate.
        Returns ``False`` if everything's fine, or a short message (e.g.
        "Ill-conditioned matrix.") if the method should terminate.
        """
        return False

    def tell(self, fx):
        """
        Performs an iteration of the optimiser algorithm, using the evaluations
        ``fx`` of the points ``x`` previously specified by ``ask``.

        For methods that require sensitivities (see
        :meth:`needs_sensitivities`), ``fx`` should be a tuple
        ``(objective, sensitivities)``, containing the values returned by
        :meth:`pints.ErrorMeasure.evaluateS1()`.
        """
        raise NotImplementedError

    def xbest(self):
        """
        Returns the current best position.
        """
        raise NotImplementedError


class PopulationBasedOptimiser(Optimiser):
    """
    Base class for optimisers that work by moving multiple points through the
    search space.

    Extends :class:`Optimiser`.
    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(PopulationBasedOptimiser, self).__init__(x0, sigma0, boundaries)

        # Set initial population size using heuristic
        self._population_size = self._suggested_population_size()

    def population_size(self):
        """
        Returns this optimiser's population size.

        If no explicit population size has been set, ``None`` may be returned.
        Once running, the correct value will always be returned.
        """
        return self._population_size

    def set_population_size(self, population_size=None):
        """
        Sets a population size to use in this optimisation.

        If `population_size` is set to ``None``, the population size will be
        set using the heuristic :meth:`suggested_population_size()`.
        """
        if self.running():
            raise Exception('Cannot change population size during run.')

        # Check population size or set using heuristic
        if population_size is not None:
            population_size = int(population_size)
            if population_size < 1:
                raise ValueError('Population size must be at least 1.')

        # Store
        self._population_size = population_size

    def suggested_population_size(self, round_up_to_multiple_of=None):
        """
        Returns a suggested population size for this method, based on the
        dimension of the search space (e.g. the parameter space).

        If the optional argument ``round_up_to_multiple_of`` is set to an
        integer greater than 1, the method will round up the estimate to a
        multiple of that number. This can be useful to obtain a population size
        based on e.g. the number of worker processes used to perform objective
        function evaluations.
        """
        population_size = self._suggested_population_size()

        if round_up_to_multiple_of is not None:
            n = int(round_up_to_multiple_of)
            if n > 1:
                population_size = n * (((population_size - 1) // n) + 1)

        return population_size

    def _suggested_population_size(self):
        """
        Returns a suggested population size for use by
        :meth:`suggested_population_size`.
        """
        raise NotImplementedError

    def n_hyper_parameters(self):
        """ See :meth:`TunableMethod.n_hyper_parameters()`. """
        return 1

    def set_hyper_parameters(self, x):
        """
        The hyper-parameter vector is ``[population_size]``.

        See :meth:`TunableMethod.set_hyper_parameters()`.
        """
        self.set_population_size(x[0])


class LineSearchBasedOptimiser(Optimiser):
    """
    Base class for optimisers that incorporate a line search
    within their algorithm.

    The Hager-Zhang line search algorithm [1] is implemented
    in this class.

    [1] Hager, W. W.; Zhang, H. Algorithm 851: CG_DESCENT,
    a Conjugate Gradient Method with Guaranteed Descent.
    ACM Trans. Math. Softw. 2006, 32 (1), 113-137.
    https://doi.org/10.1145/1132973.1132979.

    Extends :class:`Optimiser`.
    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(LineSearchBasedOptimiser, self).__init__(x0, sigma0, boundaries)

        self._evaluator = None

        # Set optimiser state
        self._running = False
        self._ready_for_tell = False

        # Best solution found
        self._xbest = self._x0
        self._fbest = float('inf')
        self._dfdx_best = [float('inf'), float('inf')]

        # Number of iterations run
        self._iterations = 0

        # Parameters for wolfe conditions on line search

        # As c1 approaches 0 and c2 approaches 1, the line search
        # terminates more quickly.
        self._c1 = 1E-4  # Parameter for Armijo condition rule, 0 < c1 < 0.5
        self._c2 = 0.9  # Parameter for curvature condition rule, c1 < c2 < 1.0

        # boundary values of alpha
        self._minimum_alpha = 0.0
        self._maximum_alpha = float("inf")
        self._proposed_alpha = 0.001  # same default value as used in stan

        self.__first_update_step_not_completed = True
        self.__update_step_not_completed = True
        self.__performing_line_search = False

        # Increase allowed between accepted positions when using approximate
        # wolfe conditions, this takes into acccount machine error and
        # insures decreasing.
        self.epsilon = 1E-6

        # range (0, 1), used in the ``self.__update()`` and
        # ``self.__initial_bracket()`` when the potential intervals violate
        # the opposite slope condition (see function definition)
        self.theta = 0.5

        self.__gamma = 0.66

        # range (0, 1) small factor used in initial guess of step size
        self.__ps_0 = 0.01
        # range (0, 1) small factor used in subsequent guesses of step size
        self.__ps_1 = 0.1
        # range (1, inf) factor used in subsequent guesses of step size
        self.__ps_2 = 2.0

        self.__current_alpha = None

        # approximate inverse hessian
        # initial the identity is used
        self._B = np.identity(self._n_parameters)

        # newton direction
        self._px = None

        # number of accepted steps/ newton direction updates
        self._k = 0

        # number of steps in the current line search iteration
        self.__j = 0
        self.__logic_steps_left = 0

        # maximum number of line search steps before a successful point

        # Current point, score, and gradient
        self._current = self._x0
        self._current_f = None
        self._current_dfdx = None

        # Proposed next point (read-only, so can be passed to user)
        self._proposed = self._x0
        self._proposed.setflags(write=False)

        self.__convergence = False

        # logic for passing to tell at the right moments
        self.__1st_wolfe_check_needed = False
        self.__1st_wolfe_check_done = False
        self.__2nd_wolfe_check_needed = False
        self.__2nd_wolfe_check_done = False
        self.__need_update = False
        self.__converged_ls = False

    def _set_function_evaluator(self, function):

        f = function
        if self.needs_sensitivities:
            f = f.evaluateS1

        # Create evaluator object
        self._evaluator = pints.SequentialEvaluator(f)

    def fbest(self):
        """ See :meth:`Optimiser.fbest()`. """
        return self._fbest

    def wolfe_line_search_parameters(self):
        """
        Returns the  wolfe line search parameters this optimiser is using
        as a vector ``[c1, c2]``.
        As c1 approaches 0 and c2 approaches 1, the line search terminates
        more quickly. ``c1`` is the parameter for the Armijo condition
        rule, ``0 < c1 < 0.5``. ``c2 `` is the parameter for the
        curvature condition rule, ``c1 < c2 < 1.0``.
        """
        return (self._c1, self._c2)

    def needs_sensitivities(self):
        """ See :meth:`Optimiser.needs_sensitivities()`. """
        return True

    def n_hyper_parameters(self):
        """ See :meth:`pints.TunableMethod.n_hyper_parameters()`. """
        return 2

    def running(self):
        """ See :meth:`Optimiser.running()`. """
        return self._running

    def set_hyper_parameters(self, x):
        """
        See :meth:`pints.TunableMethod.set_hyper_parameters()`.

        The hyper-parameter vector is ``[c1, c2, m]``.
        ``c1`` is the parameter for the Armijo condition rule,
        ``0 < c1 < 0.5``.
        ``c2`` is the parameter for the curvature condition rule,
        ``c1 < c2 < 1.0``.
        ``m`` is the number of maximum number of correction matrices
        that can be stored for the LM-BFGS update.
        """

        self.__set_wolfe_line_search_parameters(x[0], x[1])

    def __set_wolfe_line_search_parameters(self, c1, c2):
        """
        Sets the parameters for the wolfe conditions.

        Parameters
        ----------
        c1: float
        Parameter for the Armijo condition rule, ``0 < c1 < 0.5``.

        c2: float
        Parameter for the curvature condition rule, ``c1 < c2 < 1.0``.
        """
        if(0 < c1 < 0.5 and c1 < c2 < 1.0):
            self._c1 = c1
            self._c2 = c2
        else:
            cs = self.wolfe_line_search_parameters()
            print('Invalid wolfe line search parameters!!!')
            print('0 < c1 < 0.5 and c1 < c2 < 1.0')
            print('using default parameters: c1 = ', cs[0], ' c2 = ', cs[1])

    def xbest(self):
        """ See :meth:`Optimiser.xbest()`. """
        return self._xbest

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """

        # print('')
        # print('in ask')
        # print('')

        if not self._running:
            self._proposed = np.asarray(self._x0)
        else:

            if self.__j == 0:
                # working out an initial stepsize value alpha
                alpha_0 = self.__initialising(k=self._k,
                                              alpha_k0=self.__current_alpha)
                print('alpha_initial: ', alpha_0)
                # Creating an initial bracketing interval of alpha values
                # satisfying the opposite slope condition (see function
                # docstring) beginning with initial guess [0, alpha_initial].
                bracket = (self.__initial_bracket(c=alpha_0))
                self._minimum_alpha = bracket[0]
                self._maximum_alpha = bracket[1]
                self._updated_minimum_alpha = self._minimum_alpha
                self._updated_maximum_alpha = self._maximum_alpha

            # Looping while wolfe conditions don't need to be checked
            # and the line search hasn't converged.
            while (self.__1st_wolfe_check_needed is not True and
                    self.__2nd_wolfe_check_needed is not True and
                    self.__converged_ls is not True):

                # secant squared step of the line search

                # ***************************************************************
                # a, b = self.__secant2(self._minimum_alpha,
                #                       self._maximum_alpha)
                a = self._updated_minimum_alpha
                b = self._updated_maximum_alpha
                c = self._proposed_alpha

                # checking if the bracketing interval has converged
                self.__converged_ls = self.__very_close(a, b)
                self.__logic_steps_left = 'not started'
                if self.__converged_ls:
                    self.__logic_steps_left = ' converged in ls '
                # if converged is True don't do anything more.

                # ************ beginning of secant squared see [1] ************

                # Preforming a secant to propose a value of alpha.
                if (self.__1st_wolfe_check_done is not True and
                        self.__2nd_wolfe_check_done is not True and
                        self.__converged_ls is not True):

                    # step S1 in [1]
                    self._proposed_alpha = self.__secant_for_alpha(a, b)
                    # passing to tell to check wolfe conditions
                    self.__1st_wolfe_check_needed = True

                    # checking the proposed point is in range
                    if self._proposed_alpha < a or self._proposed_alpha > b:
                        # If the point is out of range there is no need to
                        # check wolfe conditions.
                        self.__1st_wolfe_check_needed = False
                        self.__1st_wolfe_check_done = True

                    self.__logic_steps_left = 2
                    self.__j += 1 / 3

                elif (self.__1st_wolfe_check_done and
                        self.__2nd_wolfe_check_done is not True and
                        self.__converged_ls is not True):

                    # (typically) updating one side of the
                    # bracketing interval
                    A, B = self.__update(a, b, c)
                    # end of step S1 in [1]

                    # (typically) updating the otherside side of
                    # the bracketing interval
                    if c == B:
                        # S2 in [1]
                        # Preforming a secant to propose a value of alpha.
                        self._proposed_alpha = self.__secant_for_alpha(b, B)

                        # checking the proposed point is in range
                        if (self._proposed_alpha < A or
                                self._proposed_alpha > B):
                            # If the point is out of range there is no need to
                            # check wolfe conditions.
                            self.__2nd_wolfe_check_needed = False
                            self.__2nd_wolfe_check_done = True
                        else:
                            self.__2nd_wolfe_check_needed = True
                            self.__need_update = True
                    elif c == A:
                        # S3 in [1]
                        # Preforming a secant to propose a value of alpha.
                        self._proposed_alpha = self.__secant_for_alpha(a, A)

                        # checking the proposed point is in range
                        if (self._proposed_alpha < A or
                                self._proposed_alpha > B):
                            # If the point is out of range there is no need to
                            # check wolfe conditions.
                            self.__2nd_wolfe_check_needed = False
                            self.__2nd_wolfe_check_done = True
                        else:
                            self.__2nd_wolfe_check_needed = True
                            self.__need_update = True
                    else:
                        # No new point has been proposed therefore there
                        # is no need to check the wolfe conditions.
                        self.__2nd_wolfe_check_needed = False
                        self.__2nd_wolfe_check_done = True

                    self._updated_minimum_alpha = A
                    self._updated_maximum_alpha = B

                    self.__logic_steps_left = 1
                    self.__j += 1 / 3

                elif (self.__1st_wolfe_check_done and
                        self.__2nd_wolfe_check_done and
                        self.__converged_ls is not True):

                    # S4 in [1], this is only preformed if S2 or S3 was done
                    # and the propsed point was in range.
                    if self.__need_update:
                        a, b = self.__update(a, b, c)
                        self.__need_update = False

                    # ***************** end of secant squared *****************

                    # preforming steps L2 from [1]
                    # determing whether a bisection step should be preformed
                    # i.e if self.__secant_for_interval() didn't shrink the
                    # bracketing interval by the propotion self.__gamma
                    new_width = b - a
                    old_width = (self._maximum_alpha - self._minimum_alpha)

                    if new_width > (self.__gamma * old_width):
                        # preforming bisection
                        c = (a + b) / 2.0
                        a, b = self.__update(a=a, b=b, c=c)

                    # preforming steps L3 from [1]
                    # updating bracketing interval
                    self._minimum_alpha, self._maximum_alpha = a, b
                    self.__j += 1 / 3
                    self.__logic_steps_left = 0

                    # reset logic
                    self.__1st_wolfe_check_needed = False
                    self.__1st_wolfe_check_done = False
                    self.__2nd_wolfe_check_needed = False
                    self.__2nd_wolfe_check_done = False

                # *********************** CONTINUE LOOP ***********************

            if self.__converged_ls:
                self._proposed = (self._current +
                                  self._updated_maximum_alpha * self._px)
            else:
                self._proposed = (self._current +
                                  self._proposed_alpha * self._px)

        # Running, and ready for tell now
        self._ready_for_tell = True
        self._running = True

        # print('')
        # print('finished ask')
        # print('')
        # Return proposed points (just the one) in the search space to evaluate
        return [self._proposed]

    def stop(self):
        """ See :meth:`Optimiser.stop()`. """

        # We use the condition number defined in the pycma code at
        # cma/evolution_strategy.py#L2965.

        cond = norm(self._dfdx_best, ord=np.inf)
        if cond <= 1e-6:
            return 'Convergences i.e gradients ~ 0'
        return False

    def tell(self, reply):
        """ See :meth:`Optimiser.tell()`. """

        # Check ask-tell pattern
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Unpack reply
        proposed_f, proposed_dfdx = reply[0]
        proposed_f = proposed_f
        proposed_dfdx = np.asarray(proposed_dfdx)

        # We need the evaluation of the gradients before we can start the BFGS,
        # the first tell gets these.
        if self._current_f is None:

            # Move to proposed point
            self._current = self._proposed
            self._current_f = np.asarray(proposed_f)
            self._current_dfdx = np.asarray(proposed_dfdx)

            # Update newton direction
            # if it isn't a descent direction the line searches will fail
            # i.e return nonsense
            self._px = - np.matmul(self._B, self._current_dfdx)

            # resetting the number of steps in the current
            # line search iteration.
            self.__j = 0

        # Checking if exact wolfe conditions.
        proposed_grad = np.matmul(np.transpose(proposed_dfdx), self._px)

        wolfe_curvature = (proposed_grad >=
                           self._c2 *
                           np.matmul(np.transpose(self._current_dfdx),
                                     self._px))

        exact_wolfe_suff_dec = (self._c1 *
                                np.matmul(np.transpose(self._current_dfdx),
                                          self._px)
                                >= proposed_f - self._current_f)

        exact_wolfe = (exact_wolfe_suff_dec and wolfe_curvature)

        # Checking if approximate wolfe conditions are meet.
        approx_wolfe_suff_dec = ((2.0 * self._c1 - 1.0) *
                                 np.matmul(np.transpose(self._current_dfdx),
                                           self._px) >= proposed_grad)

        approx_wolfe_applies = proposed_f <= self._current_f + self.epsilon

        approximate_wolfe = (approx_wolfe_suff_dec and wolfe_curvature
                             and approx_wolfe_applies)

        # If wolfe conditions meet the line search is stopped
        # and the inverse hessian matrix and newton direction are updated.
        # If the line search has converged we also accept the
        # steps and update.
        if exact_wolfe or approximate_wolfe or self.__converged_ls:

            print('Number of accepted steps: ', self._k)
            print('step size of alpha accepted: ', self._proposed_alpha)
            print('updating Hessian and changing newton direction')

            # Updating inverse hessian.
            self._B = self.inverse_hessian_update(proposed_f, proposed_dfdx)

            # Move to proposed point
            self._current = self._proposed
            self._current_f = np.asarray(proposed_f)
            self._current_dfdx = np.asarray(proposed_dfdx)

            # storing the accepted value of alpha
            self.__current_alpha = self._proposed_alpha

            # if self.__k == 0:
            #     print('\nThe first accepted value of alpha was: ',
            #           self.__current_alpha)
            #     print('set initial alpha to this in subsequent runs' +
            #           'to speed up computation')

            # Update newton direction
            self._px = - np.matmul(self._B, self._current_dfdx)

            # incrementing the number of accepted steps
            self._k += 1

            # Resetting the number of steps in the current line search
            # iteration as we have accepted a value and completed this
            # line search.
            self.__j = 0

            # resetting line search logic
            self.__1st_wolfe_check_needed = False
            self.__1st_wolfe_check_done = False
            self.__2nd_wolfe_check_needed = False
            self.__2nd_wolfe_check_done = False
            self.__need_update = False

        else:
            # wolfe conditions haven't been meet so we continue the line search
            if self.__1st_wolfe_check_needed:
                self.__1st_wolfe_check_needed = False
                self.__1st_wolfe_check_done = True
            if self.__2nd_wolfe_check_needed:
                self.__2nd_wolfe_check_needed = False
                self.__2nd_wolfe_check_done = True

        # Update xbest and fbest
        if self._fbest > proposed_f:
            self._fbest = proposed_f
            self._dfdx_best = proposed_dfdx
            self._xbest = self._current

        print('')
        print('Hessian updates: ', self._k, ' line steps: ', self.__j,
              ' propose alpha: ', self._proposed_alpha, ' propose points: ',
              self._proposed)

    def __update(self, a, b, c):
        '''
        This function is part of the Hager-Zhang line search method [1].

        Updates the bracketing boundary values of alpha. Ensuring the opposite
        slope conditions are obeyed.

        The opposite slope conditions are:
                                φ(a) ≤ φ(0) + epsilon,
                                φ'(a) < 0 (w.r.t every parameter),
                                φ'(b) ≥ 0 (w.r.t every parameter).
        where φ(α) = f(x+α*px), f() is the function binging minimised, x is
        the current mparameter set, px is the newton direction, and alpha is
        the step size.

        In the first condition, epsilon is a small positive constant. The
        condition demands that the function at the left end point be not much
        bigger than the starting point (i.e. alpha = 0). This is an easy to
        satisfy condition because by assumption, we are in a direction where
        the function value is decreasing. The second and third conditions
        together demand that there is at least one zero of the derivative in
        between a and b. In addition to the interval, the update algorithm
        requires a third point to be supplied. Usually, this point would lie
        within the interval [a, b]. If the point is outside this interval,
        the current interval is returned. If the point lies within the
        interval, the behaviour of the function and derivative value at this
        point is used to squeeze the original interval in a manner that
        preserves the opposite slope conditions.

        :param a: lower bound of alpha
        :param b: upper bound of alpha
        :param c: proposed value of alpha
        :param dfdx_c: vector of gradients at the proposed point i.e alpha = c
        :param f_c: function evaluation at the proposed point i.e alpha = c
        :param f_0: function evaluation at the previous point i.e alpha = 0
        '''

        # Check c is within the bracket conditions (steps U0 from [1]).
        if c < a or c > b:
            # if the proposed point is not in range we return
            # the old brackets unmodified
            return a, b
        else:

            # evaluate function for alpha = c i.e at the proposed boundary
            # point_c = self._current + c * self._px
            # fs_c = self._evaluator.evaluate([point_c])
            f_c, dfdx_c = self.obj_and_grad_func(c)

            # Checking if the opposite slope condition at the upper bound is
            # obeyed by the proposed point
            # (steps U1 from [1]).
            if dfdx_c >= 0.0:
                # Updating the upper bound.
                return a, c

            # Checking if the opposite slope condition at the lower bound is
            # obeyed by the proposed point, if so it is a valid lower bound
            # (steps U2 from [1]).
            elif dfdx_c < 0.0 and f_c <= self._current_f + self.epsilon:
                # Updating the lower bound.
                return c, b

            # The proposed point doesn't obey the opposite slope condition
            # i.e. dfdx_c < 0.0 and f_c > self._current_f + self.epsilon.
            # Checking this is unnecessary as it is the opposite to the above
            # conditions. A secant/bisect can narrow down an interval between
            # the current lower bound and the trial point c.
            else:
                b = c
                return self.__bisect_or_secant(a, b)

    def __bisect_or_secant(self, a, b):
        '''
        This function is part of the Hager-Zhang line search method [1].

        Actual implementation of secant (or bisetc if `self.theta` = 0.5)
        given a bracketing intervale [a, b] used in `__update()` and
        `__initial_bracketing_interval()`. (steps U3a-c from [1])


        :param a: lower bound of alpha
        :param b: upper bound of alpha
        :param c: proposed value of alpha
        '''

        secant = True

        # FIXME:
        # problem is in this while loop a and b merge but don't satisfy
        # opposite slope rule for upper,
        # probably should return when difference between a and b almost
        # nothing???

        # The interval needs updating if the upper bracket has a negative
        # slope and the value of the function at that point is too high.
        # It is not a valid lower bracket but along with the current
        # lower bracket, it encloses another minima. The below function
        # is a loop which tries to narrow the interval so that it
        # satisfies the opposite slope conditions.

        # (steps U3 from [1])

        while secant is True:
            # according to [1]] this while loop is guaranteed to terminate
            # as the intervale between a and b tends to zero

            # Preforming secant (if theta = 0.5 as is default this is a
            # bisection) to propose a new point which hopeful obeys the
            # opposite slope conditions.
            # (steps U3a from [1])
            d = (1.0 - self.theta) * a + self.theta * b

            # Evaluating function for alpha = d i.e at the proposed boundary.
            # point_d = self._current + d * self._px
            # fs = self._evaluator.evaluate([point_d])
            # f_d, dfdx_d = fs[0]

            f_d, dfdd = self.obj_and_grad_func(d)

            # Checking if the opposite slope condition at the upper bound is
            # obeyed by the proposed point.
            # If the proposed point has a positive slope, then we have found a
            # suitable upper bound to bracket a minima within opposite slopes.
            # (still steps U3a from [1])
            converged = self.__very_close(d, b) or self.__very_close(d, a)
            if dfdd >= 0.0 or converged:
                secant = False
                # Updating the upper bound.
                return a, d

            # Checking if the opposite slope condition at the lower bound is
            # obeyed by the proposed point.
            # If the proposed point has a negative slope and the function value
            # at that point is small enough, we can use it as a new lower bound
            # to narrow down the interval.
            # (steps U3b from [1])
            elif dfdd < 0.0 and f_d <= self._current_f + self.epsilon:
                # Updating the lower bound.
                a = d

            # The proposed point doesn't obey the opposite slope condition
            # i.e. dfdx_c < 0.0 and f_c > self._current_f + self.epsilon
            # Checking this is unnecessary as it is the opposite to the above
            # conditions. We are therefore in the same situation as when we
            # started the loop so we update the upper bracket and continue.
            # (steps U3c from [1])
            else:
                b = d

    def __secant_for_alpha(self, a, b):
        '''
        This function is part of the Hager-Zhang line search method [1].

        Preforms a secant step to propose a value of alpha. This is the same as
        the secant routine described in [1].

        :param a: lower bound of alpha
        :param b: upper bound of alpha
        '''

        # Evaluating function for alpha = a to obtain gradients.
        f_a, dfda = self.obj_and_grad_func(a)

        # Evaluating function for alpha = b to obtain gradients.
        f_b, dfdb = self.obj_and_grad_func(b)

        # Preforming secant.
        numerator = a * dfdb - b * dfda
        denominator = dfdb - dfda

        return float(numerator / denominator)

    def __initial_bracket(self, c, rho=5):
        '''
        This function is part of the Hager-Zhang line search method [1].

        This function is used to generate an initial interval [a, b] for alpha
        satisfying the opposite slope conditions and therefore bracketing
        the minimum, beginning with the initial guess [0, c].
        The opposite slope conditions:
                                φ(a) ≤ φ(0) + epsilon,
                                φ'(a) < 0 (w.r.t every parameter),
                                φ'(b) ≥ 0 (w.r.t every parameter).
        where φ(α) = f(x+α*px), f() is the function binging minimised, x is
        the current parameter set, px is the newton direction, and alpha is
        the step size.

        This is the same as the bracket routine described in [1] as steps B0-3

        :param c: initial guess for maximum value of alpha
        :param row: range (1, ∞), expansion factor used in the bracket rule to
                    increase the upper limit c (c_j+1 = row*c_j) until a
                    suitable interval is found that contains the minimum.
        '''

        # (steps B0 from [1])
        # Initiating a list of proposed boundary values of alpha.
        c = [c]
        # Initiating a list of error function evaluations at
        # the proposed boundary values of alpha.
        f_c = []
        # Initiating lower bound.
        a = 0

        # Initiating an iteration counter for the below while loop.
        j = 0
        bracketing = True

        while bracketing:

            # Evaluating function for alpha = c[j]
            # i.e at the proposed boundary.
            f_cj, dfdc_j = self.obj_and_grad_func(c[j])

            # Appending the error function evaluations at proposed boundary
            # values of alpha.
            f_c.append(f_cj)

            # Checking if the opposite slope condition at the upper bound is
            # obeyed by the proposed point. If the slope at the proposed point
            # is positive, then the given points already bracket a minimum.
            # (steps B1 from [1])
            if dfdc_j >= 0.0:
                # Setting the upper bound.
                b = c[j]

                bracketing = False

                # Checking if the non derivative opposite slope condition at
                # the lower bound is obeyed by any of the previously evaluated
                # points and returning the value for which the boundary
                # conditions are as close together as possible.
                for i in range(1, j + 1):
                    if f_c[j - i] <= self._current_f + self.epsilon:
                        a = c[j - i]
                        return a, b
                return a, b

            # Checking if the proposed point doesn't obey the opposite slope
            # condition. This means the upper bracket limit almost works as a
            # new lower limit but the objective function(f_cj) is too large.
            # We therefore need to preform a secant/bisect but the minimum is
            # not yet bracketed.
            # (steps B2 from [1])
            elif dfdc_j < 0.0 and f_cj > self._current_f + self.epsilon:
                # The interval needs updating if the upper bracket has a
                # negative slope and the value of the function at that point
                # is too high. It is not a valid lower bracket but along with
                # the current lower bracket, it encloses another minima. The
                # below function tries to narrow the interval so that it
                # satisfies the opposite slope conditions.
                bracketing = False
                return self.__bisect_or_secant(0.0, c[j])

            # The proposed point obeys the opposite slope condition
            # at the lower bound
            # i.e. dfdx_d < 0.0 and f_d <= self._current_f + self.epsilon.
            # Checking this is unnecessary as it is the opposite to
            # the above conditions. This means the bracket interval needs
            # expanding to ensure a minimum is bracketed.
            # (steps B3 from [1])
            else:
                # Increasing the proposed point by a factor of row to attempt
                # to bracket minimum and trying again.
                c.append(rho * c[j])

                # Increamenting the iteration counter.
                j += 1

    def __initialising(self, k, alpha_k0):
        '''
        This function is part of the Hager-Zhang line search method [1].

        Generate the starting guess of alpha, 'c', used by
        ``__initial_bracket()``. This is the same as the routine
        called initial and described as I0-2 in [1].

        :param k: number of accepted steps/newton direction updates that
        have taken place.
        :param alpha_k0: the alpha value used by the previously accepted
        steps/newton direction
        update. If k = 0 this is the initial alpha the user wants to be used
        '''
        # TODO: The Quadstep option has not been implemented yet.
        # However, the linesearch is functional without is
        QuadStep = False

        # Small factor used in initial guess of step size, range (0, 1).
        # As the initial guess is very crude this is a very small factor
        # to keep the initial step short and close the starting parameters
        # self.x_0.
        ps_0 = self.__ps_0

        # Small factor used in subsequent guesses of step size
        # if Quadstep is true, range (0, 1).
        # TODO: this hasn't been implement yet
        ps_1 = self.__ps_1

        # Sacling factor used in subsequent guesses of step size,
        # range (1, inf).
        ps_2 = self.__ps_2

        # For the first line search do the following
        # (step I0 in [1])
        if k == 0:

            if alpha_k0 is not None:
                # returning user specified initial alpha
                return alpha_k0

            # (step I0a in [1])
            elif np.all(self._x0 != 0.0):
                # Crude step size estimate
                # :math: \alpha = ps_0*||x_0||_\inf / ||dfdx||_\inf
                return ((ps_0 * norm(self._x0, ord=np.inf))
                        / (norm(self._current_dfdx, ord=np.inf)))

            # If self._x0 = 0.0 the above statement would give alpha = 0,
            # hence we use the following estimation.
            # (step I0b in [1])
            elif self._current_f != 0:
                # Crude step size estimate
                # :math: \alpha = ps_0*|f(x_0)| / ||dfdx||^2
                return ((ps_0 * abs(self._current_f))
                        / (pow(norm(self._current_dfdx, ord=2), 2.0)))

            # Otherwise self._current_f = 0.0 and we are already at the
            # minimum therefore return alpha = 1.0 it should not matter what we
            # return in this case as if self._current_f = 0.0 the gradients
            # will also equal zero.
            # (step I0c in [1])
            else:
                return 1.0

        # TODO: implement the below option using a quadratic interpolant ???
        # everything will work without this
        # (step I1 in [1])
        elif QuadStep:

            # point_current_scaled = self._current + ps_1*alpha_k0* self._px
            # fs = self._evaluator.evaluate([point_current_scaled])
            f, df = self.obj_and_grad_func(ps_1 * alpha_k0)

            if f <= self._current_f:
                pass
            #TODO: add quad step option
            pass

        # For the subsequent line search do the following
        # (step I2 in [1])
        else:
            # Increases the step size of the previous accepted step as it
            # is only decreased in subsequent boundary manipulation.
            return ps_2 * alpha_k0

    def __very_close(self, x, y):
        '''     Returns true if x is very close in value to y.   '''
        return np.nextafter(x, y) >= y

    def obj_and_grad_func(self, alpha: float):
        '''
        For a given alpha this returns the values of the objective function
        and it's derivative.
        '''
        point_alpha = self._current + alpha * self._px
        fs_alpha = self._evaluator.evaluate([point_alpha])
        f_alpha, dfdx_alpha = fs_alpha[0]

        dfdalpha = np.matmul(np.transpose(dfdx_alpha), self._px)

        return f_alpha, dfdalpha

    def inverse_hessian_update(self, proposed_f, proposed_dfdx):
        """
        Returns the newly calculated/updated inverse hessian matrix
        by whichever quasi-Newton/ linesearch based optimiser is used
        by the inherited class.
        """
        raise NotImplementedError

    # def __secant2(self, a, b):
    #     '''
    #     This function is part of the Hager-Zhang line search method [1].

    #     This function is referred to as secant^2 and described as steps
    #     S1-4 in [1].

    #     Preforms a secant step to update the bracketing interval of alpha.
    #     Given an interval that brackets a root, this procedure performs an
    #     update of both end points using two intermediate points generated
    #     using the secant interpolation `self.__secant_for_alpha()`.
    #     Assuming the interval [a, b] satisfy the opposite slope conditions.

    #     The opposite slope conditions:
    #                             φ(a) ≤ φ(0) + epsilon ,
    #                             φ'(a) < 0 (w.r.t every parameter),
    #                             φ'(b) ≥ 0 (w.r.t every parameter).
    #     where φ(α) = f(x+α*px), f() is the function binging minimised,
    #     x is the current parameter set, px is the newton direction,
    #     and alpha is the step size.

    #     :param a: power bound of alpha
    #     :param b: upper bound of alpha
    #     '''

    #     # Preforming a secant to propose a value of alpha.
    #     # (step S1 in [1])
    #     c = self.__secant_for_alpha(a,b)
    #     # CHECK IF c SATISFY THE WOLFE CONDITIONS i.e pass to tell!!!!!!!
    #     # IF IT HAS STOP SEARCHING THIS DIRECTION!!!!

    #     # IF WOLFE CONDITIONS AREAN'T MEET DO THIS thing below

    #     # (typically) updating one side of the bracketing interval
    #     print('first update secant')
    #     A, B = self.__update(a, b, c)

    #     # (typically) updating the otherside side of the bracketing interval
    #     # (step S2 in [1])
    #     if c == A:
    #         # Preforming a secant to propose a value of alpha.
    #         C = self.__secant_for_alpha (a, A)

    #     # (step S3 in [1])
    #     if c == B:
    #         # Preforming a secant to propose a value of alpha.
    #         C = self.__secant_for_alpha (b, B)

    #     # (step S4 in [1])
    #     if c == A or c == B:
    #         # CHECK IF C SATISFY THE WOLFE CONDITIONS i.e pass to tell!!!!!!!
    #         # IF IT HAS STOP SEARCHING THIS DIRECTION!!!!

    #         # IF WOLFE CONDITIONS AREAN'T MEET DO THIS thing below

    #         print('second update secant')
    #         A, B = self.__update(A, B, C)

    #     return A, B


class OptimisationController(object):
    """
    Finds the parameter values that minimise an :class:`ErrorMeasure` or
    maximise a :class:`LogPDF`.

    Parameters
    ----------
    function
        An :class:`pints.ErrorMeasure` or a :class:`pints.LogPDF` that
        evaluates points in the parameter space.
    x0
        The starting point for searches in the parameter space. This value may
        be used directly (for example as the initial position of a particle in
        :class:`PSO`) or indirectly (for example as the center of a
        distribution in :class:`XNES`).
    sigma0
        An optional initial standard deviation around ``x0``. Can be specified
        either as a scalar value (one standard deviation for all coordinates)
        or as an array with one entry per dimension. Not all methods will use
        this information.
    boundaries
        An optional set of boundaries on the parameter space.
    transform
        An optional :class:`pints.Transformation` to allow the optimiser to
        search in a transformed parameter space. If used, points shown or
        returned to the user will first be detransformed back to the original
        space.
    method
        The class of :class:`pints.Optimiser` to use for the optimisation.
        If no method is specified, :class:`CMAES` is used.
    """

    def __init__(
            self, function, x0, sigma0=None, boundaries=None, transform=None,
            method=None):

        # Convert x0 to vector
        # This converts e.g. (1, 7) shapes to (7, ), giving users a bit more
        # freedom with the exact shape passed in. For example, to allow the
        # output of LogPrior.sample(1) to be passed in.
        x0 = pints.vector(x0)

        # Check dimension of x0 against function
        if function.n_parameters() != len(x0):
            raise ValueError(
                'Starting point must have same dimension as function to'
                ' optimise.')

        # Check if minimising or maximising
        self._minimising = not isinstance(function, pints.LogPDF)

        # Apply a transformation (if given). From this point onward the
        # optimiser will see only the transformed search space and will know
        # nothing about the model parameter space.
        if transform is not None:
            # Convert error measure or log pdf
            if self._minimising:
                function = transform.convert_error_measure(function)
            else:
                function = transform.convert_log_pdf(function)

            # Convert initial position
            x0 = transform.to_search(x0)

            # Convert sigma0, if provided
            if sigma0 is not None:
                sigma0 = transform.convert_standard_deviation(sigma0, x0)
            if boundaries:
                boundaries = transform.convert_boundaries(boundaries)

        # Store transform for later detransformation: if using a transform, any
        # parameters logged to the filesystem or printed to screen should be
        # detransformed first!
        self._transform = transform

        # Store function
        if self._minimising:
            self._function = function
        else:
            self._function = pints.ProbabilityBasedError(function)
        del(function)

        # Create optimiser
        if method is None:
            method = pints.CMAES
        elif not issubclass(method, pints.Optimiser):
            raise ValueError('Method must be subclass of pints.Optimiser.')
        self._optimiser = method(x0, sigma0, boundaries)
        if issubclass(method, pints.LineSearchBasedOptimiser):
            self._optimiser._set_function_evaluator(self._function)

        # Check if sensitivities are required
        self._needs_sensitivities = self._optimiser.needs_sensitivities()

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_interval()

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

        # Maximum unchanged iterations
        self._max_unchanged_iterations = None
        self._min_significant_change = 1
        self.set_max_unchanged_iterations()

        # Threshold value
        self._threshold = None

        # Post-run statistics
        self._evaluations = None
        self._iterations = None
        self._time = None

    def evaluations(self):
        """
        Returns the number of evaluations performed during the last run, or
        ``None`` if the controller hasn't ran yet.
        """
        return self._evaluations

    def iterations(self):
        """
        Returns the number of iterations performed during the last run, or
        ``None`` if the controller hasn't ran yet.
        """
        return self._iterations

    def max_iterations(self):
        """
        Returns the maximum iterations if this stopping criterion is set, or
        ``None`` if it is not. See :meth:`set_max_iterations()`.
        """
        return self._max_iterations

    def max_unchanged_iterations(self):
        """
        Returns a tuple ``(iterations, threshold)`` specifying a maximum
        unchanged iterations stopping criterion, or ``(None, None)`` if no such
        criterion is set. See :meth:`set_max_unchanged_iterations()`.
        """
        if self._max_unchanged_iterations is None:
            return (None, None)
        return (self._max_unchanged_iterations, self._min_significant_change)

    def optimiser(self):
        """
        Returns the underlying optimiser object, allowing detailed
        configuration.
        """
        return self._optimiser

    def parallel(self):
        """
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False

    def run(self):
        """
        Runs the optimisation, returns a tuple ``(xbest, fbest)``.
        """
        # Can only run once for each controller instance
        if self._has_run:
            raise RuntimeError("Controller is valid for single use only")
        self._has_run = True

        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= (self._max_iterations is not None)
        has_stopping_criterion |= (self._max_unchanged_iterations is not None)
        has_stopping_criterion |= (self._threshold is not None)
        if not has_stopping_criterion:
            raise ValueError('At least one stopping criterion must be set.')

        # Iterations and function evaluations
        iteration = 0
        evaluations = 0

        # Unchanged iterations count (used for stopping or just for
        # information)
        unchanged_iterations = 0

        # Choose method to evaluate
        f = self._function
        if self._needs_sensitivities:
            f = f.evaluateS1

        # Create evaluator object
        if self._parallel:
            # Get number of workers
            n_workers = self._n_workers

            # For population based optimisers, don't use more workers than
            # particles!
            if isinstance(self._optimiser, PopulationBasedOptimiser):
                n_workers = min(n_workers, self._optimiser.population_size())
            evaluator = pints.ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Keep track of best position and score
        fbest = float('inf')

        # Internally we always minimise! Keep a 2nd value to show the user
        fbest_user = fbest if self._minimising else -fbest

        # Set up progress reporting
        next_message = 0

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            if self._log_to_screen:
                # Show direction
                if self._minimising:
                    print('Minimising error measure')
                else:
                    print('Maximising LogPDF')

                # Show method
                print('Using ' + str(self._optimiser.name()))

                # Show parallelisation
                if self._parallel:
                    print('Running in parallel with ' + str(n_workers) +
                          ' worker processes.')
                else:
                    print('Running in sequential mode.')

            # Show population size
            pop_size = 1
            if isinstance(self._optimiser, PopulationBasedOptimiser):
                pop_size = self._optimiser.population_size()
                if self._log_to_screen:
                    print('Population size: ' + str(pop_size))

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            max_iter_guess = max(self._max_iterations or 0, 10000)
            max_eval_guess = max_iter_guess * pop_size
            logger.add_counter('Iter.', max_value=max_iter_guess)
            logger.add_counter('Eval.', max_value=max_eval_guess)
            logger.add_float('Best')
            self._optimiser._log_init(logger)
            logger.add_time('Time m:s')

        # Start searching
        timer = pints.Timer()
        running = True
        try:
            while running:
                # Get points
                xs = self._optimiser.ask()

                # Calculate scores
                fs = evaluator.evaluate(xs)

                # Perform iteration
                self._optimiser.tell(fs)

                # Check if new best found
                fnew = self._optimiser.fbest()
                if fnew < fbest:
                    # Check if this counts as a significant change
                    if np.abs(fnew - fbest) < self._min_significant_change:
                        unchanged_iterations += 1
                    else:
                        unchanged_iterations = 0

                    # Update best
                    fbest = fnew

                    # Update user value of fbest
                    fbest_user = fbest if self._minimising else -fbest
                else:
                    unchanged_iterations += 1

                # Update evaluation count
                evaluations += len(fs)

                # Show progress
                if logging and iteration >= next_message:
                    # Log state
                    logger.log(iteration, evaluations, fbest_user)
                    self._optimiser._log_write(logger)
                    logger.log(timer.time())

                    # Choose next logging point
                    if iteration < self._message_warm_up:
                        next_message = iteration + 1
                    else:
                        next_message = self._message_interval * (
                            1 + iteration // self._message_interval)

                # Update iteration count
                iteration += 1

                #
                # Check stopping criteria
                #

                # Maximum number of iterations
                if (self._max_iterations is not None and
                        iteration >= self._max_iterations):
                    running = False
                    halt_message = ('Halting: Maximum number of iterations ('
                                    + str(iteration) + ') reached.')

                # Maximum number of iterations without significant change
                halt = (self._max_unchanged_iterations is not None and
                        unchanged_iterations >= self._max_unchanged_iterations)
                if halt:
                    running = False
                    halt_message = ('Halting: No significant change for ' +
                                    str(unchanged_iterations) + ' iterations.')

                # Threshold value
                if self._threshold is not None and fbest < self._threshold:
                    running = False
                    halt_message = ('Halting: Objective function crossed'
                                    ' threshold: ' + str(self._threshold) +
                                    '.')

                # Error in optimiser
                error = self._optimiser.stop()
                if error:   # pragma: no cover
                    running = False
                    halt_message = ('Halting: ' + str(error))

        except (Exception, SystemExit, KeyboardInterrupt):  # pragma: no cover
            # Unexpected end!
            # Show last result and exit
            print('\n' + '-' * 40)
            print('Unexpected termination.')
            print('Current best score: ' + str(fbest))
            print('Current best position:')

            # Inverse transform search parameters
            if self._transform:
                xbest = self._transform.to_model(self._optimiser.xbest())
            else:
                xbest = self._optimiser.xbest()

            for p in xbest:
                print(pints.strfloat(p))
            print('-' * 40)
            raise

        # Stop timer
        self._time = timer.time()

        # Log final values and show halt message
        if logging:
            logger.log(iteration, evaluations, fbest_user)
            self._optimiser._log_write(logger)
            logger.log(self._time)
            if self._log_to_screen:
                print(halt_message)

        # Save post-run statistics
        self._evaluations = evaluations
        self._iterations = iteration

        # Inverse transform search parameters
        if self._transform:
            xbest = self._transform.to_model(self._optimiser.xbest())
        else:
            xbest = self._optimiser.xbest()

        # Return best position and score
        return xbest, fbest_user

    def set_log_interval(self, iters=20, warm_up=3):
        """
        Changes the frequency with which messages are logged.

        Parameters
        ----------
        ``interval``
            A log message will be shown every ``iters`` iterations.
        ``warm_up``
            A log message will be shown every iteration, for the first
            ``warm_up`` iterations.
        """
        iters = int(iters)
        if iters < 1:
            raise ValueError('Interval must be greater than zero.')
        warm_up = max(0, int(warm_up))

        self._message_interval = iters
        self._message_warm_up = warm_up

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

    def set_max_iterations(self, iterations=10000):
        """
        Adds a stopping criterion, allowing the routine to halt after the
        given number of ``iterations``.

        This criterion is enabled by default. To disable it, use
        ``set_max_iterations(None)``.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError(
                    'Maximum number of iterations cannot be negative.')
        self._max_iterations = iterations

    def set_max_unchanged_iterations(self, iterations=200, threshold=1e-11):
        """
        Adds a stopping criterion, allowing the routine to halt if the
        objective function doesn't change by more than ``threshold`` for the
        given number of ``iterations``.

        This criterion is enabled by default. To disable it, use
        ``set_max_unchanged_iterations(None)``.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError(
                    'Maximum number of iterations cannot be negative.')

        threshold = float(threshold)
        if threshold < 0:
            raise ValueError('Minimum significant change cannot be negative.')

        self._max_unchanged_iterations = iterations
        self._min_significant_change = threshold

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

    def set_threshold(self, threshold):
        """
        Adds a stopping criterion, allowing the routine to halt once the
        objective function goes below a set ``threshold``.

        This criterion is disabled by default, but can be enabled by calling
        this method with a valid ``threshold``. To disable it, use
        ``set_treshold(None)``.
        """
        if threshold is None:
            self._threshold = None
        else:
            self._threshold = float(threshold)

    def threshold(self):
        """
        Returns the threshold stopping criterion, or ``None`` if no threshold
        stopping criterion is set. See :meth:`set_threshold()`.
        """
        return self._threshold

    def time(self):
        """
        Returns the time needed for the last run, in seconds, or ``None`` if
        the controller hasn't run yet.
        """
        return self._time


class Optimisation(OptimisationController):
    """ Deprecated alias for :class:`OptimisationController`. """

    def __init__(
            self, function, x0, sigma0=None, boundaries=None, transform=None,
            method=None):
        # Deprecated on 2019-02-12
        import warnings
        warnings.warn(
            'The class `pints.Optimisation` is deprecated.'
            ' Please use `pints.OptimisationController` instead.')
        super(Optimisation, self).__init__(
            function, x0, sigma0=None, boundaries=None, transform=None,
            method=None)


def optimise(
        function, x0, sigma0=None, boundaries=None, transform=None,
        method=None):
    """
    Finds the parameter values that minimise an :class:`ErrorMeasure` or
    maximise a :class:`LogPDF`.

    Parameters
    ----------
    function
        An :class:`pints.ErrorMeasure` or a :class:`pints.LogPDF` that
        evaluates points in the parameter space.
    x0
        The starting point for searches in the parameter space. This value may
        be used directly (for example as the initial position of a particle in
        :class:`PSO`) or indirectly (for example as the center of a
        distribution in :class:`XNES`).
    sigma0
        An optional initial standard deviation around ``x0``. Can be specified
        either as a scalar value (one standard deviation for all coordinates)
        or as an array with one entry per dimension. Not all methods will use
        this information.
    boundaries
        An optional set of boundaries on the parameter space.
    transform
        An optional :class:`pints.Transformation` to allow the optimiser to
        search in a transformed parameter space. If used, points shown or
        returned to the user will first be detransformed back to the original
        space.
    method
        The class of :class:`pints.Optimiser` to use for the optimisation.
        If no method is specified, :class:`CMAES` is used.

    Returns
    -------
    xbest : numpy array
        The best parameter set obtained
    fbest : float
        The corresponding score.
    """
    return OptimisationController(
        function, x0, sigma0, boundaries, transform, method).run()


class TriangleWaveTransform(object):
    """
    Transforms from unbounded to (rectangular) bounded parameter space using a
    periodic triangle-wave transform.

    Note: The transform is applied _inside_ optimisation methods, there is no
    need to wrap this around your own problem or score function.

    This can be applied as a transformation on ``x`` to implement _rectangular_
    boundaries in methods with no natural boundary mechanism. It effectively
    mirrors the search space at every boundary, leading to a continuous (but
    non-smooth) periodic landscape. While this effectively creates an infinite
    number of minima/maxima, each one maps to the same point in parameter
    space.

    It should work well for methods that maintain a single search position or a
    single search distribution (e.g. :class:`CMAES`, :class:`xNES`,
    :class:`SNES`), which will end up in one of the many mirror images.
    However, for methods that use independent search particles (e.g.
    :class:`PSO`) it could lead to a scattered population, with different
    particles exploring different mirror images. Other strategies should be
    used for such problems.
    """

    def __init__(self, boundaries):
        self._lower = boundaries.lower()
        self._upper = boundaries.upper()
        self._range = self._upper - self._lower
        self._range2 = 2 * self._range

    def __call__(self, x):
        y = np.remainder(x - self._lower, self._range2)
        z = np.remainder(y, self._range)
        return ((self._lower + z) * (y < self._range)
                + (self._upper - z) * (y >= self._range))


def curve_fit(f, x, y, p0, boundaries=None, threshold=None, max_iter=None,
              max_unchanged=200, verbose=False, parallel=False, method=None):
    """
    Fits a function ``f(x, *p)`` to a dataset ``(x, y)`` by finding the value
    of ``p`` for which ``sum((y - f(x, *p))**2) / n`` is minimised (where ``n``
    is the number of entries in ``y``).

    Returns a tuple ``(xbest, fbest)`` with the best position found, and the
    corresponding value ``fbest = f(xbest)``.

    Parameters
    ----------
    f : callable
        A function or callable class to be minimised.
    x
        The values of an independent variable, at which ``y`` was recorded.
    y
        Measured values ``y = f(x, p) + noise``.
    p0
        An initial guess for the optimal parameters ``p``.
    boundaries
        An optional :class:`pints.Boundaries` object or a tuple
        ``(lower, upper)`` specifying lower and upper boundaries for the
        search. If no boundaries are provided an unbounded search is run.
    threshold
        An optional absolute threshold stopping criterium.
    max_iter
        An optional maximum number of iterations stopping criterium.
    max_unchanged
        A stopping criterion based on the maximum number of successive
        iterations without a signficant change in ``f`` (see
        :meth:`pints.OptimisationController`).
    verbose
        Set to ``True`` to print progress messages to the screen.
    parallel
        Allows parallelisation to be enabled.
        If set to ``True``, the evaluations will happen in parallel using a
        number of worker processes equal to the detected cpu core count. The
        number of workers can be set explicitly by setting ``parallel`` to an
        integer greater than 0.
    method
        The :class:`pints.Optimiser` to use. If no method is specified,
        ``pints.CMAES`` is used.

    Returns
    -------
    xbest : numpy array
        The best parameter set obtained.
    fbest : float
        The corresponding score.

    Example
    -------
    ::

        import numpy as np
        import pints

        def f(x, a, b, c):
            return a + b * x + c * x ** 2

        x = np.linspace(-5, 5, 100)
        y = f(x, 1, 2, 3) + np.random.normal(0, 1)

        p0 = [0, 0, 0]
        popt = pints.curve_fit(f, x, y, p0)

    """
    # Test function
    if not callable(f):
        raise ValueError('The argument `f` must be callable.')

    # Get problem dimension from p0
    d = len(p0)

    # First dimension of x and y must agree
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            'The first dimension of `x` and `y` must be the same.')

    # Check boundaries
    if not (boundaries is None or isinstance(boundaries, pints.Boundaries)):
        lower, upper = boundaries
        boundaries = pints.RectangularBoundaries(lower, upper)

    # Create an error measure
    e = _CurveFitError(f, d, x, y)

    # Set up optimisation
    opt = pints.OptimisationController(
        e, p0, boundaries=boundaries, method=method)

    # Set stopping criteria
    opt.set_threshold(threshold)
    opt.set_max_iterations(max_iter)
    opt.set_max_unchanged_iterations(max_unchanged)

    # Set parallelisation
    opt.set_parallel(parallel)

    # Set output
    opt.set_log_to_screen(True if verbose else False)

    # Run and return
    return opt.run()


class _CurveFitError(pints.ErrorMeasure):
    """ Error measure for :meth:`curve_fit()`. """

    def __init__(self, function, dimension, x, y):
        self.f = function
        self.d = dimension
        self.x = x
        self.y = y
        self.n = 1 / np.product(y.shape)    # Total number of points in data

    def n_parameters(self):
        return self.d

    def __call__(self, p):
        return np.sum((self.y - self.f(self.x, *p))**2) * self.n


def fmin(f, x0, args=None, boundaries=None, threshold=None, max_iter=None,
         max_unchanged=200, verbose=False, parallel=False, method=None):
    """
    Minimises a callable function ``f``, starting from position ``x0``, using a
    :class:`pints.Optimiser`.

    Returns a tuple ``(xbest, fbest)`` with the best position found, and the
    corresponding value ``fbest = f(xbest)``.

    Parameters
    ----------
    f
        A function or callable class to be minimised.
    x0
        The initial point to search at. Must be a 1-dimensional sequence (e.g.
        a list or a numpy array).
    args
        An optional tuple of extra arguments for ``f``.
    boundaries
        An optional :class:`pints.Boundaries` object or a tuple
        ``(lower, upper)`` specifying lower and upper boundaries for the
        search. If no boundaries are provided an unbounded search is run.
    threshold
        An optional absolute threshold stopping criterium.
    max_iter
        An optional maximum number of iterations stopping criterium.
    max_unchanged
        A stopping criterion based on the maximum number of successive
        iterations without a signficant change in ``f`` (see
        :meth:`pints.OptimisationController`).
    verbose
        Set to ``True`` to print progress messages to the screen.
    parallel
        Allows parallelisation to be enabled.
        If set to ``True``, the evaluations will happen in parallel using a
        number of worker processes equal to the detected cpu core count. The
        number of workers can be set explicitly by setting ``parallel`` to an
        integer greater than 0.
    method
        The :class:`pints.Optimiser` to use. If no method is specified,
        ``pints.CMAES`` is used.

    Example
    -------
    ::

        import pints

        def f(x):
            return (x[0] - 3) ** 2 + (x[1] + 5) ** 2

        xopt, fopt = pints.fmin(f, [1, 1])
    """
    # Test function
    if not callable(f):
        raise ValueError('The argument `f` must be callable.')

    # Get problem dimension from x0
    d = len(x0)

    # Test extra arguments
    if args is not None:
        args = tuple(args)

    # Check boundaries
    if not (boundaries is None or isinstance(boundaries, pints.Boundaries)):
        lower, upper = boundaries
        boundaries = pints.RectangularBoundaries(lower, upper)

    # Create an error measure
    e = _FminError(f, d) if args is None else _FminErrorWithArgs(f, d, args)

    # Set up optimisation
    opt = pints.OptimisationController(
        e, x0, boundaries=boundaries, method=method)

    # Set stopping criteria
    opt.set_threshold(threshold)
    opt.set_max_iterations(max_iter)
    opt.set_max_unchanged_iterations(max_unchanged)

    # Set parallelisation
    opt.set_parallel(parallel)

    # Set output
    opt.set_log_to_screen(True if verbose else False)

    # Run and return
    return opt.run()


class _FminError(pints.ErrorMeasure):
    """ Error measure for :meth:`fmin()`. """

    def __init__(self, f, d):
        self.f = f
        self.d = d

    def n_parameters(self):
        return self.d

    def __call__(self, x):
        return self.f(x)


class _FminErrorWithArgs(pints.ErrorMeasure):
    """ Error measure for :meth:`fmin()` for functions with args. """

    def __init__(self, f, d, args):
        self.f = f
        self.d = d
        self.args = args

    def n_parameters(self):
        return self.d

    def __call__(self, x):
        return self.f(x, *self.args)

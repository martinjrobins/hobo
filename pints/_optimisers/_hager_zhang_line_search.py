#
# Fixed learning-rate gradient descent.
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


class HagerZhang(pints.Optimiser):
    """
    Gradient-descent method with a fixed learning rate.
    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(HagerZhang, self).__init__(x0, sigma0, boundaries)

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
        self.__px = None

        # number of accepted steps/ newton direction updates
        self._k = 0

        # number of steps in the current line search iteration
        self.__j = 0
        self.__logic_steps_left = 0

        # maximum number of line search steps before a successful point

        # packed reply of objective function and gradient evaluations
        # needed by generator implementation
        # this approach is taken rather than the send() keyword to
        # back date past Python 2.5
        self.__reply_f_and_dfdx = None

        # Current point, score, and gradient
        self._current = self._x0
        self._current_f = None
        self._current_dfdx = None

        # Proposed next point (read-only, so can be passed to user)
        self._proposed = self._x0
        self._proposed.setflags(write=False)

        self.__convergence = False

        # ask generator
        self.__ask_generator = self.__ask_generator()

        # need to check wolfe conditions
        self.wolfe_check = False
        # continue main body of line search loop
        self.__line_search_looping = None
        # wheather the wolfe conditions have been meet
        self.__wolfe_cond_meet = False

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Hager-Zhang Line Search'

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
            raise ValueError('''Invalid wolfe line search parameters!
                                Parameters must meet the following conditions:
                                0 < c1 < 0.5 and c1 < c2 < 1.0
                                currently c1 = ''', cs[0], 'and  c2 = ', cs[1])

    def set_search_direction(self, px):
        """
        Sets the search direction vector px.
        If this is not a descent direction the line searches will fail
        i.e return nonsense or actually fail

        Parameters
        ----------
        px: vector
        search direction vector this must be of the same length as x
        """
        # TODO: maybe add a check that px contains numbers?
        if(len(px) == self._n_parameters):
            self.__px = np.asarray(px)
        else:
            raise ValueError('Invalid search direction, px must be of length '
                             + str(self._n_parameters))

    def xbest(self):
        """ See :meth:`Optimiser.xbest()`. """
        return self._xbest

    def stop(self):
        """ See :meth:`Optimiser.stop()`. """

        if self.__wolfe_cond_meet:
            # wolfe conditions have been meet
            # or convergence has occurred

            # resetting so optimiser can be recalled
            self.__wolfe_cond_meet = False
            self._running = False
            return 'Wolfe conditions meet'
        else:

            return False

    def ask(self):
        """ See :meth:`Optimiser.ask()`. """
        if np.any(self.__px) is None:
            raise ValueError(''''Warning search direction has not been set
                                please set search direction using:
                                set_search_direction(px = '')
                                before optimising.
                                ''')

        self.wolfe_check, self._proposed = next(self.__ask_generator)

        # Running, and ready for tell now
        self._ready_for_tell = True
        self._running = True

        return [self._proposed]

    def __ask_generator(self):
        """ See :meth:`Optimiser.ask()`. """
        # check wolfe conditions
        # yes_wolfe_check = True
        # don't check wolfe conditions
        no_wolfe_check = False
        # if not self._running:
        self._proposed = np.asarray(self._x0)
        # need to evaluate initial position
        yield no_wolfe_check, self._proposed

        # working out an initial stepsize value alpha
        alpha_0 = self.__initialising(k=self._k,
                                      alpha_k0=self.__current_alpha)

        # Creating an initial bracketing interval of alpha values
        # satisfying the opposite slope condition (see function
        # docstring) beginning with initial guess [0, alpha_initial].
        initial_bracket_gen = self.__initial_bracket(c=alpha_0)
        continue_bracketing, postion = next(initial_bracket_gen)
        while continue_bracketing is True:
            # print(postion)
            yield no_wolfe_check, postion
            continue_bracketing, postion = next(initial_bracket_gen)
        # saving bracket information
        bracket = postion
        self._minimum_alpha = bracket[0]
        self._maximum_alpha = bracket[1]

        # loop of steps L0-L3 from [1]
        self.__line_search_looping = True
        while self.__line_search_looping:

            a = self._minimum_alpha
            b = self._maximum_alpha

            converged = self.__very_close(a, b)
            if converged:
                pass
            # step L1 of line search from [1]
            # calling secant2 as a generator
            secant2_gen = self.__secant2(a, b)
            wolfe_check, continue_secant2, postion = next(secant2_gen)
            while continue_secant2 is True:
                # print(postion)
                yield wolfe_check, postion
                wolfe_check, continue_secant2, postion = next(secant2_gen)

            # step L2 of line search from [1]
            # determing whether a bisection step should be preformed
            # i.e if self.__secant_for_interval() didn't shrink the
            # bracketing interval by the propotion self.__gamma
            new_width = b - a
            old_width = (self._maximum_alpha - self._minimum_alpha)

            if new_width > (self.__gamma * old_width):
                # preforming bisection
                c = (a + b) / 2.0
                a, b = self.__update(a=a, b=b, c=c)

            # step L3 of the line search from [1]
            # updating bracket interval
            self._minimum_alpha = a
            self._maximum_alpha = b

    def tell(self, reply):
        """ See :meth:`Optimiser.tell()`. """

        # print('in tell')

        # Check ask-tell pattern
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Unpack reply
        self.__reply_f_and_dfdx = reply
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

        elif self.wolfe_check is True:
            # Checking if exact wolfe conditions are meet.
            proposed_grad = np.matmul(np.transpose(proposed_dfdx), self.__px)

            wolfe_curvature = (proposed_grad >= self._c2 *
                               np.matmul(np.transpose(self._current_dfdx),
                                         self.__px))

            exact_wolfe_suff_dec = (self._c1 *
                                    np.matmul(np.transpose(self._current_dfdx),
                                              self.__px)
                                    >= proposed_f - self._current_f)

            exact_wolfe = (exact_wolfe_suff_dec and wolfe_curvature)

            # Checking if approximate wolfe conditions are meet.
            apprx_wolfe_suff_dec = ((2.0 * self._c1 - 1.0) *
                                    np.matmul(np.transpose(self._current_dfdx),
                                              self.__px) >= proposed_grad)

            apprx_wolfe_applies = proposed_f <= self._current_f + self.epsilon

            approximate_wolfe = (apprx_wolfe_suff_dec and wolfe_curvature
                                 and apprx_wolfe_applies)

            # If wolfe conditions are meet the line search is stopped.
            # If the line search has converged we also accept the
            # step and stop the line search.
            # The algorithm is set to be able to be called again.
            if exact_wolfe or approximate_wolfe or self.__converged_ls:

                # stopping condition have been meet i.e wolfe conditions
                self.__wolfe_cond_meet = True

                # Move to proposed point
                self._current = self._proposed
                self._current_f = np.asarray(proposed_f)
                self._current_dfdx = np.asarray(proposed_dfdx)

                # storing the accepted value of alpha
                self.__current_alpha = self._proposed_alpha

                # # Update newton direction
                # self._px = - np.matmul(self._B, self._current_dfdx)

                # incrementing the number of accepted steps
                self._k += 1

                # terminating line search loop
                self.__line_search_looping = False

        # Update xbest and fbest
        if self._fbest > proposed_f:
            self._fbest = proposed_f
            self._dfdx_best = proposed_dfdx
            self._xbest = self._current

        # print('finished tell')

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
            elif self._current_f != 0.0:
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

    def __position_after_step(self, alpha):
        '''
        For a given alpha this returns the position from this step
        relative to the starting point and search direction
        '''
        point_alpha = self._current + alpha * self.__px
        return True, point_alpha

    def __obj_and_grad_unpack(self, x):
        '''
        Unpacks the objective function and gradient from
        function evaluation after the proposed step alpha
        '''
        fs_alpha = x
        f_alpha, dfdx_alpha = fs_alpha[0]

        dfdalpha = np.matmul(np.transpose(dfdx_alpha), self.__px)

        return f_alpha, dfdalpha

    def __bisect_or_secant(self, a, b):
        '''
        This function is part of the Hager-Zhang line search method [1].

        Actual implementation of secant (or bisect if `self.theta` = 0.5)
        given a bracketing intervale [a, b] used in `__update()` and
        `__initial_bracketing_interval()`. (steps U3a-c from [1])


        :param a: lower bound of alpha
        :param b: upper bound of alpha
        :param c: proposed value of alpha
        '''

        secant = True

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
            # point_d = self._current + d * self.__px
            # fs = self._evaluator.evaluate([point_d])
            # f_d, dfdx_d = fs[0]

            yield self.__position_after_step(d)
            f_d, dfdd = self.__obj_and_grad_unpack(self.__reply_f_and_dfdx)

            # Checking if the opposite slope condition at the upper bound is
            # obeyed by the proposed point.
            # If the proposed point has a positive slope, then we have found a
            # suitable upper bound to bracket a minima within opposite slopes.
            # (still steps U3a from [1])
            converged = self.__very_close(d, b) or self.__very_close(d, a)
            if dfdd >= 0.0 or converged:
                secant = False
                # Updating the upper bound.
                # self.__bracket = (a, d)
                yield False, (a, b)

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
            # print('c[',j,']: ', c[j])
            yield self.__position_after_step(c[j])
            # print('self.__reply_f_and_dfdx: ', self.__reply_f_and_dfdx)
            f_cj, dfdc_j = self.__obj_and_grad_unpack(self.__reply_f_and_dfdx)

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
                        yield False, (a, b)
                yield False, (a, b)

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
                bisect_or_secant = True
                bi_sec_generator = self.__bisect_or_secant(0.0, c[j])
                while bisect_or_secant is True:
                    obj_and_grad, output = next(bi_sec_generator)
                    if obj_and_grad is True:
                        yield True, output
                    else:
                        # finished __bisect_or_secant
                        bisect_or_secant = False
                        # returning new brackets
                        # and ending __initial_bracket()
                        yield False, output

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
            yield False, (a, b)
        else:

            # evaluate function for alpha = c i.e at the proposed boundary

            yield self.__position_after_step(c)
            f_c, dfdx_c = self.__obj_and_grad_unpack(self.__reply_f_and_dfdx)

            # Checking if the opposite slope condition at the upper bound is
            # obeyed by the proposed point
            # (steps U1 from [1]).
            if dfdx_c >= 0.0:
                # Updating the upper bound.
                yield False, (a, c)

            # Checking if the opposite slope condition at the lower bound is
            # obeyed by the proposed point, if so it is a valid lower bound
            # (steps U2 from [1]).
            elif dfdx_c < 0.0 and f_c <= self._current_f + self.epsilon:
                # Updating the lower bound.
                yield False, (c, b)

            # The proposed point doesn't obey the opposite slope condition
            # i.e. dfdx_c < 0.0 and f_c > self._current_f + self.epsilon.
            # Checking this is unnecessary as it is the opposite to the above
            # conditions. A secant/bisect can narrow down an interval between
            # the current lower bound and the trial point c.
            else:
                print('in here !!!')
                b = c
                bisect_or_secant = True
                bi_sec_generator = self.__bisect_or_secant(a, b)
                while bisect_or_secant is True:
                    obj_and_grad, output = next(bi_sec_generator)
                    if obj_and_grad is True:
                        yield True, output
                    else:
                        # finished __bisect_or_secant
                        bisect_or_secant = False
                        # returning new brackets
                        # and ending __initial_bracket()
                        yield False, output

    def __secant_for_alpha(self, a, b, dfda, dfdb):
        '''
        This function is part of the Hager-Zhang line search method [1].

        Preforms a secant step to propose a value of alpha. This is the same as
        the secant routine described in [1].

        :param a: lower bound of alpha
        :param b: upper bound of alpha
        '''

        # # Evaluating function for alpha = a to obtain gradients.
        # f_a, dfda = self.obj_and_grad_func(a)

        # # Evaluating function for alpha = b to obtain gradients.
        # f_b, dfdb = self.obj_and_grad_func(b)

        # Preforming secant.
        numerator = a * dfdb - b * dfda
        denominator = dfdb - dfda
        if denominator == 0.0:
            raise ValueError('Dividing by zero in __secant_for_alpha')
        return float(numerator / denominator)

    def __secant2(self, a, b):
        '''
        This function is part of the Hager-Zhang line search method [1].

        This function is referred to as secant^2 and described as steps
        S1-4 in [1].

        Preforms a secant step to update the bracketing interval of alpha.
        Given an interval that brackets a root, this procedure performs an
        update of both end points using two intermediate points generated
        using the secant interpolation `self.__secant_for_alpha()`.
        Assuming the interval [a, b] satisfy the opposite slope conditions.

        The opposite slope conditions:
                                φ(a) ≤ φ(0) + epsilon ,
                                φ'(a) < 0 (w.r.t every parameter),
                                φ'(b) ≥ 0 (w.r.t every parameter).
        where φ(α) = f(x+α*px), f() is the function binging minimised,
        x is the current parameter set, px is the newton direction,
        and alpha is the step size.

        :param a: power bound of alpha
        :param b: upper bound of alpha

        :yield wolfe_check, grad_calc, output: Wolfe_check is a boolean
        determining whether the wolfe conditions need to be checked,
        continue_secant is a booleian informing the generator controller
        that the objective function and gradients need to be calculated for
        the output which will be a position vector, and output can be
        either a position vector or a step size (alpha) bracket (when
        wolfe_check, and grad_calc are false)
        '''

        # Preforming a secant to propose a value of alpha.
        # (step S1 in [1])
        continue_secant, position = self.__position_after_step(a)
        yield False, continue_secant, position
        f_a, dfdx_a = self.__obj_and_grad_unpack(self.__reply_f_and_dfdx)
        continue_secant, position = self.__position_after_step(b)
        yield False, continue_secant, position
        f_b, dfdx_b = self.__obj_and_grad_unpack(self.__reply_f_and_dfdx)
        c = self.__secant_for_alpha(a, b, dfdx_a, dfdx_b)
        # print('c: ', c)

        # CHECK IF c SATISFY THE WOLFE CONDITIONS i.e pass to tell!!!
        # IF IT HAS STOP SEARCHING THIS DIRECTION!!!!
        # checking the proposed point is in range
        if c >= a and c <= b:
            # If the point is out of range there is no need to
            # check wolfe conditions.
            self._proposed_alpha = c
            continue_secant, position = self.__position_after_step(c)
            yield True, True, position
        # IF WOLFE CONDITIONS AREN'T MEET DO THIS THING BELOW

        # (typically) updating one side of the bracketing interval
        # print('first update secant')
        update_gen = self.__update(a=a, b=b, c=c)
        continue_update, output = next(update_gen)
        while continue_update is True:
            # evaluating objective function and gradient
            yield False, True, output
            # continuing __update
            continue_update, output = next(update_gen)
        # when update has finished the output is the
        # updated bracket
        A, B = output[0], output[1]
        del(update_gen)
        del(continue_update)
        # print('first update completed')
        # A, B = self.__update(a, b, c)

        # (typically) updating the otherside side of the bracketing interval
        # (step S2 in [1])
        if c == A:
            # print('update logic 1')
            # Preforming a secant to propose a value of alpha.
            continue_secant, position = self.__position_after_step(A)
            yield False, continue_secant, position
            f_A, dfdx_A = self.__obj_and_grad_unpack(self.__reply_f_and_dfdx)
            C = self.__secant_for_alpha(a, A, dfdx_a, dfdx_A)
            # C = self.__secant_for_alpha (a, A)

        # (step S3 in [1])
        elif c == B:
            # print('update logic 2')
            # Preforming a secant to propose a value of alpha.
            continue_secant, position = self.__position_after_step(B)
            yield False, continue_secant, position
            f_B, dfdx_B = self.__obj_and_grad_unpack(self.__reply_f_and_dfdx)
            C = self.__secant_for_alpha(b, B, dfdx_b, dfdx_B)
            # C = self.__secant_for_alpha (b, B)

        # (step S4 in [1])
        if c == A or c == B:
            # CHECK IF C SATISFY THE WOLFE CONDITIONS i.e pass to tell!!!!!!!
            # IF IT HAS STOP SEARCHING THIS DIRECTION!!!!
            # WOLFE CHECK?
            # checking the proposed point is in range
            if C >= a and C <= b:
                # If the point is out of range there is no need to
                # check wolfe conditions, or preform update if they
                # are not meet
                self._proposed_alpha = C
                continue_secant, position = self.__position_after_step(C)
                yield True, True, position
                # IF WOLFE CONDITIONS AREN'T MEET DO THIS THING BELOW
                # print('second update secant')
                A, B = self.__update(A, B, C)

        # print('end update secant')
        yield False, False, (A, B)

#!/usr/bin/env python3
#
# Tests the basic methods of the Hager-Zhang line search optimiser.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import generators
import unittest
import numpy as np
# from numpy.linalg import norm

import pints
import pints.toy

from shared import CircularBoundaries


debug = False
method = pints.HagerZhang

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestHagerZhang(unittest.TestCase):
    """
    Tests the basic methods of the Hager-Zhang line search optimiser.
    """
    def setUp(self):
        """ Called before every test """
        np.random.seed(1)

    def problem(self):
        """ Returns a test problem, starting point, sigma, and boundaries """
        # true centre is 0,0
        r = pints.toy.ParabolicError(c=[0, 0])
        x = [0.1, 0.1]
        s = 0.1
        b = pints.RectangularBoundaries([-1, -1], [1, 1])
        px = [0.0001, 0.0001]
        return r, x, s, b, px

    def test_unbounded(self):
        # Runs an optimisation without boundaries.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        _B = np.identity(m._n_parameters)
        # print('original score: ', r.evaluateS1(x)[0])
        grad = r.evaluateS1(x)[1]
        px1 = - np.matmul(_B, grad)
        m.set_search_direction(px1)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        # print('found para: ', found_parameters)
        # print('found sol: ', found_solution)
        self.assertTrue(found_solution < 1e-3)

    def test_bounded(self):
        # Runs an optimisation with boundaries.
        r, x, s, b, px = self.problem()

        # Rectangular boundaries
        b = pints.RectangularBoundaries([-1, -1], [1, 1])
        opt = pints.OptimisationController(r, x, boundaries=b, method=method)
        m = opt.optimiser()
        _B = np.identity(m._n_parameters)
        grad = r.evaluateS1(x)[1]
        px1 = - np.matmul(_B, grad)
        m.set_search_direction(px1)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

        # Circular boundaries
        # Start near edge, to increase chance of out-of-bounds occurring.
        b = CircularBoundaries([0, 0], 1)
        x = [0.99, 0]
        opt = pints.OptimisationController(r, x, boundaries=b, method=method)
        m = opt.optimiser()
        _B = np.identity(m._n_parameters)
        grad = r.evaluateS1(x)[1]
        px1 = - np.matmul(_B, grad)
        m.set_search_direction(px1)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_bounded_and_sigma(self):
        # Runs an optimisation without boundaries and sigma.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, s, b, method=method)
        m = opt.optimiser()
        _B = np.identity(m._n_parameters)
        grad = r.evaluateS1(x)[1]
        px1 = - np.matmul(_B, grad)
        m.set_search_direction(px1)
        opt.set_log_to_screen(debug)
        found_parameters, found_solution = opt.run()
        self.assertTrue(found_solution < 1e-3)

    def test_ask_tell(self):
        # Tests ask-and-tell related error handling.
        r, x, s, b, px = self.problem()
        opt = method(x)
        opt.set_search_direction(px)

        # Stop called when not running
        self.assertFalse(opt.running())
        self.assertFalse(opt.stop())

        # Best position and score called before run
        self.assertEqual(list(opt.xbest()), list(x))
        self.assertEqual(opt.fbest(), float('inf'))

        # Tell before ask
        self.assertRaisesRegex(
            Exception, r'ask\(\) not called before tell\(\)', opt.tell, 5)

        # Ask
        opt.ask()

        # Now we should be running
        self.assertTrue(opt.running())

    def test_hyper_parameter_interface(self):
        # Tests the hyper parameter interface for this optimiser.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        self.assertEqual(m.n_hyper_parameters(), 2)
        cs = np.asarray(m.wolfe_line_search_parameters()) * 0.5
        m.set_hyper_parameters(cs)
        self.assertEqual(m.wolfe_line_search_parameters()[0], cs[0])
        self.assertEqual(m.wolfe_line_search_parameters()[1], cs[1])
        self.assertRaisesRegex(ValueError,
                               'Invalid wolfe line search parameters',
                               m.set_hyper_parameters,
                               [5.0, 4.0])

    def test_set_search_direction(self):
        # Tests the set_search_direction method for this optimiser.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        self.assertRaisesRegex(ValueError,
                               'Invalid search direction,',
                               m.set_search_direction,
                               [5.0, 8.0, 66, 434])
        m.set_search_direction(px)
        self.assertEqual(np.all(m._HagerZhang__px), np.all(px))

    def test_name(self):
        # Test the name() method.
        opt = method(np.array([0, 1.01]))
        self.assertIn('Hager-Zhang Line Search', opt.name())

    def test_initialising(self):
        # Tests the initialising method for this optimiser.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        m.set_search_direction(px)

        # checking starting alpha value for line searcher
        # after the first success
        self.assertEqual(m._HagerZhang__initialising(k=0, alpha_k0=2), 2)

        # using first ask tell to set up problem
        proposed = m.ask()
        reply = r.evaluateS1(proposed)
        m.tell([reply])
        # expected_result = ((0.01 * norm(x, ord=np.inf))
        #                     / (norm(grad, ord=np.inf)))
        # print(expected_result)
        self.assertEqual(m._HagerZhang__initialising(k=0, alpha_k0=None),
                         0.005)

        # test for starting point 0.0 and error function is not 0.0
        del(opt)
        del(m)
        r0 = pints.toy.ParabolicError(c=[0.1, 0.1])
        x0 = np.zeros_like(x)
        # temp = r0.evaluateS1(x0)
        # f0 = temp[0]
        # grad0 = temp[1]
        # expected_result = ((0.01 * abs(f0))
        #                    / (pow(norm(grad0, ord=2), 2.0)))
        # print(expected_result)
        # del(temp)
        opt = pints.OptimisationController(r0, x0, method=method)
        m = opt.optimiser()
        m.set_search_direction(px)
        # using first ask tell to set up problem
        proposed = m.ask()
        reply = r0.evaluateS1(proposed)
        m.tell([reply])
        self.assertEqual(m._HagerZhang__initialising(k=0, alpha_k0=None),
                         0.0024999999999999996)

        # testing for starting point at all X are 0.0
        del(opt)
        del(m)
        opt = pints.OptimisationController(r, x0, method=method)
        m = opt.optimiser()
        m.set_search_direction(px)
        # using first ask tell to set up problem
        proposed = m.ask()
        reply = r.evaluateS1(proposed)
        m.tell([reply])
        self.assertEqual(m._HagerZhang__initialising(k=0, alpha_k0=None), 1.0)

        # testing k != 0 i.e a second line search in a
        # gradient based optimiser
        self.assertEqual(m._HagerZhang__initialising(k=1, alpha_k0=3.0), 6.0)

    def test__very_close(self):
        # Tests the very close method of this optimiser.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        self.assertTrue(m._HagerZhang__very_close(2.0, (2.0 + 1E-17)))
        self.assertFalse(m._HagerZhang__very_close(2.0, (2.0 + 1E-14)))

    def test__position_after_step(self):
        # Tests the __position_after_step method of this optimiser.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        m.set_search_direction(px)
        logic, position = m._HagerZhang__position_after_step(alpha=0.5)
        self.assertTrue(logic)
        self.assertEqual(np.all(position), np.all([0.00015, 0.00015]))

    def test__obj_and_grad_unpack(self):
        # Tests the __obj_and_grad_unpack method of this optimiser.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        reply = r.evaluateS1(x)
        func = reply[0]
        grad = reply[1]
        grad = np.matmul(np.transpose(grad), np.asarray(px))
        m.set_search_direction(px)
        self.assertEqual(np.all(
                         m._HagerZhang__obj_and_grad_unpack(x=[reply])),
                         np.all([func, grad]))

    def test__bisect_or_secant(self):
        # Tests the __bisect_or_secant method of this optimiser.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        m.set_search_direction(px)
        # using first ask tell to set up problem
        proposed = m.ask()
        reply = r.evaluateS1(proposed)
        m.tell([reply])
        # grad = r.evaluateS1(x)[1]
        # grad =np.matmul(np.transpose(grad), np.asarray([0.0001, 0.0001]))
        # print('grad ',grad)
        a = 0.5  # d = 0.75
        b = 1.0

        def temp(a=a, b=b, m=m):
            generator = m._HagerZhang__bisect_or_secant(a=a, b=b)
            logic, bracket = next(generator)
            yield logic, bracket
            logic, bracket = next(generator)
            yield logic, bracket

        generator = temp(a, b, m)
        # check yield is correct
        logic, bracket = next(generator)
        self.assertTrue(logic)
        self.assertEqual(np.all(bracket), np.all([0.1000075, 0.1000075]))

        logic, bracket = next(generator)
        self.assertFalse(logic)
        self.assertEqual(np.all(bracket), np.all([0.5, 0.75]))

    def test__initial_bracket(self):
        # Tests the __initial_bracket method of this optimiser.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        m.set_search_direction(px)
        # using first ask tell to set up problem
        proposed = m.ask()
        reply = r.evaluateS1(proposed)
        m.tell([reply])
        # grad = r.evaluateS1(x)[1]
        # grad =np.matmul(np.transpose(grad), np.asarray([0.0001, 0.0001]))
        # print('grad ',grad)
        c = 1.0

        # generator = m._HagerZhang__initial_bracket(c=c)
        # logic , position = next(generator)
        # self.assertTrue(logic)
        # self.assertEqual(np.all(position),np.all([0.1001, 0.1001]))

        def temp(c=c, m=m):
            generator = m._HagerZhang__initial_bracket(c=c)
            logic = True
            while logic is True:
                logic, positon = next(generator)
                reply = r.evaluateS1(positon)
                m._HagerZhang___reply_f_and_dfdx = reply
                if logic is False:
                    yield logic, positon

        generator = temp(c, m)
        logic, bracket = next(generator)

        self.assertFalse(logic)
        self.assertEqual(np.all(bracket), np.all([0.0, 1.0]))

        # # TODO: come up with thest for bisect_or_secant call
        # # Within __inital bracket

    def test__update(self):
        # Tests the __update method of this optimiser.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        m.set_search_direction(px)
        # using first ask tell to set up problem
        proposed = m.ask()
        reply = r.evaluateS1(proposed)
        m.tell([reply])

        # check c is not in (a,b)
        generator = m._HagerZhang__update(a=0.0, b=1.0, c=4.0)
        logic, bracket = next(generator)
        self.assertFalse(logic)
        self.assertEqual(np.all(bracket), np.all([0.0, 1.0]))

        # check object func call
        generator = m._HagerZhang__update(a=0.0, b=1.0, c=0.5)
        logic, position = next(generator)
        self.assertTrue(logic)
        self.assertEqual(np.all(position), np.all([0.10005, 1.00005]))

        # check dfdx_c >= 0.0
        reply = r.evaluateS1(position)
        m._HagerZhang___reply_f_and_dfdx = reply
        logic, bracket = next(generator)
        self.assertFalse(logic)
        self.assertEqual(np.all(bracket), np.all([0.0, 0.5]))

        # checking second logic condition
        x0 = [0.1, -0.2]
        opt = pints.OptimisationController(r, x0, method=method)
        m = opt.optimiser()
        m.set_search_direction(px)
        # using first ask tell to set up problem
        proposed = m.ask()
        reply = r.evaluateS1(proposed)
        m.tell([reply])

        a = 0.0
        b = 1.0
        c = 0.9

        def temp(a=a, b=b, c=c, m=m):
            generator = m._HagerZhang__update(a=a, b=b, c=c)
            logic = True
            while logic is True:
                logic, positon = next(generator)
                reply = r.evaluateS1(positon)
                m._HagerZhang___reply_f_and_dfdx = reply
                if logic is False:
                    yield logic, positon

        generator = temp()
        logic, bracket = next(generator)

        self.assertFalse(logic)
        self.assertEqual(np.all(bracket), np.all([c, b]))

        # checking bisect logic condition
        # TODO: this requires a different problem

    def test__secant_for_alpha(self):
        # Tests the __secant_for_alpha method of this optimiser.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        px = np.asarray([0.0001, 0.0001])
        a = 0.0
        b = 1.0
        reply = r.evaluateS1(x)
        # print(reply)
        grad1 = reply[1]
        grad1 = np.matmul(np.transpose(grad1), px)
        reply = r.evaluateS1(x + px)
        # print(reply)
        grad2 = reply[1]
        grad2 = np.matmul(np.transpose(grad2), px)
        self.assertAlmostEqual(m._HagerZhang__secant_for_alpha(
                               a, b, grad1, grad2), -1000)
        self.assertRaisesRegex(ValueError,
                               'Dividing by zero',
                               m._HagerZhang__secant_for_alpha,
                               a, b, grad1, grad1)

    def test__secant2(self):
        # print('test__secant2')
        # Tests the __secant2 method of this optimiser.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()
        a = 0.0
        b = 1.0
        m.set_search_direction(px)
        # using first ask tell to set up problem
        proposed = m.ask()
        reply = r.evaluateS1(proposed)
        m.tell([reply])

        # checking initial evaluations of gradient
        generator = m._HagerZhang__secant2(a=a, b=b)
        wolfe, grad_calc, position = next(generator)
        self.assertFalse(wolfe)
        self.assertTrue(grad_calc)
        self.assertEqual(np.all(position), np.all(x))
        reply = r.evaluateS1(position)
        m._HagerZhang__reply_f_and_dfdx = [reply]
        wolfe, grad_calc, position = next(generator)
        self.assertFalse(wolfe)
        self.assertTrue(grad_calc)
        self.assertEqual(np.all(position), np.all(x + px))

        # checking first update call
        # this should not meet any logic conditions
        # and return a,b
        reply = r.evaluateS1(position)
        m._HagerZhang__reply_f_and_dfdx = [reply]
        wolfe, grad_calc, bracket = next(generator)
        self.assertFalse(wolfe)
        self.assertFalse(grad_calc)
        self.assertEqual(np.all(bracket), np.all([a, b]))

        # TODO: write more test to check other logic conditions in secant_2

    def test__ask_generator(self):
        # Tests __ask_generator logic via ask.
        r, x, s, b, px = self.problem()
        opt = pints.OptimisationController(r, x, method=method)
        m = opt.optimiser()

        # ask fails if search direction not set
        self.assertRaisesRegex(
            ValueError, 'Warning search direction', m.ask)
        m.set_search_direction(px)

        # checking first return of initial conditions x
        self.assertEqual(np.all(m.ask()), np.all(x))
        reply = r.evaluateS1(x)
        m.tell([reply])

        # checking first return from initial_bracket_gen
        self.assertEqual(np.all(m.ask()), np.all([0.1000005, 0.1000005]))


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

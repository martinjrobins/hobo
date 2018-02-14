#!/usr/bin/env python3
#
# Tests the basic methods of the PSO optimiser.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.io
import pints.toy
import unittest
import numpy as np

debug = False
method = pints.PSO


class TestPSO(unittest.TestCase):
    """
    Tests the basic methods of the PSO optimiser.
    """
    def __init__(self, name):
        super(TestPSO, self).__init__(name)

        # Create toy model
        self.model = pints.toy.LogisticModel()
        self.real_parameters = [0.015, 500]
        self.times = np.linspace(0, 1000, 1000)
        self.values = self.model.simulate(self.real_parameters, self.times)

        # Create an object with links to the model and time series
        self.problem = pints.SingleSeriesProblem(
            self.model, self.times, self.values)

        # Select a score function
        self.score = pints.SumOfSquaresError(self.problem)

        # Select some boundaries
        self.boundaries = pints.Boundaries([0, 400], [0.03, 600])

        # Set an initial position
        self.x0 = 0.014, 499

        # Set an initial guess of the standard deviation in each parameter
        self.sigma0 = [0.001, 1]

        # Minimum score function value to obtain
        self.cutoff = 1e3   # Global method!

        # Maximum tries before it counts as failed
        self.max_tries = 3

    #def test_unbounded(self):

    #    opt = pints.Optimisation(self.score, self.x0, method=method)
    #    opt.set_verbose(debug)
    #    opt.set_max_unchanged_iterations(500)
    #    with np.errstate(over='ignore'):
    #        for i in range(self.max_tries):
    #            found_parameters, found_solution = opt.run()
    #            if found_solution < self.cutoff:
    #                break
    #    self.assertTrue(found_solution < self.cutoff)
    '''
    def test_bounded(self):

        opt = pints.Optimisation(self.score, self.x0,
                                 boundaries=self.boundaries, method=method)
        opt.set_verbose(debug)
        opt.set_max_unchanged_iterations(1000)
        for i in range(self.max_tries):
            found_parameters, found_solution = opt.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)

    def test_bounded_and_sigma(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_verbose(debug)
        opt.set_max_unchanged_iterations(1000)
        for i in range(self.max_tries):
            found_parameters, found_solution = opt.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)

    def test_stopping_max_iter(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_verbose(True)
        opt.set_max_iterations(2)
        opt.set_max_unchanged_iterations(None)
        with pints.io.StdOutCapture() as c:
            opt.run()
            self.assertIn('Halting: Maximum number of iterations', c.text())

    def test_stopping_max_unchanged(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_verbose(True)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(2)
        with pints.io.StdOutCapture() as c:
            opt.run()
            self.assertIn('Halting: No significant change', c.text())

    def test_stopping_threshold(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_verbose(True)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(None)
        opt.set_threshold(1e4 * self.cutoff)
        with pints.io.StdOutCapture() as c:
            opt.run()
            self.assertIn(
                'Halting: Objective function crossed threshold', c.text())

    def test_stopping_no_criterion(self):

        opt = pints.Optimisation(self.score, self.x0, self.sigma0,
                                 self.boundaries, method)
        opt.set_verbose(debug)
        opt.set_max_iterations(None)
        opt.set_max_unchanged_iterations(None)
        self.assertRaises(ValueError, opt.run)

    '''
    def test_logpdf(self):

        #TODO: Replace this with toy problem!
        #TODO: Make one as an error measure, one as a LogPDF
        #TODO: And add to general test of Optimisation, not PSO specifically
        class RosenbrockLogPDF(pints.LogPDF):
            def __call__(self, x):
                a = 1
                b = 100
                f = (a - x[0])**2 + b * (x[1] - x[0]**2)**2
                return float('inf') if f == 0 else -np.log(f)

            def dimension(self):
                return 2

        r = RosenbrockLogPDF()
        x0 = np.array([1.1, 1.1])
        f0 = r(x0)
        b = pints.Boundaries([0.5, 0.5], [1.5, 1.5])
        opt = pints.Optimisation(r, x0, boundaries=b, method=pints.PSO)
        opt.set_max_iterations(100)
        opt.set_max_unchanged_iterations(100)
        opt.set_verbose(debug)

        # PSO isn't very good at this function, unless we give it lots more
        # iterations, check if it's moving in the right direction
        np.random.seed(1)
        x1, f1 = opt.run()
        self.assertTrue(f1 > f0)

    '''
    test_rosenbrock(self):
        """ Test running on the Rosenbrock function """
        class Rosenbrock(pints.ErrorMeasure):

            def __call__(self, x):
                a = 1
                b = 100
                f = (a - x[0])**2 + b * (x[1] - x[0]**2)**2
                return float('inf') if f == 0 else -np.log(f)

            def dimension(self):
                return 2

        r = Rosenbrock()
        x0 = np.array([1.1, 1.1])
        f0 = r(x0)
        b = pints.Boundaries([0.5, 0.5], [1.5, 1.5])
        opt = pints.Optimisation(r, x0, boundaries=b, method=pints.PSO)
        opt.set_max_iterations(100)
        opt.set_max_unchanged_iterations(100)
        opt.set_verbose(debug)
        np.random.seed(1)
        x1, f1 = opt.run()
        self.assertTrue(f1 < f0)
    '''


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

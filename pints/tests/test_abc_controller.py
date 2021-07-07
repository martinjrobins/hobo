#!/usr/bin/env python3
#
# Tests the ABC Controller.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import pints.toy
import unittest
import numpy as np
from shared import StreamCapture

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


debug = False


class TestABCController(unittest.TestCase):
    """
    Tests the ABCController class.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare problem for tests. """

        # Create toy model
        cls.model = pints.toy.StochasticDegradationModel()
        cls.real_parameters = [0.1]
        cls.times = np.linspace(0, 10, 10)
        cls.values = cls.model.simulate(cls.real_parameters, cls.times)

        # Create an object (problem) with links to the model and time series
        cls.problem = pints.SingleOutputProblem(
            cls.model, cls.times, cls.values)

        # Create a uniform prior over both the parameters
        cls.log_prior = pints.UniformLogPrior(
            [0.0],
            [0.3]
        )

        # Set error measure
        cls.error_measure = pints.RootMeanSquaredError(cls.problem)

    def test_nparameters_error(self):
        """ Test that error is thrown when parameters from log prior and error
        measure do not match"""
        log_prior = pints.UniformLogPrior(
            [0.0, 0, 0],
            [0.2, 100, 1])

        self.assertRaises(ValueError, pints.ABCController, self.error_measure,
                          log_prior)

    def test_stopping(self):
        """ Test different stopping criteria. """

        abc = pints.ABCController(self.error_measure, self.log_prior)

        # Test setting max iterations
        maxi = abc.max_iterations() + 2
        self.assertNotEqual(maxi, abc.max_iterations())
        abc.set_max_iterations(maxi)
        self.assertEqual(maxi, abc.max_iterations())
        self.assertRaisesRegex(
            ValueError, 'negative', abc.set_max_iterations, -1)

        # Test without stopping criteria
        abc.set_max_iterations(None)
        self.assertIsNone(abc.max_iterations())
        self.assertRaisesRegex(
            ValueError, 'At least one stopping criterion', abc.run)

    def test_parallel(self):
        """ Test running ABC with parallisation. """

        abc = pints.ABCController(
            self.error_measure, self.log_prior, method=pints.RejectionABC)

        # Test with auto-detected number of worker processes
        self.assertFalse(abc.parallel())
        abc.set_parallel(True)
        self.assertTrue(abc.parallel())
        self.assertEqual(abc.parallel(), pints.ParallelEvaluator.cpu_count())

        # Test with fixed number of worker processes
        abc.set_parallel(2)
        self.assertEqual(abc.parallel(), 2)

    def test_logging(self):
        # tests logging to screen
        # No output
        with StreamCapture() as capture:
            abc = pints.ABCController(
                self.error_measure, self.log_prior, method=pints.RejectionABC)
            abc.set_max_iterations(10)
            abc.set_log_to_screen(False)
            abc.set_log_to_file(False)
            abc.run()
        self.assertEqual(capture.text(), '')

        # With output to screen
        np.random.seed(1)
        with StreamCapture() as capture:
            pints.ABCController(
                self.error_measure, self.log_prior, method=pints.RejectionABC)
            abc.set_max_iterations(10)
            abc.set_log_to_screen(True)
            abc.set_log_to_file(False)
            abc.run()
        lines = capture.text().splitlines()
        self.assertTrue(len(lines) > 0)

        # With output to screen
        np.random.seed(1)
        with StreamCapture() as capture:
            pints.ABCController(
                self.error_measure, self.log_prior, method=pints.RejectionABC)
            abc.set_max_iterations(10)
            abc.set_log_to_screen(False)
            abc.set_log_to_file(True)
            abc.run()
        lines = capture.text().splitlines()
        self.assertTrue(len(lines) == 0)

        # Invalid log interval
        self.assertRaises(ValueError, abc.set_log_interval, 0)

        abc = pints.ABCController(
            self.error_measure, self.log_prior, method=pints.RejectionABC)
        abc.set_log_to_file("temp_file")
        self.assertEqual(abc.log_filename(), "temp_file")

        # tests logging to screen with parallel
        with StreamCapture() as capture:
            abc = pints.ABCController(
                self.error_measure, self.log_prior, method=pints.RejectionABC)
            abc.set_parallel(2)
            abc.set_max_iterations(10)
            abc.set_log_to_screen(False)
            abc.set_log_to_file(False)
            abc.run()
        self.assertEqual(capture.text(), '')

    def test_controller_extra(self):
        # tests various controller aspects
        self.assertRaises(ValueError, pints.ABCController, self.error_measure,
                          self.error_measure)
        self.assertRaisesRegex(
            ValueError, 'Given method must extend pints.ABCSampler',
            pints.ABCController, self.error_measure,
            self.log_prior, pints.MCMCSampler)
        self.assertRaises(ValueError, pints.ABCController, self.error_measure,
                          pints.MCMCSampler)
        self.assertRaises(ValueError, pints.ABCController, self.error_measure,
                          self.log_prior, 0.0)

        # test setters
        abc = pints.ABCController(
            self.error_measure, self.log_prior, method=pints.RejectionABC)
        abc.set_n_target(230)
        self.assertEqual(abc.n_target(), 230)

        sampler = abc.sampler()
        pt = sampler.ask(1)
        self.assertEqual(len(pt), 1)

        abc.set_parallel(False)
        self.assertEqual(abc.parallel(), 0)

        with StreamCapture() as capture:
            abc = pints.ABCController(
                self.error_measure, self.log_prior, method=pints.RejectionABC)
            abc.set_parallel(4)
            abc.sampler().set_threshold(100)
            abc.set_n_target(1)
            abc.run()
        lines = capture.text().splitlines()
        self.assertTrue(len(lines) > 0)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

#!/usr/bin/env python
#
# Tests the basic methods of the adaptive covariance MCMC routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy as toy
import unittest
import numpy as np

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestRejectionABC(unittest.TestCase):
    """
    Tests the basic methods of the ABC Rejection routine.
    """
# Set up toy model, parameter values, problem, error measure
    @classmethod
    def setUpClass(cls):
        """ Set up problem for tests. """

        # Create toy model
        cls.model = toy.StochasticDegradationModel()
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

    def test_method(self):

        # Create abc rejection scheme
        abc = pints.RejectionABC(self.log_prior)

        # Configure
        n_draws = 1
        niter = 20

        # Perform short run using ask and tell framework
        samples = []
        while len(samples) < niter:
            x = abc.ask(n_draws)[0]
            fx = self.error_measure(x)
            sample = abc.tell(fx)
            while sample is None:
                x = abc.ask(n_draws)[0]
                fx = self.error_measure(x)
                sample = abc.tell(fx)
            samples.append(sample)

        samples = np.array(samples)
        self.assertEqual(samples.shape[0], niter)

    def test_errors(self):
        # test errors in abc rejection
        abc = pints.RejectionABC(self.log_prior)
        abc.ask(1)
        # test two asks raises error
        self.assertRaises(RuntimeError, abc.ask, 1)

        # test tell with large value
        self.assertEqual(None, abc.tell(100))
        # test error raised if tell called before ask
        self.assertRaises(RuntimeError, abc.tell, 2.5)

    def test_setters_and_getters(self):
        # test setting and getting
        abc = pints.RejectionABC(self.log_prior)
        self.assertEqual('Rejection ABC', abc.name())
        self.assertEqual(abc.threshold(), 1)
        abc.set_threshold(2)
        self.assertEqual(abc.threshold(), 2)
        self.assertRaises(ValueError, abc.set_threshold, -3)


if __name__ == '__main__':
    unittest.main()

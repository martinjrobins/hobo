#!/usr/bin/env python
#
# Tests the basic methods of the Dram ACMC routine.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy as toy
import unittest
import numpy as np

from shared import StreamCapture

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestDramACMC(unittest.TestCase):
    """
    Tests the basic methods of the DRAM ACMC routine.
    """

    @classmethod
    def setUpClass(cls):
        """ Set up problem for tests. """

        # Create toy model
        cls.model = toy.LogisticModel()
        cls.real_parameters = [0.015, 500]
        cls.times = np.linspace(0, 1000, 1000)
        cls.values = cls.model.simulate(cls.real_parameters, cls.times)

        # Add noise
        cls.noise = 10
        cls.values += np.random.normal(0, cls.noise, cls.values.shape)
        cls.real_parameters.append(cls.noise)
        cls.real_parameters = np.array(cls.real_parameters)

        # Create an object with links to the model and time series
        cls.problem = pints.SingleOutputProblem(
            cls.model, cls.times, cls.values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        cls.log_prior = pints.UniformLogPrior(
            [0.01, 400, cls.noise * 0.1],
            [0.02, 600, cls.noise * 100]
        )

        # Create a log likelihood
        cls.log_likelihood = pints.GaussianLogLikelihood(cls.problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        cls.log_posterior = pints.LogPosterior(
            cls.log_likelihood, cls.log_prior)

    def test_method(self):

        # Create mcmc
        x0 = self.real_parameters * 1.1
        mcmc = pints.DramACMC(x0)

        # Configure
        mcmc.set_target_acceptance_rate(0.3)
        mcmc.set_initial_phase(True)

        # Perform short run
        rate = []
        chain = []
        for i in range(100):
            x = mcmc.ask()
            fx = self.log_posterior(x)
            reply = mcmc.tell(fx)
            while reply is None:
                x = mcmc.ask()
                fx = self.log_posterior(x)
                reply = mcmc.tell(fx)
            y, fy, ac = reply
            if i == 20:
                mcmc.set_initial_phase(False)
            if i >= 50:
                chain.append(y)
            rate.append(mcmc.acceptance_rate())
            self.assertTrue(isinstance(ac, bool))
            if ac:
                self.assertTrue(np.all(x == y))
                self.assertEqual(fx, fy)

        chain = np.array(chain)
        rate = np.array(rate)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(x0))
        self.assertEqual(rate.shape[0], 100)

        # Test with more kernels
        x0 = self.real_parameters * 1.1
        mcmc = pints.DramACMC(x0)

        # Perform short run
        rate = []
        chain = []
        for i in range(100):
            x = mcmc.ask()
            fx = self.log_posterior(x)
            reply = mcmc.tell(fx)
            while reply is None:
                x = mcmc.ask()
                fx = self.log_posterior(x)
                reply = mcmc.tell(fx)
            y, fy, ac = reply
            if i == 20:
                mcmc.set_initial_phase(False)
            if i >= 50:
                chain.append(y)
            rate.append(mcmc.acceptance_rate())
            self.assertTrue(isinstance(ac, bool))
            if ac:
                self.assertTrue(np.all(x == y))
                self.assertEqual(fx, fy)

        chain = np.array(chain)
        rate = np.array(rate)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(x0))
        self.assertEqual(rate.shape[0], 100)

    def test_options(self):

        # Test setting acceptance rate
        x0 = self.real_parameters
        mcmc = pints.DramACMC(x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0.0)
        x0 = self.real_parameters
        mcmc = pints.DramACMC(x0)
        mcmc.ask()
        self.assertRaises(ValueError, mcmc.tell, -float('inf'))

        self.assertNotEqual(mcmc.target_acceptance_rate(), 0.5)
        mcmc.set_target_acceptance_rate(0.5)
        self.assertEqual(mcmc.target_acceptance_rate(), 0.5)
        mcmc.set_target_acceptance_rate(1)
        self.assertRaises(ValueError, mcmc.set_target_acceptance_rate, 0)
        self.assertRaises(ValueError, mcmc.set_target_acceptance_rate, -1e-6)
        self.assertRaises(ValueError, mcmc.set_target_acceptance_rate, 1.00001)

        # test hyperparameter setters and getters
        self.assertEqual(mcmc.n_hyper_parameters(), 2)
        self.assertRaises(ValueError, mcmc.set_hyper_parameters, [-0.1,
                                                                  [1, 3]])
        self.assertRaises(ValueError, mcmc.set_hyper_parameters, [0.5,
                                                                  [0, 3]])
        self.assertRaises(ValueError, mcmc.set_hyper_parameters, [0.5,
                                                                  [1, -1]])
        self.assertRaises(ValueError, mcmc.set_hyper_parameters, [0.5,
                                                                  [1]])
        mcmc.set_hyper_parameters([0.1, [4, 3]])
        self.assertEqual(mcmc.eta(), 0.1)
        mcmc.ask()
        scale1 = [2, 3]
        mcmc.set_sigma_scale(scale1)
        scale = mcmc.sigma_scale()
        self.assertTrue(np.array_equal(scale, scale1))

        self.assertEqual(mcmc.name(), (
            'Delayed Rejection Adaptive Metropolis (Dram) MCMC'))

    def test_logging(self):

        # Test logging includes name.
        x = [self.real_parameters] * 3
        mcmc = pints.MCMCController(
            self.log_posterior, 3, x, method=pints.DramACMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()
        self.assertIn('Delayed Rejection Adaptive Metropolis (Dram) MCMC',
                      text)


if __name__ == '__main__':
    unittest.main()

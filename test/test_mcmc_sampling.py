#!/usr/bin/env python2
#
# Tests the basic methods of the adaptive covariance MCMC routine.
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

LOG_SCREEN = (
    'Using Adaptive covariance MCMC\n'
    'Generating 3 chains.\n'
    'Running in sequential mode.\n'
    'Iter. Eval. Accept.   Accept.   Accept.   Time m:s\n'
    '0     3      0         0         0          0:00.0\n'
    '1     6      0         0         0.5        0:00.0\n'
    '2     9      0         0         0.333      0:00.0\n'
    '3     12     0         0         0.5        0:00.0\n'
    '10    30     0.1       0         0.2        0:00.0\n'
    'Halting: Maximum number of iterations (10) reached.\n'
)

LOG_FILE = (
    'Iter. Eval. Accept.   Accept.   Accept.   Time m:s\n'
    '0     3      0         0         0          0:00.0\n'
    '1     6      0         0         0.5        0:00.0\n'
    '2     9      0         0         0.333      0:00.0\n'
    '3     12     0         0         0.5        0:00.0\n'
    '10    30     0.1       0         0.2        0:00.0\n'
)


class TestMCMCSampling(unittest.TestCase):
    """
    Tests the MCMCSampling class.
    """
    def __init__(self, name):
        super(TestMCMCSampling, self).__init__(name)

        # Create toy model
        model = pints.toy.LogisticModel()
        self.real_parameters = [0.015, 500]
        times = np.linspace(0, 1000, 1000)
        values = model.simulate(self.real_parameters, times)

        # Add noise
        np.random.seed(1)
        self.noise = 10
        values += np.random.normal(0, self.noise, values.shape)
        self.real_parameters.append(self.noise)

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        self.log_prior = pints.UniformLogPrior(
            [0.01, 400, self.noise * 0.1],
            [0.02, 600, self.noise * 100]
        )

        # Create a log-likelihood
        self.log_likelihood = pints.UnknownNoiseLogLikelihood(problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        self.log_posterior = pints.LogPosterior(
            self.log_likelihood, self.log_prior)

    def test_single(self):
        """ Test with a SingleChainMCMC method. """

        # One chain
        nchains = 1

        # Test simple run
        x0 = np.array(self.real_parameters) * 1.1
        xs = [x0]
        nparameters = len(x0)
        niterations = 10
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Check function argument
        pints.MCMCSampling(self.log_posterior, nchains, xs)
        pints.MCMCSampling(self.log_prior, nchains, xs)
        pints.MCMCSampling(self.log_likelihood, nchains, xs)

        def f(x):
            return x
        self.assertRaises(ValueError, pints.MCMCSampling, f, nchains, xs)

        # Test x0 and chain argument
        self.assertRaises(
            ValueError, pints.MCMCSampling, self.log_posterior, 0, [])
        self.assertRaises(
            ValueError, pints.MCMCSampling, self.log_posterior, 1, x0)
        self.assertRaises(
            ValueError, pints.MCMCSampling, self.log_posterior, 2, xs)

        # Check different sigma0 initialisations
        pints.MCMCSampling(self.log_posterior, nchains, xs)
        sigma0 = [0.005, 100, 0.5 * self.noise]
        pints.MCMCSampling(self.log_posterior, nchains, xs, sigma0)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise])
        pints.MCMCSampling(self.log_posterior, nchains, xs, sigma0)
        sigma0 = [0.005, 100, 0.5 * self.noise, 10]
        self.assertRaises(
            ValueError,
            pints.MCMCSampling, self.log_posterior, nchains, xs, sigma0)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise, 10])
        self.assertRaises(
            ValueError,
            pints.MCMCSampling, self.log_posterior, nchains, xs, sigma0)

        # Test multi-chain with single-chain mcmc

        # 2 chains
        x0 = np.array(self.real_parameters) * 1.1
        x1 = np.array(self.real_parameters) * 1.15
        xs = [x0, x1]
        nchains = len(xs)
        nparameters = len(x0)
        niterations = 10
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # 10 chains
        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)
        nparameters = len(xs[0])
        niterations = 20
        mcmc = pints.MCMCSampling(
            self.log_posterior, nchains, xs,
            method=pints.AdaptiveCovarianceMCMC)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

    def test_multi(self):

        # 10 chains
        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)
        nparameters = len(xs[0])
        niterations = 20

        # Test with multi-chain method
        mcmc = pints.MCMCSampling(
            self.log_posterior, nchains, xs,
            method=pints.DifferentialEvolutionMCMC)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Test without stopping criteria
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_max_iterations(None)
        self.assertRaises(ValueError, mcmc.run)

    def test_logging(self):

        np.random.seed(1)
        xs = []
        for i in range(3):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)

        # No output
        with pints.io.StreamCapture() as capture:
            mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
            mcmc.set_max_iterations(10)
            mcmc.set_log_to_screen(False)
            mcmc.run()
        self.assertEqual(capture.text(), '')

        # With output to screen
        np.random.seed(1)
        with pints.io.StreamCapture() as capture:
            mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
            mcmc.set_max_iterations(10)
            mcmc.set_log_to_screen(True)
            mcmc.run()
        self.assertEqual(capture.text(), LOG_SCREEN)

        # With output to file
        np.random.seed(1)
        with pints.io.StreamCapture() as capture:
            with pints.io.TemporaryDirectory() as d:
                filename = d.path('test.txt')
                mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
                mcmc.set_max_iterations(10)
                mcmc.set_log_to_screen(False)
                mcmc.set_log_to_file(filename)
                mcmc.run()
                with open(filename, 'r') as f:
                    self.assertEqual(f.read(), LOG_FILE)
                    pass
            self.assertEqual(capture.text(), '')

        # Invalid log rate
        self.assertRaises(ValueError, mcmc.set_log_rate, 0)

    def test_adaptation(self):

        # 2 chains
        x0 = np.array(self.real_parameters) * 1.1
        x1 = np.array(self.real_parameters) * 1.15
        xs = [x0, x1]
        nchains = len(xs)

        # Delayed adaptation
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_adaptation_free_iterations(10)
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.adaptation())
        mcmc.set_max_iterations(9)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.adaptation())

        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_adaptation_free_iterations(10)
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.adaptation())
        mcmc.set_max_iterations(19)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.adaptation())

        # No delay
        mcmc = pints.MCMCSampling(self.log_posterior, nchains, xs)
        mcmc.set_adaptation_free_iterations(0)
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.adaptation())
        mcmc.set_adaptation_free_iterations(0)
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.adaptation())


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

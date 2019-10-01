#!/usr/bin/env python3
#
# Tests the MCMC Controller.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import os
import pints
import pints.io
import pints.toy
import unittest
import numpy as np

from shared import StreamCapture, TemporaryDirectory

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


debug = False

LOG_SCREEN = [
    'Using Haario-Bardenet adaptive covariance MCMC',
    'Generating 3 chains.',
    'Running in sequential mode.',
    'Iter. Eval. Accept.   Accept.   Accept.   Time m:s',
    '0     3      0         0         0          0:00.0',
    '1     6      0         0         0          0:00.0',
    '2     9      0         0         0          0:00.0',
    '3     12     0         0         0          0:00.0',
    'Initial phase completed.',
    '10    30     0         0         0          0:00.0',
    'Halting: Maximum number of iterations (10) reached.',
]

LOG_FILE = [
    'Iter. Eval. Accept.   Accept.   Accept.   Time m:s',
    '0     3      0         0         0          0:00.0',
    '1     6      0         0         0          0:00.0',
    '2     9      0         0         0          0:00.0',
    '3     12     0         0         0          0:00.0',
    '10    30     0         0         0          0:00.0',
]


class SingleListSampler(pints.SingleChainMCMC):
    """
    Returns predetermined samples from a list.
    First sample can't be None.
    """

    def set_chain(self, chain):
        self._chain = list(chain)
        self._i = 0
        self._n = len(self._chain)

    def ask(self):
        x = self._chain[self._i] if self._i < self._n else None
        if x is None:
            x = self._chain[0]
        return x

    def name(self):
        return 'SingleListSampler'

    def tell(self, fx):
        x = self._chain[self._i] if self._i < self._n else None
        self._i += 1
        return x


class MultiListSampler(pints.MultiChainMCMC):
    """
    Returns predetermined samples from a list of chains.
    Adding a ``None`` in the first list at any point will cause ``None`` to be
    returned (for all chains) at that iteration.
    First sample can't be None.
    """

    def set_chains(self, chains):
        self._chains = [list(x) for x in chains]
        self._i = 0
        self._n = len(self._chains[0])

    def ask(self):
        x = [chain[self._i] for chain in self._chains]
        if x[0] is None:
            x = [chain[0] for chain in self._chains]
        return x

    def name(self):
        return 'MultiListSampler'

    def tell(self, fx):
        x = None
        if self._i < self._n:
            x = [chain[self._i] for chain in self._chains]
            if x[0] is None:
                x = None
            self._i += 1
        return x


class SumDistribution(pints.LogPDF):
    """ Distribution where p(x) = 1 + sum(x) """

    def __init__(self, n_parameters):
        self._n_parameters = n_parameters

    def __call__(self, x):
        return 1 + np.sum(x)

    def n_parameters(self):
        return self._n_parameters


class TestMCMCController(unittest.TestCase):
    """
    Tests the MCMCController class.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare problem for tests. """

        # Create toy model
        model = pints.toy.LogisticModel()
        cls.real_parameters = [0.015, 500]
        times = np.linspace(0, 1000, 1000)
        values = model.simulate(cls.real_parameters, times)

        # Add noise
        np.random.seed(1)
        cls.noise = 10
        values += np.random.normal(0, cls.noise, values.shape)
        cls.real_parameters.append(cls.noise)

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model, times, values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        cls.log_prior = pints.UniformLogPrior(
            [0.01, 400, cls.noise * 0.1],
            [0.02, 600, cls.noise * 100]
        )

        # Create a log-likelihood
        cls.log_likelihood = pints.GaussianLogLikelihood(problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        cls.log_posterior = pints.LogPosterior(
            cls.log_likelihood, cls.log_prior)

    def test_single(self):
        # Test with a SingleChainMCMC method.

        # One chain
        nchains = 1

        # Test simple run
        x0 = np.array(self.real_parameters) * 1.1
        xs = [x0]
        nparameters = len(x0)
        niterations = 10
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(False)
        self.assertEqual(len(mcmc.samplers()), nchains)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Check constructor arguments
        pints.MCMCController(self.log_posterior, nchains, xs)
        pints.MCMCController(self.log_prior, nchains, xs)
        pints.MCMCController(self.log_likelihood, nchains, xs)

        # Check sampler() method
        self.assertRaises(RuntimeError, mcmc.sampler)

        def f(x):
            return x
        self.assertRaisesRegex(
            ValueError, 'extend pints.LogPDF', pints.MCMCController, f,
            nchains, xs)

        # Test x0 and chain argument
        self.assertRaisesRegex(
            ValueError, 'chains must be at least 1',
            pints.MCMCController, self.log_posterior, 0, [])
        self.assertRaisesRegex(
            ValueError, 'positions must be equal to number of chains',
            pints.MCMCController, self.log_posterior, 1, x0)
        self.assertRaisesRegex(
            ValueError, 'positions must be equal to number of chains',
            pints.MCMCController, self.log_posterior, 2, xs)
        self.assertRaisesRegex(
            ValueError, 'same dimension',
            pints.MCMCController, self.log_posterior, 1, [x0[:-1]])
        self.assertRaisesRegex(
            ValueError, 'extend pints.MCMCSampler',
            pints.MCMCController, self.log_posterior, 1, xs, method=12)
        self.assertRaisesRegex(
            ValueError, 'Given method cannot be abstract base class',
            pints.MCMCController, self.log_posterior, 1, xs,
            method=pints.AdaptiveCovarianceMC)
        self.assertRaisesRegex(
            ValueError, 'Given method cannot be abstract base class',
            pints.MCMCController, self.log_posterior, 1, xs,
            method=pints.GlobalAdaptiveCovarianceMC)

        # Check different sigma0 initialisations
        pints.MCMCController(self.log_posterior, nchains, xs)
        sigma0 = [0.005, 100, 0.5 * self.noise]
        pints.MCMCController(self.log_posterior, nchains, xs, sigma0)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise])
        pints.MCMCController(self.log_posterior, nchains, xs, sigma0)
        sigma0 = [0.005, 100, 0.5 * self.noise, 10]
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, nchains, xs, sigma0)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise, 10])
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, nchains, xs, sigma0)

        # Test multi-chain with single-chain mcmc

        # 2 chains
        x0 = np.array(self.real_parameters) * 1.1
        x1 = np.array(self.real_parameters) * 1.15
        xs = [x0, x1]
        nchains = len(xs)
        nparameters = len(x0)
        niterations = 10
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
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
        mcmc = pints.MCMCController(
            self.log_posterior, nchains, xs,
            method=pints.HaarioBardenetACMC)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

    def test_multi(self):
        # Test with a multi-chain method

        # Set up problem for 10 chains
        x0 = np.array(self.real_parameters)
        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(x0 * f)
        nchains = len(xs)
        nparameters = len(xs[0])
        niterations = 20

        # Test with multi-chain method
        meth = pints.DifferentialEvolutionMCMC
        mcmc = pints.MCMCController(
            self.log_posterior, nchains, xs, method=meth)
        self.assertEqual(len(mcmc.samplers()), 1)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Check constructor arguments
        pints.MCMCController(self.log_posterior, nchains, xs, method=meth)
        pints.MCMCController(self.log_prior, nchains, xs, method=meth)
        pints.MCMCController(self.log_likelihood, nchains, xs, method=meth)

        # Test x0 and chain argument
        self.assertRaisesRegex(
            ValueError, 'chains must be at least 1', meth, 0, [])
        self.assertRaisesRegex(
            ValueError, 'at least 3',
            meth, 1, [x0])
        self.assertRaisesRegex(
            ValueError, 'positions must be equal to number of chains',
            meth, 5, xs)
        self.assertRaisesRegex(
            ValueError, 'same dimension',
            meth, 3, [x0, x0, x0[:-1]])

        # Check sampler() method
        self.assertIsInstance(mcmc.sampler(), pints.MultiChainMCMC)

        # Check different sigma0 initialisations work
        pints.MCMCController(self.log_posterior, nchains, xs, method=meth)
        sigma0 = [0.005, 100, 0.5 * self.noise]
        pints.MCMCController(
            self.log_posterior, nchains, xs, sigma0, method=meth)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise])
        pints.MCMCController(
            self.log_posterior, nchains, xs, sigma0, method=meth)
        sigma0 = [0.005, 100, 0.5 * self.noise, 10]
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, nchains, xs, sigma0,
            method=meth)
        sigma0 = np.diag([0.005, 100, 0.5 * self.noise, 10])
        self.assertRaises(
            ValueError,
            pints.MCMCController, self.log_posterior, nchains, xs, sigma0,
            method=meth)

    def test_stopping(self):
        # Test different stopping criteria.

        nchains = 1
        xs = [np.array(self.real_parameters) * 1.1]
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)

        # Test setting max iterations
        maxi = mcmc.max_iterations() + 2
        self.assertNotEqual(maxi, mcmc.max_iterations())
        mcmc.set_max_iterations(maxi)
        self.assertEqual(maxi, mcmc.max_iterations())
        self.assertRaisesRegex(
            ValueError, 'negative', mcmc.set_max_iterations, -1)

        # Test without stopping criteria
        mcmc.set_max_iterations(None)
        self.assertIsNone(mcmc.max_iterations())
        self.assertRaisesRegex(
            ValueError, 'At least one stopping criterion', mcmc.run)

    def test_parallel(self):
        # Test running MCMC with parallisation.

        xs = []
        for i in range(10):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)
        nparameters = len(xs[0])
        niterations = 20
        mcmc = pints.MCMCController(
            self.log_posterior, nchains, xs,
            method=pints.HaarioBardenetACMC)
        mcmc.set_max_iterations(niterations)
        mcmc.set_log_to_screen(debug)

        # Test with auto-detected number of worker processes
        self.assertFalse(mcmc.parallel())
        mcmc.set_parallel(True)
        self.assertTrue(mcmc.parallel())
        chains = mcmc.run()
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

        # Test with fixed number of worker processes
        mcmc.set_parallel(2)
        mcmc.set_log_to_screen(True)
        self.assertIs(mcmc._parallel, True)
        self.assertEqual(mcmc._n_workers, 2)
        with StreamCapture() as c:
            chains = mcmc.run()
        self.assertIn('with 2 worker', c.text())
        self.assertEqual(chains.shape[0], nchains)
        self.assertEqual(chains.shape[1], niterations)
        self.assertEqual(chains.shape[2], nparameters)

    def test_logging(self):
        # Test logging functions

        np.random.seed(1)
        xs = []
        for i in range(3):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)

        # No output
        with StreamCapture() as capture:
            mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
            mcmc.set_initial_phase_iterations(5)
            mcmc.set_max_iterations(10)
            mcmc.set_log_to_screen(False)
            mcmc.set_log_to_file(False)
            mcmc.run()
        self.assertEqual(capture.text(), '')

        # With output to screen
        np.random.seed(1)
        with StreamCapture() as capture:
            mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
            mcmc.set_initial_phase_iterations(5)
            mcmc.set_max_iterations(10)
            mcmc.set_log_to_screen(True)
            mcmc.set_log_to_file(False)
            mcmc.run()
        lines = capture.text().splitlines()
        for i, line in enumerate(lines):
            self.assertLess(i, len(LOG_SCREEN))
            # Chop off time bit before comparison
            if LOG_SCREEN[i][-6:] == '0:00.0':
                self.assertEqual(line[:-6], LOG_SCREEN[i][:-6])
            else:
                self.assertEqual(line, LOG_SCREEN[i])
        self.assertEqual(len(lines), len(LOG_SCREEN))

        # With output to file
        np.random.seed(1)
        with StreamCapture() as capture:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
                mcmc.set_initial_phase_iterations(5)
                mcmc.set_max_iterations(10)
                mcmc.set_log_to_screen(False)
                mcmc.set_log_to_file(filename)
                mcmc.run()
                with open(filename, 'r') as f:
                    lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    self.assertLess(i, len(LOG_FILE))
                    # Chop off time bit before comparison
                    if LOG_FILE[i][-6:] == '0:00.0':
                        self.assertEqual(line[:-6], LOG_FILE[i][:-6])
                    else:
                        self.assertEqual(line, LOG_FILE[i])
                    self.assertEqual(line[:-6], LOG_FILE[i][:-6])
                self.assertEqual(len(lines), len(LOG_FILE))
            self.assertEqual(capture.text(), '')

        # Invalid log interval
        self.assertRaises(ValueError, mcmc.set_log_interval, 0)

    def test_initial_phase(self):
        # Test if the initial phase functions work

        # 2 chains
        x0 = np.array(self.real_parameters) * 1.1
        x1 = np.array(self.real_parameters) * 1.15
        xs = [x0, x1]
        nchains = len(xs)

        # Initial phase
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
        self.assertTrue(mcmc.method_needs_initial_phase())
        self.assertNotEqual(mcmc.initial_phase_iterations(), 10)
        mcmc.set_initial_phase_iterations(10)
        self.assertEqual(mcmc.initial_phase_iterations(), 10)
        self.assertRaisesRegex(
            ValueError, 'negative', mcmc.set_initial_phase_iterations, -1)
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.in_initial_phase())
        mcmc.set_max_iterations(9)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.in_initial_phase())

        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(10)
        for sampler in mcmc._samplers:
            self.assertTrue(sampler.in_initial_phase())
        mcmc.set_max_iterations(11)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.in_initial_phase())

        # No initial phase
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(0)
        mcmc.set_max_iterations(1)
        mcmc.set_log_to_screen(False)
        mcmc.run()
        for sampler in mcmc._samplers:
            self.assertFalse(sampler.in_initial_phase())

    def test_live_chain_and_eval_logging(self):
        # Test logging to disk and screen

        np.random.seed(1)
        xs = []
        for i in range(3):
            f = 0.9 + 0.2 * np.random.rand()
            xs.append(np.array(self.real_parameters) * f)
        nchains = len(xs)

        # Test writing chains - not evals to disk (using LogPosterior)
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(cpath)
                mcmc.set_log_pdf_filename(None)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are created afterwards
                chains1 = mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertTrue(os.path.exists(p0))
                self.assertTrue(os.path.exists(p1))
                self.assertTrue(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files contain the correct chains
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, nchains))
                self.assertTrue(np.all(chains1 == chains2))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertNotIn('Writing evaluations to', text)
            self.assertNotIn('evals_0.csv', text)

        # Test writing evals - not chains to disk (using LogPosterior)
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(None)
                mcmc.set_log_pdf_filename(epath)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are created afterwards
                chains1 = mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertTrue(os.path.exists(p3))
                self.assertTrue(os.path.exists(p4))
                self.assertTrue(os.path.exists(p5))

                # Test files contain the correct values
                import pints.io as io
                evals2 = np.array(io.load_samples(epath, nchains))
                evals1 = []
                for chain in chains1:
                    logpdfs = np.array([self.log_posterior(x) for x in chain])
                    logpriors = np.array([self.log_prior(x) for x in chain])
                    loglikelihoods = logpdfs - logpriors
                    evals = np.array([logpdfs, loglikelihoods, logpriors]).T
                    evals1.append(evals)
                evals1 = np.array(evals1)
                self.assertTrue(np.all(evals1 == evals2))

            text = c.text()
            self.assertNotIn('Writing chains to', text)
            self.assertNotIn('chain_0.csv', text)
            self.assertIn('Writing evaluations to', text)
            self.assertIn('evals_0.csv', text)

        # Test writing chains and evals to disk
        # With a LogPosterior - Single chain method
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(cpath)
                mcmc.set_log_pdf_filename(epath)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are created afterwards
                chains1 = mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertTrue(os.path.exists(p0))
                self.assertTrue(os.path.exists(p1))
                self.assertTrue(os.path.exists(p2))
                self.assertTrue(os.path.exists(p3))
                self.assertTrue(os.path.exists(p4))
                self.assertTrue(os.path.exists(p5))

                # Test chain files contain the correct values
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, nchains))
                self.assertTrue(np.all(chains1 == chains2))

                # Test eval files contain the correct values
                evals2 = np.array(io.load_samples(epath, nchains))
                evals1 = []
                for chain in chains1:
                    logpdfs = np.array([self.log_posterior(x) for x in chain])
                    logpriors = np.array([self.log_prior(x) for x in chain])
                    loglikelihoods = logpdfs - logpriors
                    evals = np.array([logpdfs, loglikelihoods, logpriors]).T
                    evals1.append(evals)
                evals1 = np.array(evals1)
                self.assertTrue(np.all(evals1 == evals2))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertIn('Writing evaluations to', text)
            self.assertIn('evals_0.csv', text)

        # Test writing chains and evals to disk
        # With a LogPosterior - multi-chain method
        mcmc = pints.MCMCController(
            self.log_posterior, nchains, xs,
            method=pints.DifferentialEvolutionMCMC)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(cpath)
                mcmc.set_log_pdf_filename(epath)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are created afterwards
                chains1 = mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertTrue(os.path.exists(p0))
                self.assertTrue(os.path.exists(p1))
                self.assertTrue(os.path.exists(p2))
                self.assertTrue(os.path.exists(p3))
                self.assertTrue(os.path.exists(p4))
                self.assertTrue(os.path.exists(p5))

                # Test chain files contain the correct values
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, nchains))
                self.assertTrue(np.all(chains1 == chains2))

                # Test eval files contain the correct values
                evals2 = np.array(io.load_samples(epath, nchains))
                evals1 = []
                for chain in chains1:
                    logpdfs = np.array([self.log_posterior(x) for x in chain])
                    logpriors = np.array([self.log_prior(x) for x in chain])
                    loglikelihoods = logpdfs - logpriors
                    evals = np.array([logpdfs, loglikelihoods, logpriors]).T
                    evals1.append(evals)
                evals1 = np.array(evals1)
                self.assertTrue(np.all(evals1 == evals2))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertIn('Writing evaluations to', text)
            self.assertIn('evals_0.csv', text)

        # Test writing chains and evals to disk (with LogLikelihood)
        mcmc = pints.MCMCController(self.log_likelihood, nchains, xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(cpath)
                mcmc.set_log_pdf_filename(epath)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are created afterwards
                chains1 = mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertTrue(os.path.exists(p0))
                self.assertTrue(os.path.exists(p1))
                self.assertTrue(os.path.exists(p2))
                self.assertTrue(os.path.exists(p3))
                self.assertTrue(os.path.exists(p4))
                self.assertTrue(os.path.exists(p5))

                # Test chain files contain the correct values
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, nchains))
                self.assertTrue(np.all(chains1 == chains2))

                # Test eval files contain the correct values
                evals2 = np.array(io.load_samples(epath, nchains))
                evals1 = []
                for chain in chains1:
                    evals1.append(
                        np.array([self.log_likelihood(x) for x in chain]).T)
                evals1 = np.array(evals1).reshape(3, 20, 1)
                self.assertTrue(np.all(evals1 == evals2))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertIn('Writing evaluations to', text)
            self.assertIn('evals_0.csv', text)

        # Test logging can be disabled again
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs)
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(cpath)
                mcmc.set_log_pdf_filename(epath)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are not created afterwards
                mcmc.set_chain_filename(None)
                mcmc.set_log_pdf_filename(None)
                mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

            text = c.text()
            self.assertNotIn('Writing chains to', text)
            self.assertNotIn('chain_0.csv', text)
            self.assertNotIn('Writing evaluations to', text)
            self.assertNotIn('evals_0.csv', text)

        # Test with a single chain
        nchains = 1
        mcmc = pints.MCMCController(self.log_posterior, nchains, xs[:1])
        mcmc.set_initial_phase_iterations(5)
        mcmc.set_max_iterations(20)
        mcmc.set_log_to_screen(True)
        mcmc.set_log_to_file(False)

        with StreamCapture() as c:
            with TemporaryDirectory() as d:
                cpath = d.path('chain.csv')
                p0 = d.path('chain_0.csv')
                p1 = d.path('chain_1.csv')
                p2 = d.path('chain_2.csv')
                epath = d.path('evals.csv')
                p3 = d.path('evals_0.csv')
                p4 = d.path('evals_1.csv')
                p5 = d.path('evals_2.csv')

                # Test files aren't created before mcmc runs
                mcmc.set_chain_filename(cpath)
                mcmc.set_log_pdf_filename(epath)
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertFalse(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertFalse(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test files are created afterwards
                chains1 = mcmc.run()
                self.assertFalse(os.path.exists(cpath))
                self.assertFalse(os.path.exists(epath))
                self.assertTrue(os.path.exists(p0))
                self.assertFalse(os.path.exists(p1))
                self.assertFalse(os.path.exists(p2))
                self.assertTrue(os.path.exists(p3))
                self.assertFalse(os.path.exists(p4))
                self.assertFalse(os.path.exists(p5))

                # Test chain files contain the correct values
                import pints.io as io
                chains2 = np.array(io.load_samples(cpath, nchains))
                self.assertTrue(np.all(chains1 == chains2))

                # Test eval files contain the correct values
                evals2 = np.array(io.load_samples(epath, nchains))
                evals1 = []
                for chain in chains1:
                    logpdfs = np.array([self.log_posterior(x) for x in chain])
                    logpriors = np.array([self.log_prior(x) for x in chain])
                    loglikelihoods = logpdfs - logpriors
                    evals = np.array([logpdfs, loglikelihoods, logpriors]).T
                    evals1.append(evals)
                evals1 = np.array(evals1)
                self.assertTrue(np.all(evals1 == evals2))

            text = c.text()
            self.assertIn('Writing chains to', text)
            self.assertIn('chain_0.csv', text)
            self.assertIn('Writing evaluations to', text)
            self.assertIn('evals_0.csv', text)

    def test_deprecated_alias(self):

        mcmc = pints.MCMCSampling(
            self.log_posterior, 1, [self.real_parameters])
        self.assertIsInstance(mcmc, pints.MCMCController)

    def test_output_single_chain_samplers(self):
        # Tests if single-chain sampler output is stored correctly

        # Run with a given list of chains, return obtained output
        def run(chains):

            # Filter nones to get expected output
            expected = np.array(
                [[x for x in chain if x is not None] for chain in chains])

            # Get initial position
            x0 = [chain[0] for chain in chains]

            # Create log pdf
            f = SumDistribution(len(x0[0]))

            # Get expected evaluations
            exp_evals = np.array(
                [[[f(x)] for x in chain] for chain in expected])

            # Set up controller
            nc = len(x0)
            mcmc = pints.MCMCController(f, nc, x0, method=SingleListSampler)
            mcmc.set_log_to_screen(False)
            mcmc.set_max_iterations(len(expected[0]))

            # Pass chains to samplers
            for i, sampler in enumerate(mcmc.samplers()):
                sampler.set_chain(chains[i])

            # Run, while logging to disk
            with TemporaryDirectory() as d:
                # Store chains
                chain_path = d.path('chain.csv')
                mcmc.set_chain_filename(chain_path)

                # Store log pdfs
                evals_path = d.path('evals.csv')
                mcmc.set_log_pdf_filename(evals_path)

                # Run
                obtained = mcmc.run()

                # Load chains and log_pdfs
                disk_samples = np.array(pints.io.load_samples(chain_path, nc))
                disk_evals = np.array(pints.io.load_samples(evals_path, nc))

            # Return expected and obtained values
            return expected, obtained, disk_samples, exp_evals, disk_evals

        # One single-chain sampler, no None objects
        chains1 = [[[2], [4], [6], [3], [5]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [[[1, 2], [4, 3], [6, 1], [2, 2], [5, 7]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        # One single-chain sampler, with None objects
        chains1 = [[[1], [3], None, [1], None, [5], [2]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [[[1], [3], None, [1], None, [5], None, None, None, [2]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [[[1, 2], [4, 3], None, [6, 1], None, [2, 2], [5, 7]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        # Multiple single-chain samplers, no None objects
        chains1 = [
            [[2], [4], [6], [3], [5]],
            [[5], [1], [3], [3], [2]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [
            [[1, 2], [4, 3], [6, 1], [2, 2], [5, 7]],
            [[4, 3], [1, 1], [3, 5], [1, 4], [4, 7]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        # Multiple single-chain samplers, None at same index
        chains1 = [
            [[2], None, None, [4], [6], None, [3], None, None, [5]],
            [[5], None, None, [1], [3], None, [3], None, None, [2]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [
            [[1, 2], None, [4, 3], [6, 1], None, None, None, [2, 2], [5, 7]],
            [[4, 3], None, [1, 1], [3, 5], None, None, None, [1, 4], [4, 7]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        # Multiple single-chain samplers, None at different indices

        chains1 = [
            [[2], None, None, [4], [6], None, [3], None, None, [5]],
            [[5], None, None, [1], [3], None, [3], None, None, [2]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [
            [[1, 2], [4, 3], [6, 1], None, None, None, [2, 2], None, [5, 7]],
            [[4, 3], None, [1, 1], [3, 5], None, None, [1, 4], [4, 7]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

    def test_output_multi_chain_samplers(self):
        # Tests if multi-chain sampler output is stored correctly

        # Run with a given list of chains, return obtained output
        def run(chains):

            # Filter nones to get expected output
            expected = np.array(
                [[x for x in chain if x is not None] for chain in chains])

            # Get initial positions
            x0 = [chain[0] for chain in chains]

            # Create log pdf
            f = SumDistribution(len(x0[0]))

            # Get expected evaluations
            exp_evals = np.array(
                [[[f(x)] for x in chain] for chain in expected])

            # Set up controller
            nc = len(x0)
            mcmc = pints.MCMCController(f, nc, x0, method=MultiListSampler)
            mcmc.set_log_to_screen(False)
            mcmc.set_max_iterations(len(expected[0]))

            # Pass chains to sampler
            mcmc.sampler().set_chains(chains)

            # Run, while logging to disk
            with TemporaryDirectory() as d:
                # Store chains
                chain_path = d.path('chain.csv')
                mcmc.set_chain_filename(chain_path)

                # Store log pdfs
                evals_path = d.path('evals.csv')
                mcmc.set_log_pdf_filename(evals_path)

                # Run
                obtained = mcmc.run()

                # Load chains and log_pdfs
                disk_samples = np.array(pints.io.load_samples(chain_path, nc))
                disk_evals = np.array(pints.io.load_samples(evals_path, nc))

            # Return expected and obtained values
            return expected, obtained, disk_samples, exp_evals, disk_evals

        # Test with a single chain, no None objects
        chains1 = [[[2], [2], [6], [3], [0.5]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        chains1 = [[[1, 2], [2, 4], [3, 6], [8, 8], [1, 2]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        # Test with three chains, no None objects
        chains1 = [
            [[1, 2], [2, 4], [3, 6], [8, 8], [1, 2]],
            [[2, 3], [3, 5], [4, 7], [9, 8], [3, 2]],
            [[3, 4], [4, 6], [5, 8], [8, 3], [2, 7]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (3, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (3, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (3, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        # Test with a single chain, some None objects
        chains1 = [[[2], None, [2], None, [6], None, None, [3], [0.5]]]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (1, 5, 1))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (1, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))

        # Test with two chains, some None objects
        chains1 = [
            [[1, 2], [2, 4], None, [3, 6], None, None, [8, 8], None, [1, 2]],
            [[3, 4], [4, 6], None, [5, 8], None, None, [8, 3], None, [2, 7]],
        ]
        chains1, chains2, chains3, log_pdfs1, log_pdfs2 = run(chains1)
        self.assertEqual(chains2.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains2))
        self.assertEqual(chains3.shape, (2, 5, 2))
        self.assertTrue(np.all(chains1 == chains3))
        self.assertEqual(log_pdfs2.shape, (2, 5, 1))
        self.assertTrue(np.all(log_pdfs1 == log_pdfs2))


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

#!/usr/bin/env python3
#
# Tests SMC implementation of the SMCSampler
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#import re
import unittest
import numpy as np

import pints
import pints.toy

from shared import StreamCapture

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


method = pints.SMC


LOG_SCREEN = [
    'Using Sequential Monte Carlo',
    'Total number of particles: 10',
    'Number of temperatures: 3',
    'Number of MCMC steps at each temperature: 1',
    'Running in sequential mode.',
    'Iter. Eval. Temperature ESS       Acc.      Time m:s',
    '0     10     0.9999      9.733887  0          0:00.0',
    '1     20     0.99        9.251911  0.455      0:00.0',
    '2     30     0           3.04676   0.429      0:00.0',
]


class TestSMC(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`SMC`.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare for the test. """
        cls.pdf = pints.toy.MultimodalGaussianLogPDF()
        cls.prior = pints.UniformLogPrior([-5, -5], [15, 15])
        cls.sigma0 = [[2, 0], [0, 2]]

    def test_flow(self):

        sampler = pints.SMC(self.prior)

        # Can't call tell before ask
        self.assertRaisesRegex(RuntimeError, 'expecting ask', sampler.tell, 5)

        # Can't call ask twice
        xs = sampler.ask()
        self.assertRaisesRegex(RuntimeError, 'expecting tell', sampler.ask)

        # Can't call tell when ask is required
        fxs = [self.pdf(x) for x in xs]
        sampler.tell(fxs)
        self.assertRaisesRegex(
            RuntimeError, 'expecting ask', sampler.tell, fxs)

        # Can't run too many times
        smc = pints.SMCController(self.pdf, self.prior, method=method)
        smc.sampler().set_n_particles(10)
        smc.sampler().set_n_kernel_samples(1)
        smc.sampler().set_temperature_schedule(4)
        smc.set_log_to_screen(False)
        smc.run()
        self.assertRaisesRegex(
            RuntimeError, 'maximum number of iterations', smc.sampler().ask)

        smc = pints.SMCController(self.pdf, self.prior, method=method)
        smc.sampler().set_n_particles(10)
        smc.sampler().set_n_kernel_samples(3)
        smc.sampler().set_temperature_schedule(4)
        smc.set_log_to_screen(False)
        smc.run()
        self.assertRaisesRegex(
            RuntimeError, 'maximum number of iterations', smc.sampler().ask)

        # Can't give wrong number of samples in initial tell()
        sampler = pints.SMC(self.prior)
        sampler.set_batch_size(4)
        x0 = sampler.ask()
        self.assertRaisesRegex(
            ValueError, 'does not match number requested',
            sampler.tell, [1] * (len(x0) + 1))

        # Can't give wrong number of samples in ordinary tell()
        sampler = pints.SMC(self.prior)
        sampler.set_batch_size(4)
        x0 = sampler.ask()
        sampler.tell([1] * len(x0))
        x0 = sampler.ask()
        self.assertRaisesRegex(
            ValueError, 'does not match number requested',
            sampler.tell, [1] * (len(x0) + 1))

    def test_run(self):

        # Test with 1 kernel sample
        n = 10
        d = 2
        smc = pints.SMCController(
            self.pdf, self.prior, self.sigma0, method=method)
        smc.sampler().set_temperature_schedule(5)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_n_kernel_samples(1)
        smc.set_log_to_screen(False)
        samples = smc.run()

        # Check output has desired shape
        self.assertEqual(samples.shape, (n, d))

        # Check creation with sigma vector
        sigma0 = np.array([2, 2])
        smc = pints.SMCController(self.pdf, self.prior, sigma0, method=method)

        # Test with multiple kernel samples
        smc.sampler().set_temperature_schedule(3)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_n_kernel_samples(3)
        smc.set_log_to_screen(False)
        samples = smc.run()
        self.assertEqual(samples.shape, (n, d))

        # Test with multiple kernel samples, no resampling end 2 3
        smc = pints.SMCController(self.pdf, self.prior, sigma0, method=method)
        smc.sampler().set_temperature_schedule(3)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_n_kernel_samples(3)
        smc.sampler().set_resample_end_2_3(False)
        smc.set_log_to_screen(False)
        samples = smc.run()
        self.assertEqual(samples.shape, (n, d))

        # Test with user-specified ess threshold
        smc = pints.SMCController(self.pdf, self.prior, sigma0, method=method)
        smc.sampler().set_temperature_schedule(3)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_n_kernel_samples(3)
        smc.sampler().set_resample_end_2_3(False)
        smc.sampler().set_ess_threshold(3)
        smc.set_log_to_screen(False)
        samples = smc.run()
        self.assertEqual(samples.shape, (n, d))

        # Test unsetting user-specified ess threshold
        smc = pints.SMCController(self.pdf, self.prior, sigma0, method=method)
        smc.sampler().set_temperature_schedule(3)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_n_kernel_samples(3)
        smc.sampler().set_resample_end_2_3(False)
        smc.sampler().set_ess_threshold(1)
        smc.sampler().set_ess_threshold(n)
        self.assertRaisesRegex(
            ValueError, 'greater than zero',
            smc.sampler().set_ess_threshold, 0)
        self.assertRaisesRegex(
            ValueError, 'lower than or equal',
            smc.sampler().set_ess_threshold, n + 1)
        smc.sampler().set_ess_threshold(None)
        smc.set_log_to_screen(False)
        samples = smc.run()
        self.assertEqual(samples.shape, (n, d))

        # Sneakily set invalid ess threshold
        smc = pints.SMCController(self.pdf, self.prior, sigma0, method=method)
        smc.sampler().set_temperature_schedule(3)
        smc.sampler().set_n_particles(n * 2)
        smc.sampler().set_ess_threshold(n + 1)
        smc.sampler().set_n_particles(n)
        smc.set_log_to_screen(False)
        self.assertRaisesRegex(RuntimeError, 'lower than or equal', smc.run)

        # Test with batch size > 1
        smc = pints.SMCController(self.pdf, self.prior, sigma0, method=method)
        smc.sampler().set_temperature_schedule(3)
        smc.sampler().set_n_particles(10)
        smc.sampler().set_batch_size(5)
        smc.set_log_to_screen(False)
        samples = smc.run()
        self.assertEqual(samples.shape, (10, 2))

        # Test with batch size == n_particles
        smc = pints.SMCController(self.pdf, self.prior, sigma0, method=method)
        smc.sampler().set_temperature_schedule(3)
        smc.sampler().set_n_particles(10)
        smc.sampler().set_batch_size(10)
        smc.set_log_to_screen(False)
        samples = smc.run()
        self.assertEqual(samples.shape, (10, 2))

        # Test with akward batch size
        smc = pints.SMCController(self.pdf, self.prior, sigma0, method=method)
        smc.sampler().set_temperature_schedule(3)
        smc.sampler().set_n_particles(11)
        smc.sampler().set_batch_size(5)
        smc.set_log_to_screen(False)
        samples = smc.run()
        self.assertEqual(samples.shape, (11, 2))

    def test_info_methods(self):

        n = 10
        smc = pints.SMCController(
            self.pdf, self.prior, self.sigma0, method=method)
        smc.sampler().set_n_particles(10)

        self.assertEqual('Sequential Monte Carlo', smc.sampler().name())

        # Test weights are none initially
        w = smc.sampler().weights()
        self.assertIsNone(w)

        # Test ess is none initially
        e = smc.sampler().ess()
        self.assertIsNone(e)

        smc.set_log_to_screen(False)
        smc.run()

        # Test weights and ess
        w = smc.sampler().weights()
        self.assertEqual(w.shape, (n, ))

        # Test ess is none initially
        e = smc.sampler().ess()
        self.assertIsInstance(e, float)

    def test_high_ess_threshold(self):

        # Set very high ess threshold, triggering lots of resampling
        n = 10
        smc = pints.SMCController(self.pdf, self.prior, method=method)
        smc.sampler().set_temperature_schedule(3)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_ess_threshold(n)
        smc.set_log_to_screen(False)
        smc.run()

    def test_logging(self):

        np.random.seed(1)
        n = 10

        # No output
        with StreamCapture() as capture:
            smc = pints.SMCController(self.pdf, self.prior, method=method)
            smc.sampler().set_temperature_schedule(3)
            smc.sampler().set_n_particles(n)
            smc.set_log_to_screen(False)
            smc.set_log_to_file(False)
            smc.run()
        self.assertEqual(capture.text(), '')

        # With output to screen
        np.random.seed(1)
        with StreamCapture() as capture:
            smc = pints.SMCController(self.pdf, self.prior, method=method)
            smc.sampler().set_temperature_schedule(3)
            smc.sampler().set_n_particles(n)
            smc.set_log_to_screen(True)
            smc.set_log_to_file(False)
            smc.run()
        lines = capture.text().splitlines()
        for i, line in enumerate(lines):
            self.assertLess(i, len(LOG_SCREEN))
            # Chop off time bit before comparison
            if LOG_SCREEN[i][-6:] == '0:00.0':
                self.assertEqual(line[:-6], LOG_SCREEN[i][:-6])
            else:
                self.assertEqual(line, LOG_SCREEN[i])
        self.assertEqual(len(lines), len(LOG_SCREEN))


if __name__ == '__main__':
    unittest.main()

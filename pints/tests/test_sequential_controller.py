#!/usr/bin/env python3
#
# Tests the SMCController
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints
import pints.toy
import unittest

from shared import StreamCapture, TemporaryDirectory

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


LOG_SCREEN_1 = [
    'Using Sequential Monte Carlo',
    'Total number of particles: 10',
    'Number of temperatures: 11',
    'Number of MCMC steps at each temperature: 1',
    'Running in sequential mode.',
    'Iter. Eval. Temperature ESS       Acc.      Time m:s',
    '0     10     0.9999      9.999924  0          0:00.0',
    '1     20     0.99975     9.999819  0.545      0:00.0',
    '2     30     0.99937     9.999884  0.476      0:00.0',
    '3     40     0.99842     9.999573  0.452      0:00.0',
    #'4     50     0.99602     9.99513   0.439      0:00.0',
    #'5     60     0.99        9.940233  0.412      0:00.0',
    '6     70     0.97488     9.656355  0.361      0:00.0',
    #'7     80     0.9369      9.290603  0.338      0:00.0',
    #'8     90     0.84151     7.97389   0.321      0:00.0',
    '9     100    0.60189     8.300343  0.319      0:00.0',
    '10    110    0           5.65693   0.307      0:00.0',
]

LOG_FILE = [
    'Iter. Eval. Temperature ESS       Acc.      Time m:s',
    '0     10     0.9999      9.88839   0          0:00.0',
    '1     20     0.99        9.620243  0.545      0:00.0',
    '2     30     0           1.339102  0.381      0:00.0',
]


class TestSMCController(unittest.TestCase):
    """
    Tests the SMCController class.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare problem for tests. """

        cls.pdf = pints.toy.GaussianLogPDF([0, 0], [3, 3])
        cls.prior = pints.ComposedLogPrior(
            pints.GaussianLogPrior(0, 5), pints.GaussianLogPrior(2, 7))
        cls.sigma0 = [[5, 0], [0, 5]]

    def test_creation(self):

        n = 20

        # First argument must be log pdf
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPDF',
            pints.SMCController, 12, self.prior)

        # Second argument must be prior
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.SMCController, self.prior, self.pdf)

        # Prior and pdf dimensions must match
        self.assertRaisesRegex(
            ValueError, 'same number of parameters',
            pints.SMCController, self.pdf, pints.GaussianLogPrior(0, 2))

        # Method must be an smc method
        self.assertRaisesRegex(
            ValueError, 'must extend pints.SMCSampler',
            pints.SMCController, self.pdf, self.prior,
            method=pints.AdaptiveCovarianceMCMC)
        self.assertRaisesRegex(
            ValueError, 'must extend pints.SMCSampler',
            pints.SMCController, self.pdf, self.prior,
            method='pints.SMC')

        # Create with sigma
        smc = pints.SMCController(self.pdf, self.prior, self.sigma0)
        smc.set_log_to_screen(False)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_temperature_schedule(10)
        samples = smc.run()
        self.assertEqual(samples.shape, (n, 2))

        # Create without sigma
        smc = pints.SMCController(self.pdf, self.prior)
        smc.set_log_to_screen(False)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_temperature_schedule(10)
        samples = smc.run()
        self.assertEqual(samples.shape, (n, 2))

        # Create with method
        smc = pints.SMCController(self.pdf, self.prior, method=pints.SMC)
        self.assertIsInstance(smc.sampler(), pints.SMC)

    def test_run(self):

        n = 20
        smc = pints.SMCController(self.pdf, self.prior)
        smc.set_log_to_screen(False)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_temperature_schedule(10)
        samples = smc.run()
        self.assertEqual(samples.shape, (n, 2))

        # Can't run twice
        self.assertRaisesRegex(
            RuntimeError, 'only be run once', smc.run)

    def test_parallel(self):

        # Set True
        n = 20
        smc = pints.SMCController(self.pdf, self.prior)
        smc.set_log_to_screen(False)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_temperature_schedule(10)
        smc.set_parallel(True)
        self.assertEqual(smc.parallel(), True)
        samples = smc.run()
        self.assertEqual(samples.shape, (n, 2))

        # Set False
        smc = pints.SMCController(self.pdf, self.prior)
        smc.set_log_to_screen(False)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_temperature_schedule(10)
        smc.set_parallel(False)
        self.assertEqual(smc.parallel(), False)
        smc.set_parallel(True)
        self.assertEqual(smc.parallel(), True)
        smc.set_parallel(0)
        self.assertEqual(smc.parallel(), False)
        samples = smc.run()
        self.assertEqual(samples.shape, (n, 2))

        # Set numeric
        smc = pints.SMCController(self.pdf, self.prior)
        smc.set_log_to_screen(False)
        smc.sampler().set_n_particles(n)
        smc.sampler().set_temperature_schedule(10)
        smc.set_parallel(2)
        self.assertEqual(smc.parallel(), True)
        samples = smc.run()
        self.assertEqual(samples.shape, (n, 2))

        # Parallel cannot be configured after run
        self.assertRaisesRegex(
            RuntimeError, 'after run', smc.set_parallel, True)

    def test_logging(self):

        # Unlike methods, test with periodic output to screen
        n = 10
        np.random.seed(1)
        with StreamCapture() as capture:
            smc = pints.SMCController(self.pdf, self.prior)
            smc.sampler().set_temperature_schedule(11)
            smc.sampler().set_n_particles(n)
            smc.set_log_to_screen(True)
            smc.set_log_to_file(False)
            smc.set_log_interval(3, 3)
            smc.run()
        lines = capture.text().splitlines()
        for i, line in enumerate(lines):
            self.assertLess(i, len(LOG_SCREEN_1))
            if LOG_SCREEN_1[i][-6:] == '0:00.0':
                self.assertEqual(line[:-6], LOG_SCREEN_1[i][:-6])
            else:
                self.assertEqual(line, LOG_SCREEN_1[i])
        self.assertEqual(len(lines), len(LOG_SCREEN_1))

        # With output to file
        np.random.seed(1)
        with StreamCapture() as capture:
            with TemporaryDirectory() as d:
                filename = d.path('test.txt')
                smc = pints.SMCController(self.pdf, self.prior)
                smc.sampler().set_temperature_schedule(3)
                smc.sampler().set_n_particles(n)
                smc.set_log_to_screen(False)
                smc.set_log_to_file(filename)
                smc.run()
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

        # Can't change log settings after run
        self.assertRaisesRegex(
            RuntimeError, 'after run', smc.set_log_interval, 1)
        self.assertRaisesRegex(
            RuntimeError, 'after run', smc.set_log_to_screen, True)
        self.assertRaisesRegex(
            RuntimeError, 'after run', smc.set_log_to_file, True)

        # And in parallel mode
        np.random.seed(1)
        with StreamCapture() as capture:
            smc = pints.SMCController(self.pdf, self.prior)
            smc.sampler().set_temperature_schedule(3)
            smc.sampler().set_n_particles(n)
            smc.set_log_to_screen(True)
            smc.set_log_to_file(False)
            smc.set_parallel(2)
            smc.run()
        lines = capture.text()
        self.assertIn('2 worker processes', lines)

        # Test invalid logging interval settings
        smc = pints.SMCController(self.pdf, self.prior)
        self.assertRaisesRegex(
            ValueError, 'greater than zero', smc.set_log_interval, 0)


if __name__ == '__main__':
    unittest.main()

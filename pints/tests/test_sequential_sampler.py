#!/usr/bin/env python3
#
# Tests the SMCSampler base class
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

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestSMCSampler(unittest.TestCase):
    """
    Tests get/set methods on SMCSampler.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare for the test. """
        cls.pdf = pints.toy.MultimodalGaussianLogPDF()
        cls.prior = pints.UniformLogPrior([-5, -5], [15, 15])
        cls.sigma0 = [[2, 0], [0, 2]]

    def test_creation(self):

        # First arg must be a log prior
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.SMC, self.pdf, self.sigma0)

        # Sigma0 is optional
        pints.SMC(self.prior)

    def test_flow(self):

        sampler = pints.SMC(self.prior)

        # Configuration is OK, before running
        sampler.set_n_kernel_samples(20)
        sampler.set_n_particles(20)
        sampler.set_temperature_schedule(10)
        sampler.set_batch_size(4)

        sampler.ask()

        # Can't change settings during run
        self.assertRaisesRegex(
            RuntimeError, 'during run', sampler.set_n_kernel_samples, 20)
        self.assertRaisesRegex(
            RuntimeError, 'during run', sampler.set_n_particles, 20)
        self.assertRaisesRegex(
            RuntimeError, 'during run', sampler.set_temperature_schedule, 10)
        self.assertRaisesRegex(
            RuntimeError, 'during run', sampler.set_batch_size, 4)

    def test_get_set(self):

        n = 10
        smc = pints.SMCController(self.pdf, self.prior, self.sigma0)

        smc.sampler().set_temperature_schedule(6)
        self.assertEqual(smc.sampler().n_temperatures(), 6)
        smc.sampler().set_temperature_schedule(5)
        self.assertEqual(smc.sampler().n_temperatures(), 5)
        smc.sampler().set_n_particles(n + 1)
        self.assertEqual(smc.sampler().n_particles(), n + 1)
        smc.sampler().set_n_particles(n)
        self.assertEqual(smc.sampler().n_particles(), n)
        smc.sampler().set_n_kernel_samples(10)
        self.assertEqual(smc.sampler().n_kernel_samples(), 10)
        smc.sampler().set_n_kernel_samples(1)
        self.assertEqual(smc.sampler().n_kernel_samples(), 1)
        smc.sampler().set_batch_size(1)
        self.assertEqual(smc.sampler().batch_size(), 1)
        smc.sampler().set_batch_size(2)
        self.assertEqual(smc.sampler().batch_size(), 2)
        smc.sampler().set_batch_size(10)
        self.assertEqual(smc.sampler().batch_size(), 10)

        # Test invalid values
        self.assertRaisesRegex(
            ValueError, 'must be >= 1',
            smc.sampler().set_n_kernel_samples, 0)
        smc.sampler().set_n_kernel_samples(1)
        self.assertRaisesRegex(
            ValueError, 'at least 10',
            smc.sampler().set_n_particles, 9)
        smc.sampler().set_n_particles(10)
        self.assertRaisesRegex(
            ValueError, 'at least two',
            smc.sampler().set_temperature_schedule, 1)
        smc.sampler().set_temperature_schedule(2)
        self.assertRaisesRegex(
            ValueError, '1 or greater',
            smc.sampler().set_batch_size, 0)
        smc.sampler().set_batch_size(3)

        # Test setting custom schedule
        sched = [0, 0.1, 0.2, 0.5, 1]
        smc.sampler().set_temperature_schedule(sched)
        self.assertEqual(smc.sampler().n_temperatures(), len(sched))
        sched = [0, 0.1, 0.2, 0.5, 0.6, 1]
        smc.sampler().set_temperature_schedule(sched)
        self.assertEqual(smc.sampler().n_temperatures(), len(sched))

        # Test setting invalid schedules
        self.assertRaisesRegex(
            ValueError, 'at least two',
            smc.sampler().set_temperature_schedule, [0])
        self.assertRaisesRegex(
            ValueError, 'First element',
            smc.sampler().set_temperature_schedule, [0.1, 0.5, 1])
        self.assertRaisesRegex(
            ValueError, 'non-negative',
            smc.sampler().set_temperature_schedule, [0, -0.5, 1])
        self.assertRaisesRegex(
            ValueError, 'exceed 1',
            smc.sampler().set_temperature_schedule, [0, 0.5, 1.1])


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
#
# Tests ellipsoidal nested sampler.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy
from pints._nested.__init__ import Ellipsoid

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestNestedEllipsoidSampler(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`NestedEllipsoidSampler`.
    """

    @classmethod
    def setUpClass(cls):
        """ Prepare for the test. """
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
            [0.01, 400],
            [0.02, 600]
        )

        # Create a log-likelihood
        cls.log_likelihood = pints.GaussianKnownSigmaLogLikelihood(
            problem, cls.noise)

    def test_construction_errors(self):
        # Tests if invalid constructor calls are picked up.

        # First arg must be a log likelihood
        self.assertRaisesRegex(
            ValueError, 'must extend pints.LogPrior',
            pints.NestedEllipsoidSampler, self.log_likelihood)

    def test_hyper_params(self):
        # Tests the hyper parameter interface is working.
        sampler = pints.NestedEllipsoidSampler(self.log_prior)
        self.assertEqual(sampler.n_hyper_parameters(), 6)
        sampler.set_hyper_parameters([220, 130, 2.0, 133, 1, 0.8])
        self.assertEqual(sampler.n_active_points(), 220)
        self.assertEqual(sampler.n_rejection_samples(), 130)
        self.assertEqual(sampler.enlargement_factor(), 2.0)
        self.assertEqual(sampler.ellipsoid_update_gap(), 133)
        self.assertTrue(sampler.dynamic_enlargement_factor())
        self.assertTrue(sampler.alpha(), 0.8)

    def test_getters_and_setters(self):
        # Tests various get() and set() methods.
        sampler = pints.NestedEllipsoidSampler(self.log_prior)

        # Active points
        x = sampler.n_active_points() + 1
        self.assertNotEqual(sampler.n_active_points(), x)
        sampler.set_n_active_points(x)
        self.assertEqual(sampler.n_active_points(), x)
        self.assertRaisesRegex(
            ValueError, 'greater than 5', sampler.set_n_active_points, 5)

        # Rejection samples
        x = sampler.n_rejection_samples() + 1
        self.assertNotEqual(sampler.n_rejection_samples(), x)
        sampler.set_n_rejection_samples(x)
        self.assertEqual(sampler.n_rejection_samples(), x)
        self.assertRaisesRegex(
            ValueError, 'negative', sampler.set_n_rejection_samples, -1)

        # Enlargement factor
        x = sampler.enlargement_factor() * 2
        self.assertNotEqual(sampler.enlargement_factor(), x)
        sampler.set_enlargement_factor(x)
        self.assertEqual(sampler.enlargement_factor(), x)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_enlargement_factor, 0.5)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_enlargement_factor, 1)

        # Ellipsoid update gap
        x = sampler.ellipsoid_update_gap() * 2
        self.assertNotEqual(sampler.ellipsoid_update_gap(), x)
        sampler.set_ellipsoid_update_gap(x)
        self.assertEqual(sampler.ellipsoid_update_gap(), x)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_ellipsoid_update_gap, 0.5)
        self.assertRaisesRegex(
            ValueError, 'exceed 1', sampler.set_ellipsoid_update_gap, 1)

        # dynamic enlargement factor
        self.assertTrue(not sampler.dynamic_enlargement_factor())
        sampler.set_dynamic_enlargement_factor(1)
        self.assertTrue(sampler.dynamic_enlargement_factor())

        # alpha
        self.assertRaises(ValueError, sampler.set_alpha, -0.2)
        self.assertRaises(ValueError, sampler.set_alpha, 1.2)
        self.assertEqual(sampler.alpha(), 0.2)
        sampler.set_alpha(0.4)
        self.assertEqual(sampler.alpha(), 0.4)

        # initial phase
        self.assertTrue(sampler.needs_initial_phase())
        self.assertTrue(sampler.in_initial_phase())
        sampler.set_initial_phase(False)
        self.assertTrue(not sampler.in_initial_phase())
        self.assertEqual(sampler.name(), 'Nested ellipsoidal sampler')

    def test_ask_tell(self):
        # Tests ask and tell

        # test that ellipses are estimated
        sampler = pints.NestedEllipsoidSampler(self.log_prior)
        sampler.set_n_rejection_samples(100)
        sampler.set_ellipsoid_update_gap(10)
        for i in range(5000):
            pt = sampler.ask(1)
            fx = self.log_likelihood(pt)
            sampler.tell(fx)
        self.assertTrue(isinstance(sampler.ellipsoid(), Ellipsoid))

        # test multiple points being asked and tell'd
        sampler = pints.NestedEllipsoidSampler(self.log_prior)
        pts = sampler.ask(50)
        self.assertEqual(len(pts), 50)
        fx = [self.log_likelihood(pt) for pt in pts]
        proposed = sampler.tell(fx)
        self.assertTrue(len(proposed) > 1)

        # test multiple ask points after rejection samples
        sampler = pints.NestedEllipsoidSampler(self.log_prior)
        sampler.set_n_rejection_samples(10)
        for i in range(100):
            self.assertEqual(len(sampler.ask(20)), 20)

    def test_dynamic_enlargement_factor(self):
        # tests dynamic enlargement factor runs
        sampler = pints.NestedController(self.log_likelihood,
                                         self.log_prior)
        sampler.sampler().set_dynamic_enlargement_factor(1)
        sampler.set_log_to_screen(False)
        ef1 = sampler.sampler().enlargement_factor()
        sampler.run()
        ef2 = sampler.sampler().enlargement_factor()
        self.assertTrue(ef2 < ef1)

    def test_sensitivities(self):
        # tests whether sensitivities bit runs
        sampler = pints.NestedController(self.log_likelihood,
                                         self.log_prior)
        # hacky but currently no samplers need sensitivities
        sampler._needs_sensitivities = True
        sampler._initialise_callable()


if __name__ == '__main__':
    unittest.main()

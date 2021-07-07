#!/usr/bin/env python3
#
# Tests ellipsoidal nested sampler.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import division
import unittest
import numpy as np

import pints
import pints.toy
from pints._nested._multinest import EllipsoidTree
from pints._nested.__init__ import Ellipsoid

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


class TestMultiNestSampler(unittest.TestCase):
    """
    Unit (not functional!) tests for :class:`MultiNestSampler`.
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

    def test_getters_and_setters(self):
        # tests various get() and set() methods.
        controller = pints.NestedController(self.log_likelihood,
                                            self.log_prior,
                                            method=pints.MultiNestSampler)
        self.assertEqual(controller.sampler().f_s_threshold(), 1.1)
        controller.sampler().set_f_s_threshold(4)
        self.assertEqual(controller.sampler().f_s_threshold(), 4)
        self.assertRaises(ValueError, controller.sampler().set_f_s_threshold,
                          0.5)
        controller.sampler().set_ellipsoid_update_gap(43)
        self.assertEqual(controller.sampler().ellipsoid_update_gap(), 43)
        controller.sampler().set_enlargement_factor(4)
        self.assertEqual(controller.sampler().enlargement_factor(), 4)
        self.assertTrue(controller.sampler().in_initial_phase())
        self.assertEqual(controller.sampler().n_hyper_parameters(), 4)
        controller.sampler().set_n_rejection_samples(12)
        self.assertEqual(controller.sampler().n_rejection_samples(), 12)
        self.assertEqual(controller.sampler().name(),
                         "MultiNest sampler")
        self.assertTrue(controller.sampler().needs_initial_phase())
        controller.sampler().set_hyper_parameters([100, 100, 2, 30])

        # test errors
        self.assertRaises(ValueError,
                          controller.sampler().set_ellipsoid_update_gap,
                          0)
        self.assertRaises(ValueError,
                          controller.sampler().set_enlargement_factor,
                          0.5)
        self.assertRaises(ValueError,
                          controller.sampler().set_n_rejection_samples,
                          -1)

    def test_runs(self):
        # tests that sampler runs
        controller = pints.NestedController(self.log_likelihood,
                                            self.log_prior,
                                            method=pints.MultiNestSampler)
        controller.set_iterations(450)
        controller.set_log_to_screen(False)
        controller.run()

        # test getting ellipsoid tree post-run
        et = controller.sampler().ellipsoid_tree()
        self.assertTrue(et.n_leaf_ellipsoids() >= 1)

    def test_multiple(self):
        # tests that ask /tell work with multiple points

        # test multiple points being asked and tell'd
        sampler = pints.MultiNestSampler(self.log_prior)
        pts = sampler.ask(50)
        self.assertEqual(len(pts), 50)
        fx = [self.log_likelihood(pt) for pt in pts]
        proposed = sampler.tell(fx)
        self.assertTrue(len(proposed) > 1)

        # test with a longer run so that post-rejection sampling multiple
        # point evaluation gets triggered
        controller = pints.NestedController(self.log_likelihood,
                                            self.log_prior,
                                            method=pints.MultiNestSampler)
        controller.set_iterations(450)
        controller.set_log_to_screen(False)
        controller.set_parallel(True)
        controller.run()


class TestEllipsoidTree(unittest.TestCase):
    """
    Unit tests for the Ellipsoid binary tree class.
    """

    @classmethod
    def setUpClass(cls):
        # prepare for tests
        n = 1000
        gaussian = pints.MultivariateGaussianLogPrior([0, 0], [[1, 0], [0, 1]])
        cls.gaussian = gaussian
        draws = gaussian.sample(n)
        cls.draws = [gaussian.convert_to_unit_cube(x) for x in draws]

    def unit_draws(self, n):
        # helper function to produce unit draws
        draws = self.gaussian.sample(n)
        return [self.gaussian.convert_to_unit_cube(x) for x in draws]

    def test_construction(self):
        # tests construction

        # above threshold draws
        draws = self.unit_draws(20)
        EllipsoidTree(draws, 1)
        # below threshold draws
        draws = self.unit_draws(5)
        EllipsoidTree(draws, 1)

        # check errors
        # zero length points?
        self.assertRaises(ValueError, EllipsoidTree, [], 1)
        # negative iteration?
        self.assertRaises(ValueError, EllipsoidTree, self.draws, -1)
        # points within unit cube?
        gaussian = pints.MultivariateGaussianLogPrior([5, 5], [[1, 0], [0, 1]])
        draws = gaussian.sample(20)
        self.assertRaises(ValueError, EllipsoidTree, draws, 1)

    def test_getters(self):
        # tests get()

        tree = EllipsoidTree(self.draws, 100)

        # leaf nodes
        self.assertTrue(tree.n_leaf_ellipsoids() >= 1)
        leaves = np.copy(tree.leaf_ellipsoids())
        self.assertEqual(tree.n_leaf_ellipsoids(), len(leaves))
        [self.assertTrue(isinstance(x, Ellipsoid)) for x in leaves]

        # bounding ellipsoid
        ellipsoid = tree.ellipsoid()
        self.assertTrue(isinstance(ellipsoid, Ellipsoid))

        # Check a given draw is within bounding ellipsoids
        self.assertTrue(tree.count_within_leaf_ellipsoids(
            self.draws[0]) >= 1)
        self.assertTrue(tree.count_within_leaf_ellipsoids(
            [-100, -100]) == 0)

        # check f_s
        self.assertTrue(tree.f_s() >= 1)

        # updating triggers changes?
        f_s_old = np.copy(tree.f_s())
        tree.update_leaf_ellipsoids(1)
        self.assertFalse(tree.f_s() == f_s_old)

    def test_calculations(self):
        # tests the calculations done

        # vs
        n = 20
        iteration = 2
        draws = self.unit_draws(n)
        ellipsoidTree = EllipsoidTree(draws, iteration)
        self.assertEqual(ellipsoidTree.vs(), np.exp(-iteration / n))

        # vsk
        n1 = 5
        draws = self.unit_draws(n1)
        ellipsoid = Ellipsoid.minimum_volume_ellipsoid(draws)
        V_S = ellipsoidTree._V_S
        self.assertEqual(ellipsoidTree.vsk(ellipsoid), n1 * V_S / n)

        # more points in ellipsoid than tree?
        n1 = 21
        draws = self.unit_draws(n1)
        ellipsoid = Ellipsoid.minimum_volume_ellipsoid(draws)
        self.assertRaises(ValueError, ellipsoidTree.vsk, ellipsoid)

        # hk
        ellipsoid = ellipsoidTree.ellipsoid()
        d = Ellipsoid.mahalanobis_distance(self.draws[0],
                                           ellipsoid.weight_matrix(),
                                           ellipsoid.centroid())
        V_S_k = 1
        self.assertEqual(ellipsoid.volume() * d / V_S_k,
                         ellipsoidTree.h_k(self.draws[0], ellipsoid, V_S_k))

    def test_sampling(self):
        # tests sampling from leaf ellipsoids

        # check all samples in leaf ellipsoids
        tree = EllipsoidTree(self.draws, 100)
        samples = tree.sample_leaf_ellipsoids(100)
        for sample in samples:
            self.assertTrue(tree.count_within_leaf_ellipsoids(
                sample) >= 1)

    def test_splitting(self):
        # tests splitting and enlarging ellipsoids

        # singular matrix error?
        tree = EllipsoidTree(self.draws, 100)
        points = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        assignments = np.array([0, 0, 0])
        _, _, bool = tree.split_ellipsoids(points, assignments, 0)
        self.assertFalse(bool)

        # enlarging
        ellipsoid = tree.ellipsoid()
        tree.compare_enlarge(ellipsoid, 20)

    def test_multiple_nodes(self):
        # test that tree does actually split when supplied with multiple modes
        # and that functions still run ok with multiple leaves

        # need multiple replicates to ensure at least one splits
        n = 400
        nreps = 10
        log_pdf = pints.toy.AnnulusLogPDF()
        gaussian = pints.MultivariateGaussianLogPrior([0, 0],
                                                      [[100, 0], [0, 100]])
        n_leaves = []
        for rep in range(nreps):
            draws = log_pdf.sample(n)
            draws = [gaussian.convert_to_unit_cube(x) for x in draws]
            draws = np.vstack(draws)
            ellipsoid_tree = EllipsoidTree(draws, 600)
            n_leaf = ellipsoid_tree.n_leaf_ellipsoids()
            n_leaves.append(n_leaf)
            self.assertEqual(n_leaf, len(ellipsoid_tree.leaf_ellipsoids()))
            self.assertTrue(ellipsoid_tree.leaf_ellipsoids_volume() > 0)
            ellipsoid_tree.sample_leaf_ellipsoids(1000)
        self.assertTrue(sum(np.array(n_leaves) > 1) > 1)


if __name__ == '__main__':
    unittest.main()

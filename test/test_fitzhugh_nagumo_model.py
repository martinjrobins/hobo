#!/usr/bin/env python3
#
# Tests if the Fitzhugh-Nagumo toy model runs.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import pints
import pints.toy
import numpy as np


class TestFitzhughNagumoModel(unittest.TestCase):
    """
    Tests if the Fitzhugh-Nagumo toy model runs.
    """

    def test_run(self):

        # Test basic properties
        model = pints.toy.FitzhughNagumoModel()
        self.assertEqual(model.n_parameters(), 3)
        self.assertEqual(model.n_outputs(), 2)

        # Test simulation
        x = model.suggested_parameters()
        times = model.suggested_times()
        values = model.simulate(x, times)
        self.assertEqual(values.shape, (len(times), 2))

        # Simulation with sensitivities
        values, dvalues_dp = model.simulateS1(x, times)
        self.assertEqual(values.shape, (len(times), 2))
        self.assertEqual(dvalues_dp.shape, (len(times), 2, 3))

        values_only = model.simulate(x, times)
        self.assertAlmostEqual(np.linalg.norm(values - values_only), 0.0, 6)

        # Test alternative starting position
        model = pints.toy.FitzhughNagumoModel([0.1, 0.1])
        values = model.simulate(x, times)
        self.assertEqual(values.shape, (len(times), 2))

        # Times can't be negative
        times = [-1, 2, 3, 4]
        self.assertRaises(ValueError, model.simulate, x, times)

        # Initial value must have size 2
        pints.toy.FitzhughNagumoModel([1, 1])
        self.assertRaises(ValueError, pints.toy.FitzhughNagumoModel, [1])


if __name__ == '__main__':
    unittest.main()

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


class TestFitzhughNagumoModel(unittest.TestCase):
    """
    Tests if the Fitzhugh-Nagumo toy model runs.
    """
    def test_run(self):

        model = pints.toy.FitzhughNagumoModel()
        self.assertEqual(model.n_parameters(), 3)
        self.assertEqual(model.n_outputs(), 2)

        x = [1, 1, 1]
        times = [1, 2, 3, 4]
        values = model.simulate(x, times)
        self.assertEqual(values.shape, (4, 2))

        # Test alternative starting position
        model = pints.toy.FitzhughNagumoModel([0.1, 0.1])
        values = model.simulate(x, times)
        self.assertEqual(values.shape, (4, 2))

        # Test errors
        times = [-1, 2, 3, 4]
        self.assertRaises(ValueError, model.simulate, x, times)
        self.assertRaises(ValueError, pints.toy.FitzhughNagumoModel, [1])


if __name__ == '__main__':
    unittest.main()

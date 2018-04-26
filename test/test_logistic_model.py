#!/usr/bin/env python3
#
# Tests if the logistic (toy) model works.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import numpy as np
import pints
import pints.toy


class TestLogistic(unittest.TestCase):
    """
    Tests if the logistic (toy) model works.
    """

    def test_start_with_zero(self):
        # Test the special case where the initial size is zero
        model = pints.toy.LogisticModel(0)
        times = [0, 1, 2, 10000]
        parameters = [1, 5]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        for v in values:
            self.assertEqual(v, 0)

    def test_start_with_two(self):
        # Run small simulation
        model = pints.toy.LogisticModel(2)
        times = [0, 1, 2, 10000]
        parameters = [1, 5]
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 2)
        self.assertEqual(values[-1], parameters[-1])
        # Run large simulation
        times = np.arange(0, 1000)
        values = model.simulate(parameters, times)
        self.assertEqual(len(values), len(times))
        self.assertEqual(values[0], 2)
        self.assertEqual(values[-1], parameters[-1])
        self.assertTrue(np.all(values[1:] >= values[:-1]))
        self.assertTrue(np.all(values >= 2))
        self.assertTrue(np.all(values <= parameters[-1]))

    def test_negative_k(self):
        model = pints.toy.LogisticModel(2)
        times = [0, 1, 2, 10000]
        parameters = [1, -1]
        values = model.simulate(parameters, times)
        self.assertTrue(np.all(values == np.zeros(4)))

    def test_errors(self):
        model = pints.toy.LogisticModel(2)
        times = [0, -1, 2, 10000]
        parameters = [1, 0]
        self.assertRaises(ValueError, model.simulate, parameters, times)
        self.assertRaises(ValueError, pints.toy.LogisticModel, -1)


class TestLogisticS1(unittest.TestCase):
    """
    Tests if the logistic toy model with sensitivities works.
    """
    def test_errors(self):
        model = pints.toy.LogisticModel(2)
        times = [0, -1, 2, 10000]
        parameters = [1, 0]
        self.assertRaises(ValueError, model.simulate, parameters, times)
        self.assertRaises(ValueError, pints.toy.LogisticModel, -1)

    def test_simulate(self):
        model = pints.toy.LogisticModelWithSensitivities(2)
        times = [0, 1, 2, 10000]
        parameters = [1, 5]
        values, dvdp = model.simulate(parameters, times)
        self.assertEqual(dvdp[0, 0], 0)
        self.assertEqual(dvdp[-1, 0], 0.0)
        self.assertEqual(dvdp[0, 1], 0)
        self.assertEqual(dvdp[-1, 1], 1.0)


if __name__ == '__main__':
    unittest.main()

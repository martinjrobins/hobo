#!/usr/bin/env python
#
# Tests the basic methods of the PSO optimiser.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy as toy
import unittest
import numpy as np

debug = False

class TestPSO(unittest.TestCase):
    """
    Tests the basic methods of the PSO optimiser.
    """
    def __init__(self, name):
        super(TestPSO, self).__init__(name)
    
        # Create toy model
        self.model = toy.LogisticModel()
        self.real_parameters = [0.015, 500]
        self.times = np.linspace(0, 1000, 1000)
        self.values = self.model.simulate(self.real_parameters, self.times)

        # Create an object with links to the model and time series
        self.problem = pints.SingleSeriesProblem(
            self.model, self.times, self.values)

        # Select a score function
        self.score = pints.SumOfSquaresError(self.problem)

        # Select some boundaries
        self.boundaries = pints.Boundaries([0, 200], [1, 1000])
        
        # Set an initial position
        self.x0 = 0.014, 499
        
        # Set an initial guess of the standard deviation in each parameter
        self.sigma0 = [0.001, 1]
        
        # Minimum score function value to obtain
        self.cutoff = 1e3 # Global method!
        
        # Maximum tries before it counts as failed
        self.max_tries = 3

    def test_unbounded_no_hint(self):
        
        np.random.seed(1)
        opt = pints.PSO(self.score)
        opt.set_verbose(debug)
        found_parameters, found_solution = opt.run()
        # Will be terrible, don't check        
        
    def test_bounded_no_hint(self):
    
        np.random.seed(1)
        opt = pints.PSO(self.score, self.boundaries)
        opt.set_verbose(debug)
        for i in xrange(self.max_tries):        
            found_parameters, found_solution = opt.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)
        
    def test_unbounded_with_hint(self):
    
        np.random.seed(1)
        opt = pints.PSO(self.score, x0=self.x0)
        opt.set_verbose(debug)
        found_parameters, found_solution = opt.run()
        # Will be terrible, don't check        
        
    def test_bounded_with_hint(self):
    
        np.random.seed(1)
        opt = pints.PSO(self.score, self.boundaries, self.x0)
        opt.set_verbose(debug)
        for i in xrange(self.max_tries):        
            found_parameters, found_solution = opt.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)

    def test_bounded_with_hint_and_sigma(self):
    
        np.random.seed(1)
        opt = pints.PSO(self.score, self.boundaries, self.x0, self.sigma0)
        opt.set_verbose(debug)
        for i in xrange(self.max_tries):        
            found_parameters, found_solution = opt.run()
            if found_solution < self.cutoff:
                break
        self.assertTrue(found_solution < self.cutoff)

if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

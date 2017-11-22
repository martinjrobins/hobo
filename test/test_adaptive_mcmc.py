#!/usr/bin/env python
#
# Tests the basic methods of the adaptive covariance MCMC routine.
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
import matplotlib.pyplot as pl

debug = False

class TestAdaptiveCovarianceMCMC(unittest.TestCase):
    """
    Tests the basic methods of the adaptive covariance MCMC routine.
    """
    def __init__(self, name):
        super(TestAdaptiveCovarianceMCMC, self).__init__(name)
    
        # Create toy model
        self.model = toy.LogisticModel()
        self.real_parameters = [0.015, 500]
        self.times = np.linspace(0, 1000, 1000)
        self.values = self.model.simulate(self.real_parameters, self.times)
        
        # Add noise
        noise = 10
        self.values += np.random.normal(0, noise, self.values.shape)
        self.real_parameters.append(noise)

        # Create an object with links to the model and time series
        self.problem = pints.SingleSeriesProblem(
            self.model, self.times, self.values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        self.prior = pints.UniformPrior(
            [0.01, 400, noise*0.1],
            [0.02, 600, noise*100]
            )

        # Create a Bayesian log-likelihood (prior * likelihood)
        self.log_likelihood = pints.BayesianLogLikelihood(self.prior,
            pints.UnknownNoiseLogLikelihood(self.problem))

        # Select initial point and covariance
        self.x0 = np.array(self.real_parameters) * 1.1
        self.sigma0 = [0.005, 100, 0.5*noise]

    def test_settings(self):
        
        mcmc = pints.AdaptiveCovarianceMCMC(self.log_likelihood, self.x0)
        
        r = mcmc.acceptance_rate() * 0.5
        mcmc.set_acceptance_rate(r)
        self.assertEqual(mcmc.acceptance_rate(), r)

        i = int(mcmc.iterations() * 0.5)        
        mcmc.set_iterations(i)
        self.assertEqual(mcmc.iterations(), i)

        i = int(mcmc.non_adaptive_iterations() * 0.5)        
        mcmc.set_non_adaptive_iterations(i)
        self.assertEqual(mcmc.non_adaptive_iterations(), i)

        i = int(mcmc.burn_in() * 0.5)        
        mcmc.set_burn_in(i)
        self.assertEqual(mcmc.burn_in(), i)

        # Store only every 4th sample
        r = int(mcmc.thinning_rate() * 0.5)
        mcmc.set_thinning_rate(r)
        self.assertEqual(mcmc.thinning_rate(), r)

        # Disable verbose mode
        v = not mcmc.verbose()
        mcmc.set_verbose(v)
        self.assertEqual(mcmc.verbose(), v)

    def test_with_hint_and_sigma(self):
    
        mcmc = pints.AdaptiveCovarianceMCMC(self.log_likelihood, self.x0,
            self.sigma0)
        mcmc.set_verbose(debug)
        chain = mcmc.run()
        mean = np.mean(chain, axis=0)
        self.assertTrue(np.linalg.norm(mean - self.real_parameters) < 1.5)
        
        
        
if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()

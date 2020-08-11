#!/usr/bin/env python3
#
# Tests the basic methods of the Neal Langevin MCMC routine.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy

from shared import StreamCapture


class TestNealLangevinMCMC(unittest.TestCase):
    """
    Tests the basic methods of the NealLangevin MCMC routine.
    """

    def test_method(self):

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])

        # Create mcmc
        x0 = np.array([2, 2])
        sigma = [[3, 0], [0, 3]]
        mcmc = pints.NealLangevinMCMC(x0, sigma)

        # This method needs sensitivities
        self.assertTrue(mcmc.needs_sensitivities())

        # Set number of steps
        number_steps = 1000

        # Perform short run
        chain = []
        for i in range(number_steps):
            x = mcmc.ask()
            fx, gr = log_pdf.evaluateS1(x)
            sample = mcmc.tell((fx, gr))
            if i >= 0.5 * number_steps and sample is not None:
                chain.append(sample)
            if np.all(sample == x):
                self.assertEqual(mcmc.current_log_pdf(), fx)

        chain = np.array(chain)
        self.assertEqual(chain.shape[0], 0.5 * number_steps)
        self.assertEqual(chain.shape[1], len(x0))

        # Perform short run with negative steps

        # Create mcmc
        x0 = np.array([2, 2])
        sigma = [[3, 0], [0, 3]]
        mcmc = pints.NealLangevinMCMC(x0, sigma)

        # set delta
        mcmc.set_delta(mean=-0.1)

        # run
        chain = []
        for i in range(number_steps):
            x = mcmc.ask()
            fx, gr = log_pdf.evaluateS1(x)
            sample = mcmc.tell((fx, gr))
            if i >= 0.5 * number_steps and sample is not None:
                chain.append(sample)
            if np.all(sample == x):
                self.assertEqual(mcmc.current_log_pdf(), fx)

        chain = np.array(chain)
        self.assertEqual(chain.shape[0], 0.5 * number_steps)
        self.assertEqual(chain.shape[1], len(x0))

    def test_logging(self):
        # Test logging includes name and custom fields.

        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = [np.array([2, 2]), np.array([8, 8])]

        mcmc = pints.MCMCController(
            log_pdf, 2, x0, method=pints.NealLangevinMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()

        self.assertIn('Neal Langevin MCMC', text)
        self.assertIn(' Accept.', text)

    def test_flow(self):

        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = np.array([2, 2])

        # Test initial proposal is first point
        mcmc = pints.NealLangevinMCMC(x0)
        self.assertTrue(np.all(mcmc.ask() == mcmc._x0))

        # Repeated asks
        self.assertRaises(RuntimeError, mcmc.ask)

        # Tell without ask
        mcmc = pints.NealLangevinMCMC(x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0)

        # Repeated tells should fail
        x = mcmc.ask()
        mcmc.tell(log_pdf.evaluateS1(x))
        self.assertRaises(RuntimeError, mcmc.tell, log_pdf.evaluateS1(x))

        # Bad starting point
        mcmc = pints.NealLangevinMCMC(x0)
        mcmc.ask()
        self.assertRaises(
            ValueError, mcmc.tell, (float('-inf'), np.array([1, 1])))

    def test_set_hyper_parameters(self):
        # Tests the parameter interface for this sampler.

        x0 = np.array([2, 2])
        mcmc = pints.NealLangevinMCMC(x0)

        # Test default alpha
        default_alpha = 0.9
        self.assertEqual(mcmc.alpha(), default_alpha)

        # Test setting alpha
        alpha = 0.1
        mcmc.set_alpha(alpha)
        self.assertEqual(mcmc.alpha(), alpha)

        # Test setting invalid alpha: alpha > 1
        self.assertRaisesRegex(
            ValueError,
            r'Alpha must lie in the interval \[0\,1\]\.',
            mcmc.set_alpha,
            2
        )

        # Test setting invalid alpha: alpha < 0
        self.assertRaisesRegex(
            ValueError,
            r'Alpha must lie in the interval \[0\,1\]\.',
            mcmc.set_alpha,
            -1
        )

        # Test leapfrog parameters
        d = mcmc.leapfrog_step_size()
        self.assertTrue(len(d) == mcmc._n_parameters)

        mcmc.set_leapfrog_step_size(0.5)
        self.assertEqual(mcmc.leapfrog_step_size()[0], 0.5)
        self.assertRaises(ValueError, mcmc.set_leapfrog_step_size, -1)

        mcmc.set_epsilon(0.4)
        self.assertEqual(mcmc.epsilon(), 0.4)
        self.assertRaises(ValueError, mcmc.set_epsilon, -0.1)
        mcmc.set_leapfrog_step_size(1)
        self.assertEqual(len(mcmc.scaled_epsilon()), 2)
        self.assertEqual(mcmc.scaled_epsilon()[0], 0.4)
        self.assertEqual(len(mcmc.divergent_iterations()), 0)
        self.assertRaises(ValueError, mcmc.set_leapfrog_step_size, [1, 2, 3])

        mcmc.set_leapfrog_step_size([1.5, 3])
        self.assertEqual(mcmc.leapfrog_step_size()[0], 1.5)
        self.assertEqual(mcmc.leapfrog_step_size()[1], 3)

        # Test u updating parameters

        # Default values
        mean, std = mcmc.delta()
        self.assertEqual(mean, 0.05)
        self.assertAlmostEqual(std, 0.005)

        # Check setting values
        mcmc.set_delta(mean=1)
        mean, std = mcmc.delta()
        self.assertEqual(mean, 1)
        self.assertEqual(std, None)

        mcmc.set_delta(mean=1, sigma=0)
        mean, std = mcmc.delta()
        self.assertEqual(mean, 1)
        self.assertEqual(std, None)

        mcmc.set_delta(mean=-0.5, sigma=0.1)
        mean, std = mcmc.delta()
        self.assertEqual(mean, -0.5)
        self.assertEqual(std, 0.1)

        # Check invalid sigmas
        self.assertRaisesRegex(
            ValueError,
            r'The standard deviation of delta can only take non-negative'
            r' values\.',
            mcmc.set_delta,
            1,
            -1
        )

        # Test hyper parameters
        self.assertEqual(mcmc.n_hyper_parameters(), 4)

        mcmc.set_hyper_parameters([0.5, 2, 3])
        mean, std = mcmc.delta()
        self.assertEqual(mcmc.alpha(), 0.5)
        self.assertEqual(mcmc.leapfrog_step_size()[0], 2)
        self.assertEqual(mean, 3)
        self.assertEqual(std, None)

        mcmc.set_hyper_parameters([0.5, 2, 3, 4])
        mean, std = mcmc.delta()
        self.assertEqual(mcmc.alpha(), 0.5)
        self.assertEqual(mcmc.leapfrog_step_size()[0], 2)
        self.assertEqual(mean, 3)
        self.assertEqual(std, 4)

    def test_other_setters(self):
        # Tests other setters and getters.
        x0 = np.array([2, 2])
        mcmc = pints.NealLangevinMCMC(x0)
        self.assertRaises(ValueError, mcmc.set_hamiltonian_threshold, -0.3)
        threshold1 = mcmc.hamiltonian_threshold()
        self.assertEqual(threshold1, 10**3)
        threshold2 = 10
        mcmc.set_hamiltonian_threshold(threshold2)
        self.assertEqual(mcmc.hamiltonian_threshold(), threshold2)


if __name__ == '__main__':
    unittest.main()

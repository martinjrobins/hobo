#!/usr/bin/env python3
#
# Tests the basic methods of the Relativistic MCMC routine.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np
from scipy.integrate import cumtrapz, quad
from scipy.interpolate import interp1d
import pints
import pints.toy

from shared import StreamCapture


class TestRelativisticMCMC(unittest.TestCase):
    """
    Tests the basic methods of the Relativistic MCMC routine.
    """

    def test_method(self):

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])

        # Create mcmc
        x0 = np.array([2, 2])
        sigma = [[3, 0], [0, 3]]
        mcmc = pints.RelativisticMCMC(x0, sigma)

        # This method needs sensitivities
        self.assertTrue(mcmc.needs_sensitivities())

        # Set number of leapfrog steps
        ifrog = 10
        mcmc.set_leapfrog_steps(ifrog)

        # Perform short run
        chain = []
        for i in range(100 * ifrog):
            x = mcmc.ask()
            fx, gr = log_pdf.evaluateS1(x)
            reply = mcmc.tell((fx, gr))
            if reply is not None:
                y, fy, ac = reply
                if i >= 50 * ifrog:
                    chain.append(y)
                self.assertTrue(isinstance(ac, bool))
                if ac:
                    self.assertTrue(np.all(x == y))
                    self.assertEqual(fx, fy[0])
                    self.assertTrue(np.all(gr == fy[1]))

        chain = np.array(chain)
        self.assertEqual(chain.shape[0], 50)
        self.assertEqual(chain.shape[1], len(x0))

    def test_logging(self):
        # Test logging includes name and custom fields.

        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = [np.array([2, 2]), np.array([8, 8])]

        mcmc = pints.MCMCController(
            log_pdf, 2, x0, method=pints.RelativisticMCMC)
        mcmc.set_max_iterations(5)
        with StreamCapture() as c:
            mcmc.run()
        text = c.text()

        self.assertIn('Relativistic MCMC', text)
        self.assertIn(' Accept.', text)

    def test_flow(self):

        log_pdf = pints.toy.GaussianLogPDF([5, 5], [[4, 1], [1, 3]])
        x0 = np.array([2, 2])

        # Test initial proposal is first point
        mcmc = pints.RelativisticMCMC(x0)
        self.assertTrue(np.all(mcmc.ask() == mcmc._x0))

        # Repeated asks
        self.assertRaises(RuntimeError, mcmc.ask)

        # Tell without ask
        mcmc = pints.RelativisticMCMC(x0)
        self.assertRaises(RuntimeError, mcmc.tell, 0)

        # Repeated tells should fail
        x = mcmc.ask()
        mcmc.tell(log_pdf.evaluateS1(x))
        self.assertRaises(RuntimeError, mcmc.tell, log_pdf.evaluateS1(x))

        # Bad starting point
        mcmc = pints.RelativisticMCMC(x0)
        mcmc.ask()
        self.assertRaises(
            ValueError, mcmc.tell, (float('-inf'), np.array([1, 1])))

    def test_kinetic_energy(self):
        # Tests kinetic energy values

        x0 = np.array([2, 2])
        model = pints.RelativisticMCMC(x0)
        model.ask()

        # kinetic energy
        mc2 = 100.0
        m2c2 = 100.0
        momentum = [1.0, 2.0]
        squared = np.sum(np.array(momentum)**2)
        ke1 = mc2 * (squared / m2c2 + 1.0)**0.5
        ke2 = model._kinetic_energy(momentum)
        self.assertEqual(ke1, ke2)

        c = 1.0
        m = 1.0
        mc2 = m * c**2
        m2c2 = m**2 * c**2
        squared = np.sum(np.array(momentum)**2)
        ke1 = mc2 * (squared / m2c2 + 1.0)**0.5
        model = pints.RelativisticMCMC(x0)
        model.set_speed_of_light(c)
        model.ask()
        ke2 = model._kinetic_energy(momentum)
        self.assertEqual(ke1, ke2)

    def test_set_hyper_parameters(self):
        # Tests the parameter interface for this sampler.

        x0 = np.array([2, 2])
        mcmc = pints.RelativisticMCMC(x0)

        # Test leapfrog parameters
        n = mcmc.leapfrog_steps()
        d = mcmc.leapfrog_step_size()
        self.assertIsInstance(n, int)
        self.assertTrue(len(d) == mcmc._n_parameters)

        mcmc.set_leapfrog_steps(n + 1)
        self.assertEqual(mcmc.leapfrog_steps(), n + 1)
        self.assertRaises(ValueError, mcmc.set_leapfrog_steps, 0)

        mcmc.set_leapfrog_step_size(0.5)
        self.assertEqual(mcmc.leapfrog_step_size()[0], 0.5)
        self.assertRaises(ValueError, mcmc.set_leapfrog_step_size, -1)

        self.assertEqual(mcmc.n_hyper_parameters(), 4)
        mcmc.set_hyper_parameters([n + 2, 2, 0.4, 2.3])
        self.assertEqual(mcmc.leapfrog_steps(), n + 2)
        self.assertEqual(mcmc.leapfrog_step_size()[0], 2)
        self.assertEqual(mcmc.mass(), 0.4)
        self.assertEqual(mcmc.speed_of_light(), 2.3)

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

        c = 3.5
        mcmc.set_speed_of_light(c)
        self.assertEqual(mcmc.speed_of_light(), c)
        self.assertRaises(ValueError, mcmc.set_speed_of_light, -0.1)

        m = 2.9
        mcmc.set_mass(m)
        self.assertEqual(mcmc.mass(), m)
        self.assertRaises(ValueError, mcmc.set_mass, -1.8)
        self.assertRaises(ValueError, mcmc.set_mass, [1, 3])

    def test_other_setters(self):
        # Tests other setters and getters.

        x0 = np.array([2, 2])
        mcmc = pints.RelativisticMCMC(x0)
        self.assertRaises(ValueError, mcmc.set_hamiltonian_threshold, -0.3)
        threshold1 = mcmc.hamiltonian_threshold()
        self.assertEqual(threshold1, 10**3)
        threshold2 = 10
        mcmc.set_hamiltonian_threshold(threshold2)
        self.assertEqual(mcmc.hamiltonian_threshold(), threshold2)

    def test_momentum_logpdf(self):
        # Test log pdf of momentum magnitude

        x0 = np.array([2, 2, 2, 2, 2])
        model = pints.RelativisticMCMC(x0)
        m = 1.6
        c = 3.4
        n = len(x0)
        model.set_mass(m)
        model.set_speed_of_light(c)
        model.ask()

        mag = 1.7
        f1 = model._momentum_logpdf(mag)
        f2 = np.exp(-m * c**2 * np.sqrt(mag ** 2 / (m**2 * c**2) + 1)) \
            * mag ** (n - 1)

        self.assertAlmostEqual(f1, np.log(f2))

    def test_calculate_momentum_distribution(self):
        # Test calculation of inv cdf of momentum magnitude
        x0 = np.array([2, 2, 2, 2, 2])
        model = pints.RelativisticMCMC(x0)

        # Choose values of m and c for which the distribution is known to be
        # calculable without using logarithms
        m = 1.6
        c = 3.4
        model.set_mass(m)
        model.set_speed_of_light(c)
        model.ask()

        # Integrate the pdf to get normalizing constant
        def pdf(u):
            return np.exp(model._momentum_logpdf(u))

        c = quad(pdf, 0, 100)[0]

        # Integrate to get cumulative distribution function
        integration_grid = np.arange(1e-6, 10, 1e-5)
        cdf = cumtrapz(1 / c * pdf(integration_grid), x=integration_grid)

        # Interpolate to get approximate inverse
        inv_cdf = interp1d([0.0] + list(cdf), integration_grid)

        # Compare outputs of inverse CDF at selected points
        model_inv_cdf = model._inv_cdf
        test_points = [0.0, 0.1, 0.5, 0.75, 0.9]

        for test_point in test_points:
            self.assertAlmostEqual(
                inv_cdf(test_point), model_inv_cdf(test_point), places=5)

    def test_sample_momentum(self):
        # Test sampler of momentum
        x0 = np.array([2, 2, 2, 2, 2])
        model = pints.RelativisticMCMC(x0)
        model.ask()
        p = model._sample_momentum()
        self.assertEqual(len(p), len(x0))

        # Test sampler with a small value of m, triggering a refinement of the
        # integration grid
        m = 0.01
        c = 10.0
        model = pints.RelativisticMCMC(x0)
        model.set_mass(m)
        model.set_speed_of_light(c)
        model.ask()
        p = model._sample_momentum()
        self.assertEqual(len(p), len(x0))

        # Test sampler with unpleasant values of m and c that will trigger the
        # sampler warning
        m = 1e-10
        c = 1e-10
        model = pints.RelativisticMCMC(x0)
        model.set_mass(m)
        model.set_speed_of_light(c)
        model.ask()
        p = model._sample_momentum()
        self.assertEqual(len(p), len(x0))


if __name__ == '__main__':
    unittest.main()

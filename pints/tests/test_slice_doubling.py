#!/usr/bin/env python3
#
# Tests the basic methods of the Slice Sampling with Doubling routine.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#

import unittest
import numpy as np

import pints
import pints.toy

from shared import StreamCapture

debug = False


class TestSliceDoubling(unittest.TestCase):
    """
    Tests the basic methods of the Slice Sampling with Doubling routine.

    Please refer to the _slice_doubling.py script in ..\_mcmc\_slice_doubling.py
    """

    def test_initialisation(self):
        """
        Tests whether all instance attributes are initialised correctly.
        """
        # Create mcmc
        x0 = np.array([2, 4])
        mcmc = pints.SliceDoublingMCMC(x0)

        # Test attributes initialisation
        self.assertFalse(mcmc._running)
        self.assertFalse(mcmc._ready_for_tell)
        self.assertFalse(mcmc._first_expansion)
        self.assertFalse(mcmc._interval_found)
        self.assertFalse(mcmc._d)
        self.assertFalse(mcmc._init_check)
        self.assertFalse(mcmc._continue_check)

        self.assertEqual(mcmc._current, None)
        self.assertEqual(mcmc._current_log_pdf, None)
        self.assertEqual(mcmc._current_log_y, None)
        self.assertEqual(mcmc._proposed, None)
        self.assertEqual(mcmc._l, None)
        self.assertEqual(mcmc._r, None)
        self.assertEqual(mcmc._temp_l, None)
        self.assertEqual(mcmc._temp_r, None)
        self.assertEqual(mcmc._l_hat, None)
        self.assertEqual(mcmc._r_hat, None)
        self.assertEqual(mcmc._temp_l_hat, None)
        self.assertEqual(mcmc._temp_r_hat, None)
        self.assertEqual(mcmc._fx_l, None)
        self.assertEqual(mcmc._fx_r, None)

        self.assertEqual(mcmc._w, 1)
        self.assertEqual(mcmc._p, 10)
        self.assertEqual(mcmc._k, 0)
        self.assertEqual(mcmc._i, 0)
        self.assertEqual(mcmc._mcmc_iteration, 0)


    def test_first_run(self):
        """
        Tests the very first run of the sampler. 
        """
        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceDoublingMCMC(x0)

        # Ask should fail if _ready_for_tell flag is True
        with self.assertRaises(RuntimeError):
            mcmc._ready_for_tell = True
            mcmc.ask()

        # Undo changes
        mcmc._ready_for_tell = False

        # Check whether _running flag becomes True when ask() is called
        # Check whether first iteration of ask() returns x0
        self.assertFalse(mcmc._running)
        self.assertTrue(np.all(mcmc.ask() == x0))
        self.assertTrue(mcmc._running)
        self.assertTrue(mcmc._ready_for_tell)

        # Tell should fail when log pdf of x0 is infinite
        with self.assertRaises(ValueError):
            fx = np.inf
            mcmc.tell(fx)

        # Calculate log pdf for x0
        fx = log_pdf.evaluateS1(x0)[0]

        # Tell should fail when _ready_for_tell is False
        with self.assertRaises(RuntimeError):
            mcmc._ready_for_tell = False
            mcmc.tell(fx)

        # Undo changes
        mcmc._ready_for_tell = True

        # Test first iteration of tell(). The first point in the chain should be x0
        self.assertTrue(np.all(mcmc.tell(fx) == x0))

        # We update the current sample
        self.assertTrue(np.all(mcmc._current == x0))
        self.assertTrue(np.all(mcmc._current == mcmc._proposed))

        # We update the _current_log_pdf value used to generate the new slice
        self.assertEqual(mcmc._current_log_pdf, fx)

        # Check that the new slice has been constructed appropriately 
        self.assertTrue(mcmc._current_log_y == (mcmc._current_log_pdf - mcmc._e))
        self.assertTrue(mcmc._current_log_y < mcmc._current_log_pdf)

        # Check flag
        self.assertTrue(mcmc._first_expansion)

    def test_cycle(self):
        """
        Tests every step of a single MCMC iteration.
        """
        # Set seed for monitoring
        np.random.seed(1)

        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([2., 4.])
        mcmc = pints.SliceDoublingMCMC(x0)

        # VERY FIRST RUN
        x = mcmc.ask()
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertTrue(np.all(sample == x0))
        self.assertEqual(mcmc._fx_l, None)
        self.assertEqual(mcmc._fx_r, None)

        ##################################
        ### FIRST PARAMETER  - INDEX 0 ###
        ##################################

        # FIRST RUN: create initial interval edges
        x = mcmc.ask()

        # Check that interval edges are initialised appropriately
        self.assertTrue(x[0][0] < mcmc._current[0])
        self.assertTrue(x[1][0] > mcmc._current[0])

        # Check that interval I expansion steps are correct
        self.assertEqual(mcmc._k, 10)

        # We calculate the log pdf of the interval edges and since they are
        # within the slice, we return None 
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(sample, None)
        self.assertTrue(mcmc._fx_l, fx[0])
        self.assertTrue(mcmc._fx_r, fx[1])

        # SECOND RUN: begin expanding the interval
        x = mcmc.ask()

        # v < .5, therefore we expand the left edge
        self.assertEqual(x[0][0], mcmc._l)

        # Check that we are still within the slice or that k > 0
        self.assertTrue(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
        
        # The edges are inside the slice: return None and update the log pdf of the left edge
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(sample, None)
        self.assertEqual(fx[0], mcmc._fx_l)

        # SUBSEQUENT EXPANSIONs: expand left edge n-1 times and on the nth iteration we expand the right edge
        while mcmc._v < .5 or mcmc._k == 0:
            x = mcmc.ask()
            self.assertEqual(x[0][0], mcmc._l)
            self.assertTrue(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
            self.assertEqual(sample, None)
            self.assertEqual(fx[0], mcmc._fx_l)

        # COMPLETE INTERVAL EXPANSION: Check whether the edges are now outside the slice
        self.assertFalse(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
           
        # PROPOSE PARAMETER: now that we have estimated the interval, we sample a new parameter,
        # set the interval_found, _init_check and _continue_check flags to True 
        x = mcmc.ask()
        self.assertTrue(mcmc._interval_found)
        self.assertTrue(mcmc._init_check)
        self.assertTrue(mcmc._continue_check)

        # The log pdf of the proposed point is smaller than the slice height. The ``Threshold Check``
        # was not passed, so we reject, shrink and set the _init_check and _continue_check flags to 
        # False
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(sample, None)
        self.assertEqual(mcmc._l, mcmc._proposed[mcmc._i])
        self.assertFalse(mcmc._init_check)
        self.assertFalse(mcmc._continue_check)


        # TRY NEW PROPOSALS: Stop when a point is proposed within the slice
        while mcmc._current_log_y >= fx:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
        self.assertTrue(mcmc._current_log_y <= fx)


        # START ACCEPTANCE CHECK: Since the new proposed point passed the threshold test (fx > log_y),
        # we mantain the flags for proceeding with the ``Acceptance Check``
        self.assertTrue(mcmc._init_check)
        self.assertTrue(mcmc._continue_check)     
        x = mcmc.ask()

        # Since the intervals generated from the new point do not differ from the ones generated 
        # by the current sample, the following condition should be false, therefore d should remain
        # False
        self.assertFalse((mcmc._current[mcmc._i] < mcmc._m and mcmc._proposed[mcmc._i] >= mcmc._m) or (mcmc._current[mcmc._i] >= mcmc._m and mcmc._proposed[mcmc._i] < mcmc._m))
        self.assertFalse(mcmc._d)

        # Check whether the edges of the acceptance interval ``A=(l_hat, r_hat)`` have been initialised correctly
        self.assertTrue(np.all(x[0] == mcmc._temp_l_hat))
        self.assertTrue(np.all(x[1] == mcmc._temp_r_hat))
        self.assertFalse(mcmc._init_check)
        self.assertFalse(mcmc._d)

        # We proceed with the ``Acceptance Check``
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
 
        # The rejection condition in the ``Acceptance Check`` procedure should be False
        self.assertFalse(mcmc._d == True and mcmc._current_log_y >= fx[0] and mcmc._current_log_y >= fx[1])

        # Since the point hasn't been rejected, we will continue the ``Acceptance Check`` process
        while (mcmc._r_hat - mcmc._l_hat) > 1.1 * mcmc._w:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
 
        # The loop ends once the interval is smaller than ``1.1*w```
        self.assertFalse((mcmc._r_hat - mcmc._l_hat) > 1.1 * mcmc._w)

        # Since the ``Acceptance Check`` is finished, the _continue_check has been set to False
        # and we accepted the point
        x = mcmc.ask()
        self.assertFalse(mcmc._continue_check)
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)

        # As we have accepted the new point, we reset the interval expansion flags
        self.assertTrue(mcmc._first_expansion)
        self.assertFalse(mcmc._interval_found)

        # We increase the index _i to 1 to move to the next parameter to update
        self.assertEqual(mcmc._i, 1)

        ##################################
        ### SECOND PARAMETER - INDEX 1 ###
        ##################################

        # FIRST RUN: create initial interval edges
        x = mcmc.ask()

        # Check that interval edges are initialised appropriately
        self.assertTrue(x[0][1] < mcmc._current[1])
        self.assertTrue(x[1][1] > mcmc._current[1])
        self.assertEqual(mcmc._k, 10)

        # We calculate the log pdf of the interval edges and since they are
        # within the slice, we return None 
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(sample, None)
        self.assertTrue(mcmc._fx_l, fx[0])
        self.assertTrue(mcmc._fx_r, fx[1])

        # SECOND RUN: begin expanding the interval
        x = mcmc.ask()

        # v < .5, therefore we expand the left edge
        self.assertEqual(x[0][1], mcmc._l)

        # Check that we are still within the slice or that k > 0
        self.assertTrue(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
        
        # The edges are inside the slice: return None and update the log pdf of the left edge
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertEqual(sample, None)
        self.assertEqual(fx[0], mcmc._fx_l)

        # SUBSEQUENT EXPANSION: expand left edge n-1 times and on the nth iteration we expand the right edge
        while mcmc._v < .5 or mcmc._k == 0:
            x = mcmc.ask()
            self.assertEqual(x[0][1], mcmc._l)
            self.assertTrue(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
            self.assertEqual(sample, None)
            self.assertEqual(fx[0], mcmc._fx_l)

        # COMPLETE INTERVAL EXPANSION: Check whether the edges are now outside the slice or if we
        # have reached max number of expansion steps
        self.assertFalse(mcmc._k > 0 and (mcmc._current_log_y < mcmc._fx_l or mcmc._current_log_y < mcmc._fx_r))
        
        # PROPOSE PARAMETER: now that we have estimated the interval, we sample a new parameter,
        # set the interval_found, _init_check and _continue_check flags to True 
        x = mcmc.ask()
        self.assertTrue(mcmc._interval_found)
        self.assertTrue(mcmc._init_check)
        self.assertTrue(mcmc._continue_check)

        # The log pdf of the proposed point is greater than the slice height.
        # The point has passed the "Threshold Check"
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
        self.assertTrue(fx > mcmc._current_log_y)

        # START ACCEPTANCE TEST: Since the new proposed point passed the threshold test (fx > log_y),
        # we mantain the flags for proceeding with the ``Acceptance Check``
        self.assertTrue(mcmc._init_check)
        self.assertTrue(mcmc._continue_check)   
        x = mcmc.ask()

        # Since the intervals generated from the new point do not differ from the ones generated 
        # by the current sample, the following condition should be false, therefore d should remain
        # False
        self.assertFalse((mcmc._current[mcmc._i] < mcmc._m and mcmc._proposed[mcmc._i] >= mcmc._m) or (mcmc._current[mcmc._i] >= mcmc._m and mcmc._proposed[mcmc._i] < mcmc._m))

        # Check whether the edges of the acceptance interval ''A=(l_hat, r_hat)'' have been initialised correctly
        self.assertTrue(np.all(x[0] == mcmc._temp_l_hat))
        self.assertTrue(np.all(x[1] == mcmc._temp_r_hat))
        self.assertFalse(mcmc._init_check)
        self.assertFalse(mcmc._d)

        # We proceed with the ``Acceptance Check``
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)
 
        # The rejection condition in the Acceptance Check procedure should be False
        self.assertFalse(mcmc._d == True and mcmc._current_log_y >= fx[0] and mcmc._current_log_y >= fx[1])

        # Since the point hasn't been rejected, we will continue the ``Acceptance Check`` process
        while (mcmc._r_hat - mcmc._l_hat) > 1.1 * mcmc._w:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
 
        # The loop ends once the interval is smaller than ``1.1*w``
        self.assertFalse((mcmc._r_hat - mcmc._l_hat) > 1.1 * mcmc._w)
        self.assertFalse((mcmc._d == True and mcmc._current_log_y >= fx[0] and mcmc._current_log_y >= fx[1]))

        # Since the Acceptance check is finished, the _continue_check has been set to False
        # and we accepted the point
        x = mcmc.ask()
        self.assertFalse(mcmc._continue_check)
        fx = log_pdf.evaluateS1(x)[0]
        sample = mcmc.tell(fx)

        # As we have accepted the new point, we reset the interval expansion flags
        self.assertTrue(mcmc._first_expansion)
        self.assertFalse(mcmc._interval_found)

        # All the parameters of the sample have been updates, so we reset the index to 0 
        self.assertEqual(mcmc._i, 0)

        # Now that we have generated the new sample, we set this to be the current sample
        self.assertTrue(np.all(mcmc._current == mcmc._proposed))

        # We generate a new log_y for the height of the new slice
        self.assertEqual(fx, mcmc._current_log_pdf)

        # Check whether the new slice has been generated correctly
        self.assertEqual(mcmc._current_log_y, mcmc._current_log_pdf - mcmc._e)


    def test_complete_run(self):
        """
        Test multiple MCMC iterations of the sample
        """
        # Create log pdf
        log_pdf = pints.toy.GaussianLogPDF([2, 4], [[1, 0], [0, 3]])

        # Create mcmc
        x0 = np.array([1,1])
        mcmc = pints.SliceDoublingMCMC(x0)

        # Run multiple iterations of the sampler
        chain = []
        while mcmc._mcmc_iteration < 10000:
            x = mcmc.ask()
            fx = log_pdf.evaluateS1(x)[0]
            sample = mcmc.tell(fx)
            if sample is not None:
                chain.append(sample)

        # Fit Multivariate Gaussian to chain samples
        mean = np.mean(chain, axis=0)
        cov = np.cov(chain, rowvar=0)

        #print(mean) [1.99 3.98]
        #print(cov)  [[0.99, 0.00][0.00, 3.05]]

if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
    
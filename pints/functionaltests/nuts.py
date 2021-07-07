#!/usr/bin/env python3
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

from __future__ import division

import pints

from ._problems import (RunMcmcMethodOnTwoDimGaussian,
                        RunMcmcMethodOnBanana,
                        RunMcmcMethodOnHighDimensionalGaussian)


def test_nuts_on_two_dim_gaussian(n_iterations=None):
    if n_iterations is None:
        n_iterations = 1000
    problem = RunMcmcMethodOnTwoDimGaussian(
        method=pints.NoUTurnMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=200
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def test_nuts_on_banana(n_iterations=None):
    if n_iterations is None:
        n_iterations = 2000
    problem = RunMcmcMethodOnBanana(
        method=pints.NoUTurnMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=500
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def test_nuts_on_high_dim_gaussian(n_iterations=None):
    if n_iterations is None:
        n_iterations = 4000
    problem = RunMcmcMethodOnHighDimensionalGaussian(
        method=pints.NoUTurnMCMC,
        n_chains=4,
        n_iterations=n_iterations,
        n_warmup=1000
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }

#
# Logistic model.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints


class LogisticModel(pints.ForwardModel):
    """
    Logistic model.

    Has two parameters: A growth rate ``r`` and a carrying capacity ``k``.
    """
    def __init__(self, initial_population_size=2):
        super(LogisticModel, self).__init__()
        self._p0 = float(initial_population_size)
        if self._p0 < 0:
            raise ValueError('Population size cannot be negative.')

    def dimension(self):
        return 2

    def simulate(self, parameters, times):
        r, k = [float(x) for x in parameters]
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')
        if self._p0 == 0:
            return np.zeros(times.shape)
        if k < 0:
            return np.zeros(times.shape)
        return k / (1 + (k / self._p0 - 1) * np.exp(-r * times))

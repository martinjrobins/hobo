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
from scipy.integrate import odeint


class FitzhughNagumoModel(pints.ForwardModel):
    """
    Fitzhugh Nagumo model of action potential.

    Has two states, and three phenomenological parameters: ``a`` , ``b``,
    ``c``. All states are visible

    .. math::
        \\frac{d \mathbf{y}}{dt} = \\mathbf{f}(\\mathbf{y},\\mathbf{p},t)

    where

    .. math::
        \\mathbf{y} &= (V,R)\\\\
        \\mathbf{p} &= (a,b,c)

    The RHS is given by

    .. math::
        \\mathbf{f}(\\mathbf{y},\\mathbf{p},t) &= \\left[\\begin{matrix}
                    c \\left(R - V^{3}/3+V\\right)\\\\
                    - \\frac{1}{c} \\left(R b + V - a\\right)\\end{matrix}
                    \\right]\\\\

    Arguments:

    ``y0``
        The system's initial state
    """

    def __init__(self, y0=None):
        super(FitzhughNagumoModel, self).__init__()

        # Check initial values
        if y0 is None:
            self._y0 = np.array([-1, 1], dtype=float)
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 2:
                raise ValueError('Initial value must have size 2.')

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters)`. """
        return 3

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.outputs()`. """
        return 2

    def simulate(self, parameters, times):

        # Unpack parameters
        a, b, c = [float(x) for x in parameters]

        # Check times
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        # Define RHS
        def r(y, t, p):
            V, R = y
            dV_dt = (V - V**3 / 3 + R) * c
            dR_dt = (V - a + b * R) / -c
            return dV_dt, dR_dt

        # Evaluate and return
        values = odeint(r, self._y0, times, (parameters,))
        return values


class FitzhughNagumoModelWithSensitivities(pints.ForwardModelS1):
    """
    Fitzhugh Nagumo model of action potential that provides first-order
    derivatives (sensitivities).

    Has two states, and three phenomenological parameters: ``a`` , ``b``,
    ``c``. All states are visible

    .. math::
        \\frac{d \mathbf{y}}{dt} = \\mathbf{f}(\\mathbf{y},\\mathbf{p},t)

    where

    .. math::
        \\mathbf{y} &= (V,R)\\\\
        \\mathbf{p} &= (a,b,c)

    The RHS, jacobian and change in RHS with the parameters are given by

    .. math::
        \\mathbf{f}(\\mathbf{y},\\mathbf{p},t) &= \\left[\\begin{matrix}
                    c \\left(R - V^{3}/3+V\\right)\\\\
                    - \\frac{1}{c} \\left(R b + V - a\\right)\\end{matrix}
                    \\right]\\\\
        \\frac{\partial \mathbf{f}}{\partial \mathbf{y}} &=
        \\left[\\begin{matrix} c \\left(1- V^{2}\\right) & c \\\\
                    - \\frac{1}{c} & - \\frac{b}{c}\\end{matrix}\\right] \\\\
        \\frac{\partial \mathbf{f}}{\partial \mathbf{p}} &=
                        \\left[\\begin{matrix}0 & 0 & R - V^{3}/3 + V\\\\
                        \\frac{1}{c} & - \\frac{R}{c} &
                        \\frac{1}{c^{2}} \\left(R b + V - a\\right)
                        \\end{matrix}\\right]

    Arguments:

    ``y0``
        The system's initial state
    """

    def __init__(self, y0=None):
        super(FitzhughNagumoModelWithSensitivities, self).__init__()

        # Check initial values
        if y0 is None:
            self._y0 = np.array([-1, 1], dtype=float)
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 2:
                raise ValueError('Initial value must have size 2.')

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters)`. """
        return 3

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.outputs()`. """
        return 2

    def simulate(self, parameters, times):

        # Unpack parameters
        a, b, c = [float(x) for x in parameters]

        # Check times
        times = np.asarray(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        # Define simple RHS
        def r(y, t, p):
            V, R = y
            dV_dt = (V - V**3 / 3 + R) * c
            dR_dt = (V - a + b * R) / -c
            return dV_dt, dR_dt

        # Define jacobian
        def jac(y):
            V, R = y
            ret = np.empty((2, 2))
            ret[0, 0] = c * (1 - V**2)
            ret[0, 1] = c
            ret[1, 0] = -1 / c
            ret[1, 1] = -b / c
            return ret

        # Define dfdp
        def dfdp(y):
            V, R = y
            ret = np.empty((2, 3))
            ret[0, 0] = 0
            ret[0, 1] = 0
            ret[0, 2] = R - V**3 / 3 + V
            ret[1, 0] = 1 / c
            ret[1, 1] = -R / c
            ret[1, 2] = (R * b + V - a) / c**2
            return ret

        # Define combined RHS
        def rhs(y_and_dydp, t, p):
            y = y_and_dydp[0:2]
            dydp = y_and_dydp[2:].reshape((2, 3))

            dydt = r(y, t, p)
            d_dydp_dt = np.matmul(jac(y), dydp) + dfdp(y)

            return np.concatenate((dydt, d_dydp_dt.reshape(-1)))

        # Calculate and return
        y0 = np.zeros(8)
        y0[0:2] = self._y0
        result = odeint(rhs, y0, times, (parameters,))
        values = result[:, 0:2]
        dvalues_dp = result[:, 2:].reshape((len(times), 2, 3))
        return values, dvalues_dp


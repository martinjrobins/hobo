#
# Beeler-Reuter model for mammalian ventricular action potential
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import print_function
import pints
import numpy as np
import scipy.integrate


class ActionPotentialModel(pints.ForwardModel):
    """
    The 1977 Beeler-Reuter model of the mammalian ventricular action potential
    (AP) [1].

    This model describes several ion currents, each with a maximum conductance
    parameter, that together give rise to the cardiac AP and calcium transient.
    In this (non-trivial) 'toy' model, we use the maximum conductances as the
    parameters, and the AP and calcium transient as observable outputs. All
    other model parameters are assumed to be known.

    The parameters are _scaled_: instead of passing in the conductances
    directly, users should provide the natural log of the maximum conductances.
    This makes the parameters easier to find for optimisation algorithms.

    References:

    [1] Reconstruction of the action potential of ventricular myocardial
    fibres
    Beeler, Reuter (1977) Journal of Physiology

    Arguments:

    ``y0``
        (Optional) The initial condition of the observables ``v`` and ``cai``,
        where ``cai >= 0``.

    *Extends:* :class:`pints.ForwardModel`.
    """
    def __init__(self, y0=None):
        if y0 is None:
            self.set_initial_conditions([-84.622, 2e-7])
        else:
            self.set_initial_conditions(y0)

        # Initial condition for non-observable states
        self._m0 = 0.01
        self._h0 = 0.99
        self._j0 = 0.98
        self._d0 = 0.003
        self._f0 = 0.99
        self._x10 = 0.0004
        # membrane capacitance, in uF/cm^2
        self._C_m = 1.0
        # Nernst reversal potentials, in mV
        self._E_Na = 50.0
        # Stimulus current
        self._I_Stim_amp = 25.0
        self._I_Stim_period = 1000.0
        self._I_Stim_length = 2.0

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        # 5 conductance values
        return 5

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        # membrane voltage and calcium concentration
        return 2

    def _rhs(self, states, time, parameters):
        """
        Right-hand side equation of the ode to solve.
        """
        # Set-up
        V, Cai, m, h, j, d, f, x1 = states
        gNaBar, gNaC, gCaBar, gK1Bar, gx1Bar = np.exp(parameters)
        # Equations
        # INa
        INa = (gNaBar * m**3 * h * j + gNaC) * (V - self._E_Na)
        alpha = (V + 47) / (1 - np.exp(-0.1 * (V + 47)))
        beta = 40 * np.exp(-0.056 * (V + 72))
        dmdt = alpha * (1 - m) - beta * m
        alpha = 0.126 * np.exp(-0.25 * (V + 77))
        beta = 1.7 / (1 + np.exp(-0.082 * (V + 22.5)))
        dhdt = alpha * (1 - h) - beta * h
        alpha = 0.055 * np.exp(-0.25 * (V + 78)) \
            / (1 + np.exp(-0.2 * (V + 78)))
        beta = 0.3 / (1 + np.exp(-0.1 * (V + 32)))
        djdt = alpha * (1 - j) - beta * j
        # ICa
        E_Ca = -82.3 - 13.0287 * np.log(Cai)
        ICa = gCaBar * d * f * (V - E_Ca)
        alpha = 0.095 * np.exp(-0.01 * (V + -5)) \
            / (np.exp(-0.072 * (V + -5)) + 1)
        beta = 0.07 * np.exp(-0.017 * (V + 44)) \
            / (np.exp(0.05 * (V + 44)) + 1)
        dddt = alpha * (1 - d) - beta * d
        alpha = 0.012 * np.exp(-0.008 * (V + 28)) \
            / (np.exp(0.15 * (V + 28)) + 1)
        beta = 0.0065 * np.exp(-0.02 * (V + 30)) \
            / (np.exp(-0.2 * (V + 30)) + 1)
        dfdt = alpha * (1 - f) - beta * f
        # Cai
        dCaidt = -1e-7 * ICa + 0.07 * (1e-7 - Cai)
        # IK1
        IK1 = gK1Bar * (
            4 * (np.exp(0.04 * (V + 85)) - 1)
            / (np.exp(0.08 * (V + 53)) + np.exp(0.04 * (V + 53)))
            + 0.2 * (V + 23)
            / (1 - np.exp(-0.04 * (V + 23)))
        )
        # IX1
        Ix1 = gx1Bar * x1 * (np.exp(0.04 * (V + 77)) - 1) \
            / np.exp(0.04 * (V + 35))
        alpha = 0.0005 * np.exp(0.083 * (V + 50)) \
            / (np.exp(0.057 * (V + 50)) + 1)
        beta = 0.0013 * np.exp(-0.06 * (V + 20)) \
            / (np.exp(-0.04 * (V + 333)) + 1)
        dx1dt = alpha * (1 - x1) - beta * x1
        # I_Stim
        if (time % self._I_Stim_period) < self._I_Stim_length:
            IStim = self._I_Stim_amp
        else:
            IStim = 0
        # V
        dVdt = -(1 / self._C_m) * (IK1 + Ix1 + INa + ICa - IStim)
        # Output
        output = np.array([dVdt,
                           dCaidt,
                           dmdt,
                           dhdt,
                           djdt,
                           dddt,
                           dfdt,
                           dx1dt])
        return output

    def set_initial_conditions(self, y0):
        """
        Changes the initial conditions for this model.
        """
        if y0[1] < 0:
            raise ValueError('Initial condition of ``cai`` cannot be'
                             ' negative.')
        self._v0 = y0[0]
        self._cai0 = y0[1]

    def initial_conditions(self):
        """
        Returns the initial conditions of this model.
        """
        return [self._v0, self._cai0]

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        y0 = [self._v0,
              self._cai0,
              self._m0,
              self._h0,
              self._j0,
              self._d0,
              self._f0,
              self._x10]
        solved_states = scipy.integrate.odeint(
            self._rhs, y0, times, args=(parameters,), hmax=self._I_Stim_length,
            rtol=1e-7, atol=1e-9)
        # Only return the observable (V, Cai)
        return solved_states[:, 0:2]

    def simulate_all_states(self, parameters, times):
        """
        Returns all state variables that ``simulate()`` does not return.
        """
        y0 = [self._v0,
              self._cai0,
              self._m0,
              self._h0,
              self._j0,
              self._d0,
              self._f0,
              self._x10]
        solved_states = scipy.integrate.odeint(
            self._rhs, y0, times, args=(parameters,), hmax=self._I_Stim_length)
        # Return all states
        return solved_states

    def suggested_parameters(self):
        """
        Returns a suggested array of parameter values.
        """
        # maximum conducances, in mS/cm^2
        g_Na = 4.0
        g_NaC = 0.003
        g_Ca = 0.09
        g_K1 = 0.35
        g_x1 = 0.8
        return np.log([g_Na, g_NaC, g_Ca, g_K1, g_x1])

    def suggested_times(self):
        """
        Returns a suggested set of sampling times.
        """
        return np.arange(0, 500, 0.1)


#
# Log-likelihood functions
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
import scipy.special


class KalmanFilterLogLikelihood(pints.ProblemLogLikelihood):
    """
    *Extends:* :class:`ProblemLogLikelihood`

The idea is (I think) would be to define the measurements to come from a base model m(p) with fixed parameters p (i.e. any pints model), plus a linear term with the varying parameters x, plus a normal noise term. That is, defined at time points k =1..N the measurements are:

z_k = m_k(p) + H_k x_k + v_k

that you would have a model for the varying parameters as

x_{k+1} = A_k * x_k + w_k

where x_k is the vector of varying parameters (i.e. states), A_k is a matrix defining how the states evolve over time, and w_k are samples from a multivariate normal distribution.

Given a set of fixed paramters p, everything else becomes linear you can use a kalman filter to calculate the likelihood https://en.wikipedia.org/wiki/Kalman_filter#Marginal_likelihood

The user would specify the base model m, the measurement matrix H_k, the transition matrix A_k, and the variances for v_k and w_k (or perhaps these could be unknowns).


    """
    def __init__(self, problem, measurement_matrix,measurement_sigma,
                                transition_matrix, transition_sigma):
        super(KalmanFilterLogLikelihood, self).__init__(problem)

        # Store counts
        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._nt = problem.n_times()

        self._H = measurement_matrix
        self._v = measurement_sigma

        self._A = transition_matrix
        self._w = transition_sigma

        # Check sigmas
        for sigma in [measurement_sigma,transition_sigma]:
            if np.isscalar(sigma):
                sigma = np.ones(self._no) * float(sigma)
            else:
                sigma = pints.vector(sigma)
                if len(sigma) != self._no:
                    raise ValueError(
                        'Sigma must be a scalar or a vector of length n_outputs.')
            if np.any(sigma <= 0):
                raise ValueError('Standard deviation must be greater than zero.')

        # Pre-calculate parts
        self._offset = -0.5 * self._nt * np.log(2 * np.pi)
        self._offset -= self._nt * np.log(sigma)
        self._multip = -1 / (2.0 * sigma**2)

        # Pre-calculate S1 parts
        self._isigma2 = sigma**-2

    def __call__(self, x):
        sim = self._problem.evaluate(x)
        x = x0
        P = ?
        H = self._H
        A = self._A
        log_like = 0.0
        for m, z in zip(self._problem.evaluate(x),self._values):
            # predict
            x = A.dot(x)
            P = np.matmul(A,np.matmul(P * A.T)) + Q # Q is transition covariance

            # update
            y = z - H.dot(x) - m
            S = R + np.matmul(H , np.matmul(P * H.T)) # R is measurement covariance
            invS = np.linalg.inv(S)
            K = np.matmul(P,np.matmul(H.T * invS))
            x += P.dot(H.T.dot(K.dot(y)))
            tmp = I - np.matmul(K,H)
            P = np.matmul(tmp,np.matmul(P ,tmp.T)) + np.matmul(K,np.matmul(R,K.T))
            # or P = np.matmul(tmp,P) # only valid for optimal gain?
            #postfit_residual = z - H.dot(x) - m

            log_like -= 0.5*(np.inner(y,invS.dot(y)) + np.linalg.slogdet(S)[1] + no*log(2*pi))


        error = self._values - self._problem.evaluate(x)
        return np.sum(self._offset + self._multip * np.sum(error**2, axis=0))

    def evaluateS1(self, x):
        """ See :meth:`LogPDF.evaluateS1()`. """


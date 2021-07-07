#
# Broyden-Fletcher-Goldfarb-Shanno algorithm
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from numpy.linalg import norm
import pints


class LBFGS(pints.LineSearchBasedOptimiser):
    """
    Broyden-Fletcher-Goldfarb-Shanno algorithm [2], [3], [4]

    The Hager-Zhang line search algorithm [1] is implemented in this class

    [1] Hager, W. W.; Zhang, H. Algorithm 851: CG_DESCENT,
    a Conjugate Gradient Method with Guaranteed Descent.
    ACM Trans. Math. Softw. 2006, 32 (1), 113-137.
    https://doi.org/10.1145/1132973.1132979.

    [2] Liu, D. C.; Nocedal, J.
    On the Limited Memory BFGS Method for Large Scale Optimization.
    Mathematical Programming 1989, 45 (1),
    503-528. https://doi.org/10.1007/BF01589116.

    [3] Nocedal, J. Updating Quasi-Newton Matrices with Limited Storage.
    Math. Comp. 1980, 35 (151), 773-782.
    https://doi.org/10.1090/S0025-5718-1980-0572855-7.

    [4] Nash, S. G.; Nocedal, J. A Numerical Study of the Limited Memory
    BFGS Method and the Truncated-Newton Method for Large Scale Optimization.
    SIAM J. Optim. 1991, 1 (3), 358-372. https://doi.org/10.1137/0801023.

    """

    def __init__(self, x0, sigma0=None, boundaries=None):
        super(LBFGS, self).__init__(x0, sigma0, boundaries)

        # maximum number of correction matrices stored
        self._m = 5

        # Storage for vectors constructing the correction matrix
        # this is the advised way of storing them.
        self._S = np.zeros(shape=(self._n_parameters, self._m))
        self._Y = np.zeros(shape=(self._n_parameters, self._m))

    def max_correction_matrice_storage(self):
        """
        Returns ``m``, the maximum number of correction matrice for
        calculating the inverse hessian used and stored
        """
        return self._m

    def name(self):
        """ See :meth:`Optimiser.name()`. """
        return 'Broyden-Fletcher-Goldfarb-Shanno (BFGS)'

    def __set_max_correction_matrice_storage(self, m):
        """
        Sets the maximum number of correction matrice to be stored and used,
        in subsequent inverse hessian updates, if ``m`` is set large enough
        for the problem this method becomes the BFGS rather than the L-BFGS.

        Parameters
        ----------
        m: int
        The maximum number of correction matrice for calculating the
        inverse hessian used and stored.
        """
        if(m == int(m)):
            self._m = m
            self._S = np.zeros(self._n_parameters, m)
            self._Y = np.zeros(self._n_parameters, m)
        else:
            print('Invalid value of m!!!\nm must be an integer')
            print('using default parameters: m = ', self._m)

    def inverse_hessian_update(self, proposed_f, proposed_dfdx):
        '''
        The inverse hessian matrix and newton direction are updated by the
        L-BFGS/BFGS approximation of the hessian described in reference [2]
        [3], and [4].
        '''

        # identity matrix
        I = np.identity(self._n_parameters)

        # We do this if we haven't exhausted existing memory yet, this is
        # identical to the BFGS algorithm
        if self._k <= self._m - 1:
            k = self._k
            # Defining the next column.
            self._S[:, k] = self._proposed - self._current
            self._Y[:, k] = proposed_dfdx - self._current_dfdx

            # Defining B_0. Scaling taken from [4].
            B = ((np.matmul(np.transpose(self._Y[:, k]),
                            self._S[:, k])
                  / (norm(self._Y[:, k], ord=2) ** 2)) * I)

            # Updating inverse hessian.
            for k in range(self._k + 1):

                V = (I - np.matmul(self._Y[:, k],
                                   np.transpose(self._S[:, k]))
                     / np.matmul(np.transpose(self._Y[:, k]),
                                 self._S[:, k]))

                B = np.matmul(np.transpose(V), np.matmul(self._B, V))
                B += (np.matmul(self._S[:, k],
                                np.transpose(self._S[:, k]))
                      / np.matmul(np.transpose(self._Y[:, k]),
                                  self._S[:, k]))

        # We have exhausted the limited memory and now enter
        # the LM-BFGS algorithm
        else:

            m = self._m - 1
            # Shifting everything one column to the left.
            self._S[:, 0:m] = self._S[:, 1:self._m]
            self._Y[:, 0:m] = self._Y[:, 1:self._m]

            # Renewing last column.
            self._S[:, m] = self._proposed - self._current
            self._Y[:, m] = proposed_dfdx - self._current_dfdx

            # Defining B_0. Scaling taken from [4].
            B = ((np.matmul(np.transpose(self._Y[:, m]),
                            self._S[:, m])
                  / (norm(self._Y[:, m], ord=2) ** 2)) * I)

            # Updating inverse hessian.
            for k in range(self._m):

                V = (I - np.matmul(self._Y[:, k],
                                   np.transpose(self._S[:, k]))
                     / np.matmul(np.transpose(self._Y[:, k]),
                                 self._S[:, k]))

                B = np.matmul(np.transpose(V), np.matmul(self._B, V))
                B += (np.matmul(self._S[:, k],
                                np.transpose(self._S[:, k]))
                      / np.matmul(np.transpose(self._Y[:, k]),
                                  self._S[:, k]))

                B = ((np.matmul(np.transpose(self._Y[:, m]),
                                self._S[:, m])
                      / (norm(self._Y[:, m], ord=2)**2)) * I)

        return B

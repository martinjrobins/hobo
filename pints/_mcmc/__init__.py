#
# Sub-module containing MCMC inference routines
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import division
import pints
import numpy as np

class MCMC(object):
    """
    Takes a :class:`LogLikelihood` function and returns a markov chain
    representative of its distribution.
    
    Arguments:
    
    ``function``
        A :class:`LogLikelihood` function that evaluates points in the
        parameter space.
    ``x0``
        An starting point in the parameter space.
    ``sigma0=None``
        An optional initial covariance matrix, i.e., a guess of the the
        covariance of the likelihood around ``x0``.

    """
    def __init__(self, log_likelihood, x0, sigma0=None):

        # Store function
        if not isinstance(log_likelihood, pints.LogLikelihood):
            raise ValueError('Given function must extend pints.LogLikelihood')
        self._log_likelihood = log_likelihood
        
        # Get dimension
        self._dimension = self._log_likelihood.dimension()
        
        # Check initial position
        self._x0 = pints.vector(x0)
        if len(self._x0) != self._dimension:
            raise ValueError('Initial position must have same dimension as'
                ' loglikelihood function.')
        
        # Check initial standard deviation
        if sigma0 is None:
            self._sigma0 = np.diag(0.01 * self._x0)
        else:
            self._sigma0 = np.array(sigma0)
            if np.product(self._sigma0.shape) == self._dimension:
                # Convert from 1d array
                self._sigma0 = self._sigma0.reshape((self._dimension,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                # Check if 2d matrix of correct size
                self._sigma0 = self._sigma0.reshape((self._dimension,
                    self._dimension))
            self._sigma0.setflags(write=False)
        
        # Print info to console
        self._verbose = verbose
        
    def run(self):
        """
        Runs the MCMC routine and returns a markov chain representing the
        distribution of the given log-likelihood function.
        """
        raise NotImplementedError

    def set_verbose(self, value):
        """
        Enables or disables verbose mode for this MCMC routine. In verbose
        mode, lots of output is generated during a run.
        """
        self._verbose = bool(value)
    
    def verbose(self):
        """
        Returns ``True`` if the MCMC routine is set to run in verbose mode.
        """
        return self._verbose



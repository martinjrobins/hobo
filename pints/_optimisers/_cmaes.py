#
# Uses the Python `cma` module to runs CMA-ES optimisations.
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class CMAES(pints.PopulationBasedOptimiser):
    """
    *Extends:* :class:`Optimiser`

    Finds the best parameters using the CMA-ES method described in [1, 2] and
    implemented in the `cma` module.

    CMA-ES stands for Covariance Matrix Adaptation Evolution Strategy, and is
    designed for non-linear derivative-free optimization problems.

    [1] https://www.lri.fr/~hansen/cmaesintro.html

    [2] Hansen, Mueller, Koumoutsakos (2006) Reducing the time complexity of
    the derandomized evolution strategy with covariance matrix adaptation
    (CMA-ES).

    """
    def __init__(self, x0, sigma0=None, boundaries=None):
        super(CMAES, self).__init__(x0, sigma0, boundaries)

        # Set initial state
        self._running = False
        self._ready_for_tell = False

        # Set default settings
        self.set_population_size()

        # Best solution found
        self._xbest = pints.vector(x0)
        self._fbest = float('inf')

    def ask(self):
        """See :meth:`Optimiser.ask()`."""
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Ready for tell now
        self._ready_for_tell = True

        # Create new samples
        self._xs = np.array(self._es.ask())

        # Set as read-only and return
        self._xs.setflags(write=False)
        return self._xs

    def fbest(self):
        """ See :meth:`Optimiser.fbest()`. """
        f = self._es.result.fbest
        return float('inf') if f is None else f

    def _initialise(self):
        """
        Initialises the optimiser for the first iteration.
        """
        if self._running:
            raise Exception('Already initialised.')

        #TODO
        # Import cma (may fail!)
        # Only the first time this is called in a running program incurs
        # much overhead.
        import cma

        # Get BestSolution in cma 1.x and 2.x
        # try:
        #    from cma import BestSolution
        # except ImportError:
        #    from cma.optimization_tools import BestSolution

        # Set up simulation
        options = cma.CMAOptions()

        # Set boundaries
        if self._boundaries is not None:
            options.set(
                'bounds',
                [list(self._boundaries._lower), list(self._boundaries._upper)]
            )

        # Set stopping criteria
        #options.set('maxiter', max_iter)
        #options.set('tolfun', min_significant_change)
        # options.set('ftarget', target)

        # Tell CMA not to worry about growing step sizes too much
        #options.set('tolfacupx', 10000)

        # CMA-ES wants a single standard deviation as input, use the smallest
        # in the vector (if the user passed in a scalar, this will be the
        # value used).
        self._sigma0 = np.min(self._sigma0)

        # Tell cma-es to be quiet
        options.set('verbose', -9)

        # Set population size
        options.set('popsize', self._population_size)

        # Search
        self._es = cma.CMAEvolutionStrategy(self._x0, self._sigma0, options)

        # Update optimiser state
        self._running = True

    def name(self):
        """See: :meth:`Optimiser.name()`."""
        return 'Exponential Natural Evolution Strategy (xNES)'

    def population_size(self):
        """ See :meth:`PopulationBasedOptimiser.population_size`. """
        return self._population_size

    def set_population_size(self, population_size=None, parallel=False):
        """ See :meth:`PopulationBasedOptimiser.set_population_size`. """
        if self._running:
            raise Exception('Cannot change settings during run.')

        # Check population size or set using heuristic
        if population_size is None:
            population_size = 4 + int(3 * np.log(self._dimension))
        else:
            population_size = int(population_size)
            if population_size < 1:
                raise ValueError('Population size must be at least 1.')

        # Round up to number of CPU cores
        if parallel:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            population_size = cpu_count * (
                ((population_size - 1) // cpu_count) + 1)

        # Store
        self._population_size = population_size

    def tell(self, fx):
        """See: :meth:`Optimiser.tell()`."""
        if not self._ready_for_tell:
            raise Exception('ask() not called before tell()')
        self._ready_for_tell = False

        # Tell CMA-ES
        self._es.tell(self._xs, fx)

    def xbest(self):
        """ See :meth:`Optimiser.xbest`. """
        x = self._es.result.xbest
        return self._x0 if x is None else x


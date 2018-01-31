#
# Root of the pints module.
# Provides access to all shared functionality (optimisation, mcmc, etc.).
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import sys

#
# Version info: Remember to keep this in sync with setup.py!
#
VERSION_INT = 0, 0, 1
VERSION = '.'.join([str(x) for x in VERSION_INT])
if sys.version_info[0] < 3:
    del(x)  # Before Python3, list comprehension iterators leaked

#
# Expose pints version
#
def version(formatted=False):
    if formatted:
        return 'Pints ' + VERSION
    else:
        return VERSION_INT

#
# Constants
#
FLOAT_FORMAT = '{:< 1.17e}'

#
# Core classes
#
from ._core import ForwardModel, SingleSeriesProblem

#
# Utility classes and methods
#
from ._util import strfloat, vector
from ._util import Timer

#
# Logs of probability density functions (not necessarily normalised)
#
from ._log_pdfs import LogPDF, LogPrior, LogLikelihood, LogPosterior

#
# Log-priors
#
from ._log_priors import (
    ComposedLogPrior,
    MultivariateNormalLogPrior,
    NormalLogPrior,
    UniformLogPrior,
)

#
# Log-likelihoods
#
from ._log_likelihoods import (
    KnownNoiseLogLikelihood,
    ScaledLogLikelihood,
    UnknownNoiseLogLikelihood,
)

#
# Boundaries
#
from ._boundaries import Boundaries

#
# Error measures
#
from ._error_measures import (
    ErrorMeasure,
    ProbabilityBasedError,
    RMSError,
    SumOfSquaresError,
)

#
# Parallel function evaluation
#
from ._evaluation import (
    evaluate,
    Evaluator,
    ParallelEvaluator,
    SequentialEvaluator,
)


#
# Optimisation
#
from ._optimisers import (
    Optimiser,
    TriangleWaveTransform,
    InfBoundaryTransform,
)
from ._optimisers._cmaes import CMAES, cmaes
from ._optimisers._pso import PSO, pso
from ._optimisers._snes import SNES, snes
from ._optimisers._xnes import XNES, xnes


# diagnostics
from ._diagnostics import (
    effective_sample_size,
    rhat,
    rhat_all_params,
)


#
# MCMC
#
from ._mcmc import MCMC
from ._mcmc._adaptive import (
    AdaptiveCovarianceMCMC,
    adaptive_covariance_mcmc,
)
from ._mcmc._differential_evolution import (
    DifferentialEvolutionMCMC,
    differential_evolution_mcmc,
    DreamMCMC
)
from ._mcmc._result import McmcResultObject

#
# Nested samplers
#
from ._nested import NestedSampler
from ._nested._rejection import NestedRejectionSampler
from ._nested._ellipsoid import NestedEllipsoidSampler

#
# Noise adders
#
import pints.noise

#
# Remove any imported modules, so we don't expose them as part of pints
#
del(sys)

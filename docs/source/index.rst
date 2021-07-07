.. Root of all pints docs

.. _GitHub: https://github.com/pints-team/pints
.. _Detailed examples: https://github.com/pints-team/pints/blob/master/examples/README.md

Welcome to the pints documentation
==================================

**Pints** is hosted on GitHub_, where you can find **downloads** and
**installation instructions**.

`Detailed examples`_ can also be found there.

*This* page provides the *API*, or *developer documentation* for ``pints``.

* :ref:`genindex`
* :ref:`search`

Contents
========

.. module:: pints

.. toctree::

    abc_samplers/index
    boundaries
    core_classes_and_methods
    diagnostics
    diagnostic_plots
    error_measures
    function_evaluation
    io
    log_likelihoods
    log_pdfs
    log_priors
    mcmc_samplers/index
    nested_samplers/index
    noise_generators
    optimisers/index
    noise_model_diagnostics
    toy/index
    transformations
    utilities

Hierarchy of methods
====================

Pints contains different types of methods, that can be roughly arranged into a
hierarchy, as follows.

Sampling
--------

#. :class:`MCMC without gradients<MCMCSampler>`

   - :class:`MetropolisRandomWalkMCMC`, works on any :class:`LogPDF`.
   - Metropolis-Hastings
   - Adaptive methods

     - :class:`AdaptiveCovarianceMC`, works on any :class:`LogPDF`.

   - :class:`PopulationMCMC`, works on any :class:`LogPDF`.
   - Differential evolution methods

     - :class:`DifferentialEvolutionMCMC`, works on any :class:`LogPDF`.
     - :class:`DreamMCMC`, works on any :class:`LogPDF`.
     - :class:`EmceeHammerMCMC`, works on any :class:`LogPDF`.

#. :class:`Nested sampling<NestedSampler>`

   - :class:`NestedEllipsoidSampler`, requires a :class:`LogPDF` and a
     :class:`LogPrior` that can be sampled from.
   - :class:`NestedRejectionSampler`, requires a :class:`LogPDF` and a
     :class:`LogPrior` that can be sampled from.

#. Particle based samplers

   - SMC

#. :class:`ABC sampling<ABCSampler>`

   - :class:`RejectionABC`, requires a :class:`LogPrior` that can be sampled
     from and an error measure.

#. 1st order sensitivity MCMC samplers (Need derivatives of :class:`LogPDF`)

   - :class:`Metropolis-Adjusted Langevin Algorithm (MALA) <MALAMCMC>`, works
     on any :class:`LogPDF` that provides 1st order sensitivities.
   - :class:`Hamiltonian Monte Carlo<HamiltonianMCMC>`, works on any
     :class:`LogPDF` that provides 1st order sensitivities.
   - NUTS

#. Differential geometric methods (Need Hessian of :class:`LogPDF`)

   - smMALA
   - RMHMC

Optimisation
------------

All methods shown here are derivative-free methods that work on any
:class:`ErrorMeasure` or :class:`LogPDF`.

1. Particle-based methods

   - Evolution strategies (global/local methods)

     - :class:`CMAES`
     - :class:`SNES`
     - :class:`XNES`

   - :class:`PSO` (global method)



Problems in Pints
=================

Pints defines :class:`single<SingleOutputProblem>` and
:class:`multi-output<MultiOutputProblem>` problem classes that wrap around
models and data, and over which :class:`error measures<ErrorMeasure>` or
:class:`log-likelihoods<LogLikelihood>` can be defined.

To find the appropriate type of Problem to use, see the overview below:

#. Systems with a single observable output

   - Single data set: Use a :class:`SingleOutputProblem` and any of the
     appropriate error measures or log-likelihoods
   - Multiple, independent data sets: Define multiple
     :class:`SingleOutputProblems<SingleOutputProblem>` and an error measure
     / log-likelihood on each, and then combine using e.g.
     :class:`SumOfErrors` or :class:`SumOfIndependentLogPDFs`.

#. Systems with multiple observable outputs

   - Single data set: Use a :class:`MultiOutputProblem` and any of the
     appropriate error measures or log-likelihoods

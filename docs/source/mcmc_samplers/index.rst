*************
MCMC Samplers
*************

.. currentmodule:: pints

Pints provides a number of MCMC methods, all implementing the :class:`MCMC`
interface, that can be used to sample from an unknown
:class:`PDF<pints.LogPDF>` (usually a Bayesian
:class:`Posterior<pints.LogPosterior>`).

.. toctree::

    running
    base_classes
    adaptive_covariance_mc
    differential_evolution_mcmc
    dram_ac_mcmc
    dream_mcmc
    emcee_hammer_mcmc
    haario_ac_mcmc
    haario_bardenet_ac_mcmc
    hamiltonian_mcmc
    mala_mcmc
    metropolis_mcmc
    monomial_gamma_hamiltonian_mcmc
    neal_langevin_mcmc
    population_mcmc
    rao_blackwell_ac_mcmc
    relativistic_mcmc
    slice_doubling_mcmc
    slice_rank_shrinking_mcmc
    slice_stepout_mcmc
    summary_mcmc

#
# Functions to calculate various MCMC diagnostics
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

import numpy as np

import pints.diagnostics


def _get_geyer_truncation(autocorrelation):
    """
    Estmiates the lag index of a autocorrelation sequence at which the
    estimation noise starts to dominate.

    The theory is following Geyer (1992) and essentially looks at the
    sum of even and odd lag pairs starting at 0. If the sum is negative
    the noise dominates the estimate.

    E.g.
    If sum of lag 0 and 1 is positive, check sum of lag 2 and 3.
    If sum of lag 2 and 3 is positive, check sum of lag 4 and 5.
    etc.
    if sum of lag n and n+1 is negative, truncate the autocorrelation
    sequence at n-1.
    """
    # Split sequence into even and odd
    even = autocorrelation[::2]
    odd = autocorrelation[1::2]

    # Make sure there are equally many even and odd entries
    max_index = np.min([len(even), len(odd)])
    even = even[:max_index]
    odd = odd[:max_index]

    # Compute sum and determine first negative entry
    even_and_odd_sum = even + odd
    mask = even_and_odd_sum < 0
    index = np.argmax(mask)
    if np.alltrue(~mask):
        index = max_index

    # Get truncation index for original sequence
    truncation_index = 2 * index - 1

    return truncation_index


def autocorrelation(chains):
    r"""
    Calculates the autocorrelation of samples for different lags.

    The autocorrelation of MCMC samples at lag :math:`n` is defined
    as the mean correlation between the samples in the chain and the
    samples shifted by :math:`n` iterations

    .. math::
        \rho _n = \frac{1}{\sigma ^2}\int \text{d}\theta \,
        \text{d}\theta _n \,(\theta - \mu ) (\theta _n - \mu )
        \, p(\theta ,\theta _n),

    where :math:`p(\theta , \theta _n )` is the joint distribution of the
    samples and the samples at lag :math:`n`. :math:`\mu` and
    :math:`\sigma ^2` are the mean and variance of the samples which is
    invariant under the lag :math:`n`. If there is no correlation
    between :math:`\theta` and :math:`\theta _n` for :math:`n\neq 0`, i.e.
    if :math:`\theta` and :math:`\theta _n` are i.i.d. distributed,
    :math:`\rho _n` is zero for all lags greater :math:`n\neq 0`. For
    :math:`n=0` the correlation is one.

    In practice we will approximate the above expression by the finite
    sample size analogon

    .. math::
        \hat{\rho} _n = \frac{1}{N\hat{\sigma} ^2}\sum ^N_{i=1}
        (\theta _i - \hat{\mu }) (\theta _{i+n} - \hat{\mu } ),

    where :math:`N` is the total number of samples in the chain and
    :math:`\theta _i` is the sample at iteration :math:`i`. Here,
    :math:`\hat{\mu }` and :math:`\hat{\sigma } ^2` are the estimates
    of the chain mean and variance

    .. math::
        \hat{\mu } = \frac{1}{N}\sum ^N_{i=1}\theta _i \quad \text{and}
        \quad
        \hat{\sigma } ^2 = \frac{1}{N-1}\sum ^N_{i=1}
        \left( \theta _i - \hat{\mu} \right) ^2.

    If samples for multiple parameters and/or chains are provided, the
    autocorrelation for different lags is computed for all marginal
    chains independently.

    .. note::
        The estimation of the autocorrelation is only justified if the
        samples are ordered according to their sample iteration.

    :param chains: A numpy array with :math:`n` samples and :math:`p`
        parameters. Optionally the autocorrelation may be computed
        simultaneously for :math:`m` chains.
    :type chains: np.ndarray of shape (n, p) or (m, n, p)

    Returns
    -------
    autocorr : np.ndarray of shape (m, n, p)
        The autocorrelation of :math:`m` chains for each parameter
        :math:`p` at lag :math:`n`.
    """
    # Make sure samples have the correct dimensions
    chains = np.asarray(chains)
    if (chains.dims < 2) or (chains.dims > 3):
        raise ValueError(
            'The chains array must have 2 or 3 dimensions.')

    # Reshape chains for later convenience
    if chains.dims == 2:
        # Add chain dimension
        chains = chains[np.newaxis, :, :]

    # Standardise sample (center and normalise by std.)
    chains = (chains - np.mean(chains, axis=1)) / np.std(chains, axis=1)

    # Create container for autocorrelations
    n_chains, n_samples, n_parameters = chains.shape
    autocorrelations = np.empty(shape=(n_chains, n_samples, n_parameters))

    # Compute autocorrelations for each chain and each parameter
    chains = chains.swapaxes(chains, axis1=1, axis2=2)
    for chain_id, chain_samples in enumerate(chains):
        for param_id, parameter_samples in enumerate(chain_samples):
            # Compute mean correlation
            # (np.correlate returns cumulative correlation)
            autocorr = np.correlate(
                parameter_samples, parameter_samples, mode='full')
            autocorr = autocorr / n_samples

            # At the center the correlation is 1 because the sliding samples
            # exactly overlap with the other samples, i.e. we compute compute
            # the correlation of each sample with itself.
            # (autocorr has length 2*n_samples - 1)
            autocorr = autocorr[n_samples:]

            # Add to container
            autocorrelations[chain_id, :, param_id]

    return autocorrelations


def effective_sample_size(chains, combine_chains=True):
    r"""
    Estimates the effective samples size (ESS) of MCMC chains.

    The effective sample size approximates the effective number of i.i.d.
    samples generated by a MCMC routine. For a single chain the effective
    samples size is defined as

    .. math::
        N_{\text{eff}} = \frac{N}{1 + 2\sum ^{\infty}_{n=1}\rho _n},

    where :math:`N` is the number of samples and :math:`\rho _n` is the
    mean correlation between the samples and the samples at lag :math:`n`, see
    :func:`autocorrelation`. The autocorrelation at lag
    :math:`n` and at :math:`-n` yield the same value, so we focus on
    positive lags and multiply by 2.

    Intuitively, the denominator reduces to 1 (:math:`N_{\text{eff}}=N`)
    when there is no correlation between the samples and will be greater
    than 1 (:math:`N_{\text{eff}}<N`) when there is positive correlation
    between the samples. Some samplers may also lead to negative
    correlation between samples (:math:`N_{\text{eff}}>N`) which is
    referred to as superefficient sampling.

    The error in the autocorrelation estimate :math:`\hat{\rho }_n`
    increases with the lag which motivates to truncate the autocorrelation
    sum in practice

    .. math::
        N_{\text{eff}} = \frac{N}{1 + 2\sum ^{t}_{n=1}\hat{\rho} _n},

    where :math:`t` is the truncation lag. We follow a widely accepted
    truncation criterion introduced by Geyer (1992).

    For :math:`M` chains with :math:`N` samples the total effective
    sample size is similarly estimated by

    .. math::
        N_{\text{eff}} = \frac{NM}{1 + 2\sum ^{t}_{n=1}\tilde{\rho} _n},

    where :math:`\tilde{\rho} _n` is a autocorrelation measure that
    incorporates the autocorrelations within the individual chains, the
    explored parameter range of the individual chains and the overall
    convergence

    .. math::
        \tilde{\rho} _n =
        1 - \frac{1 - \bar{\rho}_n}{\hat{R}^2}.

    Here :math:`\bar{\rho}_n` is the mean of the autocorrelations weighted
    by the normalised within chain variances

    .. math::
        \bar{\rho}_n = \sum ^M_{m=1}w_m \hat{\rho}_{nm}\quad
        \text{and} \quad w_m = \frac{\sigma ^2_m}{MW},

    where :math:`\hat{\rho}_{nm}` is the autocorrelation of chain
    :math:`m` at lag :math:`n`. :math:`W` is the mean within chain variance
    estimator and :math:`\hat{R}` is the MCMC chain convergence metric,
    see :func:`rhat`. Note that :math:`\bar{\rho} _0=1`.

    Intuitivley, :math:`\tilde{\rho}` tries to incorporate the mixing /
    convergence of the chains by comparing the normalised mean
    autocorrelation :math:`\bar{\rho}_n` to the convergence
    of the chains :math:`\hat{R}`. If the convergence is terrible
    :math:`\hat{R} \gg 1` the correlation of the samples is high, even if the
    estimated within chain autocorrelation is low
    :math:`\bar{\rho}_n \approx 0`. On the other hand if the mean within chain
    autocorrelation is large :math:`\bar{\rho}_n \approx 1` the overall
    autocorrelation is large irrespective of the convergence.

    :param chains: A numpy array with :math:`n` samples and :math:`p`
        parameters. Optionally the effective sample size may be computed
        simultaneously for :math:`m` chains.
    :type chains: np.ndarray of shape (n, p) or (m, n, p)
    :param combine_chains: A boolean that determines whether the effective
        samples sizes of multiple chains are kept separate or whether a
        combined estimate is returned according to the above equation.
        If ``False`` the effective sample sizes for each chain are returned
        separately as a :class:`numpy.ndarray` of shape (m, p).
    :type combine_chains: bool, optional

    Returns
    -------
    ess : float or np.ndarray of shape (p,) or (m, p)
        :math:`N_{\text{eff}}` of the chains for each parameter. Optionally
        the effective samples sizes across chains are combined or returned
        separately.
    """
    # Make sure chains have the correct dimensions
    chains = np.asarray(chains)
    if (chains.dims < 2) or (chains.dims > 3):
        raise ValueError(
            'The chains array must have 2 or 3 dimensions.')

    # Reshape chains for later convenience
    if chains.dims == 2:
        # Add chain dimension
        chains = chains[np.newaxis, :, :]

    # Compute autocorrelation of each chain and reshape for convenience
    autocorrs = autocorrelation(chains)
    autocorrs = np.swapaxes(autocorrs, axis1=1, axis2=2)

    # Return individual ESS if only one chain has been passed or
    # combine_chains=False
    n_chains, n_samples, n_parameters = chains.shape
    if (n_chains == 1) or (combine_chains is False):
        # Create container for ESSs
        eff_sample_sizes = np.empty(shape=(n_chains, n_parameters))

        # Compute ESS for each chain
        for chain_id, chain_autorrs in autocorrs:
            # Compute ESS for each parameter
            for param_id, autocorr in chain_autorrs:
                # Compute autocorrelation sum
                trunc_index = _get_geyer_truncation(autocorr)
                autocorr = np.sum(autocorr[1:trunc_index])

                # Compute effective samples size
                ess = n_samples / (1 + 2 * autocorr)

                # Add to conatainer
                eff_sample_sizes[chain_id, param_id] = ess

        # Remove padded dimensions
        if n_chains == 1:
            return eff_sample_sizes[0]

        return eff_sample_sizes

    # Create container for combined ESSs
    eff_sample_sizes = np.empty(shape=n_parameters)

    # Compute unnormalised weights for autocorrelations
    within_chain_variances = np.var(chains, axis=1, ddof=1)

    # Compute rhat
    rhats = pints.diagnostics.rhat(chains)

    # Compute ESS for each parameter
    for param_id in range(n_parameters):
        # Compute weighted average of autocorrelations
        autocorr = np.average(
            autocorrs[:, param_id],
            axis=0,
            weights=within_chain_variances[:, param_id])

        # Compute rho tilde
        autocorr = 1 - (1 - autocorr) / rhats[param_id]**2

        # Compute autocorrelation sum
        trunc_index = _get_geyer_truncation(autocorr)
        autocorr = np.sum(autocorr[1:trunc_index])

        # Compute effective samples size
        ess = n_chains * n_samples / (1 + 2 * autocorr)

        # Add to conatainer
        eff_sample_sizes[param_id] = ess

    # Remove padded dimensions
    if n_parameters == 1:
        return eff_sample_sizes[0]

    return eff_sample_sizes

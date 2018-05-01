#
# Core modules and methods
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


class ForwardModel(object):
    """
    Defines an interface for user-supplied forward models.

    Classes extending ``ForwardModel`` can implement the required methods
    directly in Python or interface with other languages (for example via
    Python wrappers around C code).
    """

    def __init__(self):
        super(ForwardModel, self).__init__()

    def n_parameters(self):
        """
        Returns the dimension of the parameter space.
        """
        raise NotImplementedError

    def simulate(self, parameters, times):
        """
        Runs a forward simulation with the given ``parameters`` and returns a
        time-series with data points corresponding to the given ``times``.

        Arguments:

        ``parameters``
            An ordered list of parameter values.
        ``times``
            The times at which to evaluate. Must be an ordered sequence,
            without duplicates, and without negative values.
            All simulations are started at time 0, regardless of whether this
            value appears in ``times``.

        Returns:

        A numpy array of length ``len(times)`` representing the values of the
        model at the given ``times``.

        Note: For efficiency, both ``parameters`` and ``times`` will be passed
        in as read-only numpy arrays.
        """
        raise NotImplementedError

    def n_outputs(self):
        """
        Returns the number of outputs this model has. The default is 1.
        """
        return 1


class ToyModel(object):
    """
    Defines an interface for the toy problems.
    """

    def __init__(self):
        super(ToyModel, self).__init__()

    def suggested_times(self):
        """
        Returns an numpy array of time points that is representative of the 
        model
        """
        raise NotImplementedError

    def suggested_parameters(self):
        """
        Returns an numpy array of the parameter values that are representative
        of the model. For example, these parameters might reproduce a
        particular result that the model is famous for.
        """
        raise NotImplementedError


class ForwardModelS1(ForwardModel):
    """
    Defines an interface for user-supplied forward models which can calculate
    the first-order derivative of the simulated values with respect to the
    parameters.

    Derived from :class:`pints.ForwardModel`.
    """

    def __init__(self):
        super(ForwardModelS1, self).__init__()

    def simulateS1(self, parameters, times):
        """
        Runs a forward simulation with the given ``parameters`` and returns a
        time-series with data points corresponding to the given ``times``,
        along with the sensitivities of the forward simulation with respect to
        the parameters.

        Arguments:

        ``parameters``
            An ordered list of parameter values.
        ``times``
            The times at which to evaluate. Must be an ordered sequence,
            without duplicates, and without negative values.
            All simulations are started at time 0, regardless of whether this
            value appears in ``times``.

        Returns:

        A tuple ``(y, y')`` of the simulated values ``y`` and their derivatives
        ``y'`` with resepect to the ``parameters``.
        The first entry ``y`` must be a sequence of ``n_times`` values, or
        a NumPy array of shape ``(n_times, n_outputs)``.
        The second entry ``y'`` must be a numpy array of shape
        ``(n_times, n_parameters)`` or an array of shape
        ``(n_times, n_outputs, n_parameters)``.
        """
        raise NotImplementedError


class SingleOutputProblem(object):
    """
    Represents an inference problem where a model is fit to a single time
    series, such as measured from a system with a single output.

    Arguments:

    ``model``
        A model or model wrapper extending :class:`ForwardModel`.
    ``times``
        A sequence of points in time. Must be non-negative and increasing.
    ``values``
        A sequence of scalar output values, measured at the times in ``times``.

    """

    def __init__(self, model, times, values):

        # Check model
        self._model = model
        if model.n_outputs() != 1:
            raise ValueError(
                'Only single-output models can be used for a'
                ' SingleOutputProblem.')

        # Check times, copy so that they can no longer be changed and set them
        # to read-only
        self._times = pints.vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times can not be negative.')
        if np.any(self._times[:-1] >= self._times[1:]):
            raise ValueError('Times must be increasing.')

        # Check values, copy so that they can no longer be changed
        self._values = pints.vector(values)

        # Check dimensions
        self._n_parameters = int(model.n_parameters())
        self._n_times = len(self._times)

        # Check times and values array have write shape
        if len(self._values) != self._n_times:
            raise ValueError(
                'Times and values arrays must have same length.')

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.
        """
        return self._model.simulate(parameters, self._times)

    def evaluateS1(self, parameters):
        """
        Runs a simulation with first-order sensitivity calculation, returning
        the simulated values and derivatives.

        The returned data is a tuple ``(y, y')``, where ``y`` is a sequence of
        length ``n_times``, while ``y'`` has shape ``(n_times, n_parameters)``.

        *This method only works for problems whose model implements the
        :class:`ForwardModelS1` interface.*
        """
        return self._model.simulateS1(parameters, self._times)

    def n_outputs(self):
        """
        Returns the number of outputs for this problem (always 1).
        """
        return 1

    def n_parameters(self):
        """
        Returns the dimension (the number of parameters) of this problem.
        """
        return self._n_parameters

    def n_times(self):
        """
        Returns the number of sampling points, i.e. the length of the vectors
        returned by :meth:`times()` and :meth:`values()`.
        """
        return self._n_times

    def times(self):
        """
        Returns this problem's times.

        The returned value is a read-only numpy array of shape ``(n_times, )``,
        where ``n_times`` is the number of time points.
        """
        return self._times

    def values(self):
        """
        Returns this problem's values.

        The returned value is a read-only numpy array of shape ``(n_times, )``,
        where ``n_times`` is the number of time points.
        """
        return self._values


class MultiOutputProblem(object):
    """
    Represents an inference problem where a model is fit to a multi-valued time
    series, such as measured from a system with multiple outputs.

    Arguments:

    ``model``
        A model or model wrapper extending :class:`ForwardModel`.
    ``times``
        A sequence of points in time. Must be non-negative and non-decreasing.
    ``values``
        A sequence of multi-valued measurements. Must have shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of points in
        ``times`` and ``n_outputs`` is the number of outputs in the model.

    """

    def __init__(self, model, times, values):

        # Check model
        self._model = model

        # Check times, copy so that they can no longer be changed and set them
        # to read-only
        self._times = pints.vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
        if np.any(self._times[:-1] > self._times[1:]):
            raise ValueError('Times must be non-decreasing.')

        # Check values, copy so that they can no longer be changed
        self._values = pints.matrix2d(values)

        # Check dimensions
        self._n_parameters = int(model.n_parameters())
        self._n_outputs = int(model.n_outputs())
        self._n_times = len(self._times)

        # Check for correct shape
        if self._values.shape != (self._n_times, self._n_outputs):
            raise ValueError(
                'Values array must have shape `(n_times, n_outputs)`.')

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.

        The returned data has shape ``(n_times, n_outputs)``.
        """
        return self._model.simulate(parameters, self._times)

    def evaluateS1(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.

        The returned data is a tuple ``(y, y')``, where ``y`` has shape
        ``(n_times, n_outputs)``, while ``y'`` has shape
        ``(n_times, n_outputs, n_parameters)``.

        *This method only works for problems whose model implements the
        :class:`ForwardModelS1` interface.*
        """
        return self._model.simulateS1(parameters, self._times)

    def n_outputs(self):
        """
        Returns the number of outputs for this problem.
        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the dimension (the number of parameters) of this problem.
        """
        return self._n_parameters

    def n_times(self):
        """
        Returns the number of sampling points, i.e. the length of the vectors
        returned by :meth:`times()` and :meth:`values()`.
        """
        return self._n_times

    def times(self):
        """
        Returns this problem's times.

        The returned value is a read-only numpy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._times

    def values(self):
        """
        Returns this problem's values.

        The returned value is a read-only numpy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._values

**********
Optimisers
**********

.. currentmodule:: pints

Pints provides a number of optimisers, all implementing the :class:`Optimiser`
interface, that can be used to find the parameters that minimise an
:class:`ErrorMeasure` or maximise a :class:`LogPDF`.

The easiest way to run an optimisation is by using the :func:`optimise` method
or the :class:`OptimisationController` class.

.. toctree::

    running
    base_classes
    convenience_methods
    boundary_transformations
    cmaes_bare
    cmaes
    gradient_descent
    lbfgs
    nelder_mead
    pso
    snes
    xnes


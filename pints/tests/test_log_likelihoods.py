#!/usr/bin/env python3
#
# Tests the log likelihood classes.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import pints
import pints.toy
import numpy as np


class TestAR1LogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(4)

        # Generate test data
        cls.times = np.asarray([1, 2, 3])
        cls.n_times = len(cls.times)
        cls.data_single = np.asarray([1.0, -10.7, 15.5])
        cls.data_multi = np.asarray([
            [3.5, 7.6, 8.5, 3.4],
            [1.1, -10.3, 15.6, 5.5],
            [-10, -30.5, -5, 7.6]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.AR1LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.5, 5]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -19.706737485492436)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.AR1LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.5, 5]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -19.706737485492436)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.AR1LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.5, 5]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -19.706737485492436)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.AR1LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [
            0, 0, 0, 0, 0.5, 1.0, -0.25, 3.0, 0.9, 10.0, 0.0, 2.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -179.22342804581092)


class TestARMA11LogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(4)

        # Generate test data
        cls.times = np.asarray([1, 2, 3, 4])
        cls.n_times = len(cls.times)
        cls.data_single = np.asarray([3, -4.5, 10.5, 0.3])
        cls.data_multi = np.asarray([
            [3.5, 7.6, 8.5, 3.4],
            [1.1, -10.3, 15.6, 5.5],
            [-10, -30.5, -5, 7.6],
            [-12, -10.1, -4, 2.3]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ARMA11LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.9, -0.4, 1]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -171.53031588534171)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ARMA11LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.9, -0.4, 1]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -171.53031588534171)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ARMA11LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0.9, -0.4, 1]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -171.53031588534171)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.ARMA11LogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [
            0, 0, 0, 0, 0.5, 0.34, 1.0, -0.25, 0.1, 3.0, 0.9, 0.0, 10.0, 0.0,
            0.9, 2.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -214.17034137601107)


class TestCauchyLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(4)

        # Generate test data
        cls.times = np.asarray([1, 2, 3])
        cls.n_times = len(cls.times)
        cls.data_single = np.asarray([1.0, -10.7, 15.5])
        cls.data_multi = np.asarray([
            [3.5, 7.6, 8.5, 3.4],
            [1.1, -10.3, 15.6, 5.5],
            [-10, -30.5, -5, 7.6]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 10]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -12.339498654173603)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 10]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -12.339498654173603)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 10]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -12.339498654173603)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [
            0, 0, 0, 0, 13, 8, 13.5, 10.5]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -49.51182454195375)

    def test_evaluate_pointwise_loglikelihoods_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [0, 10]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [0, 10]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [0, 10]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        likelihood = pints.CauchyLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [0, 0, 0, 0, 13, 8, 13.5, 10.5]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, self.data_multi.shape)

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))


class TestConstantAndMultiplicativeGaussianLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(3)

        # Generate test data
        cls.times = np.array([1, 2, 3, 4])
        cls.n_times = len(cls.times)
        cls.data_single = np.array([1, 2, 3, 4]) / 5.0
        cls.data_multi = np.array([
            [10.7, 3.5, 3.8],
            [1.1, 3.2, -1.4],
            [9.3, 0.0, 4.5],
            [1.2, -3, -10]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -8.222479586661642)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -8.222479586661642)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -8.222479586661642)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [
            2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -42.87921520701031)

    def test_call_gaussian_log_likelihood_agrees_single(self):
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, self.data_single)

        # Create CombinedGaussianLL and GaussianLL
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        gauss_log_likelihood = pints.GaussianLogLikelihood(problem)

        # Check that CombinedGaussianLL agrees with GaussianLoglikelihood when
        # sigma_rel = 0 and sigma_base = sigma
        test_parameters = [2.0, 0.5, 1.1, 0.0]
        gauss_test_parameters = [2.0, 0.5]
        score = log_likelihood(test_parameters)
        gauss_score = gauss_log_likelihood(gauss_test_parameters)
        self.assertAlmostEqual(score, gauss_score)

    def test_call_gaussian_log_likelihood_agrees_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create CombinedGaussianLL and GaussianLL
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        gauss_log_likelihood = pints.GaussianLogLikelihood(problem)

        # Check that CombinedGaussianLL agrees with GaussianLoglikelihood when
        # sigma_rel = 0 and sigma_base = sigma
        test_parameters = [
            2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 1.1, 1.1, 1.1, 0.0, 0.0, 0.0]
        gauss_test_parameters = [2.0, 2.0, 2.0, 0.5, 0.5, 0.5]
        score = log_likelihood(test_parameters)
        gauss_score = gauss_log_likelihood(gauss_test_parameters)
        self.assertAlmostEqual(score, gauss_score)

    def test_call_multiplicative_gaussian_log_likelihood_agrees_single(self):
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, self.data_single)

        # Create CombinedGaussianLL and MultplicativeGaussianLL
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        multi_log_likelihood = pints.MultiplicativeGaussianLogLikelihood(
            problem)

        # Check that CombinedGaussianLL agrees with
        # MultiplicativeGaussianLoglikelihood when sigma_base = 0,
        # eta = eta, and sigma_rel = sigma
        test_parameters = [2.0, 0.0, 1.1, 1.0]
        multi_test_parameters = [2.0, 1.1, 1.0]
        score = log_likelihood(test_parameters)
        multi_score = multi_log_likelihood(multi_test_parameters)
        self.assertAlmostEqual(score, multi_score)

    def test_call_multiplicative_gaussian_log_likelihood_agrees_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create CombinedGaussianLL and MultplicativeGaussianLL
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        multi_log_likelihood = pints.MultiplicativeGaussianLogLikelihood(
            problem)

        # Check that CombinedGaussianLL agrees with
        # MultiplicativeGaussianLoglikelihood when sigma_base = 0,
        # eta = eta, and sigma_rel = sigma
        test_parameters = [
            2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0]
        multi_test_parameters = [2.0, 2.0, 2.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0]
        score = log_likelihood(test_parameters)
        multi_score = multi_log_likelihood(multi_test_parameters)
        self.assertAlmostEqual(score, multi_score)

    def test_evaluate_pointwise_loglikelihoods_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        parameters = [2.0, 0.5, 1.1, 1.0]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        parameters = [2.0, 0.5, 1.1, 1.0]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        parameters = [2.0, 0.5, 1.1, 1.0]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        parameters = [
            2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, self.data_multi.shape)

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluateS1_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that likelihood score agrees with call
        # There are floating point deviations because in evaluateS1
        # log(sigma_tot) is for efficiency computed as -log(1/sigma_tot)
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that number of partials is correct
        self.assertAlmostEqual(deriv.shape, (4,))

        # Check that partials are computed correctly
        self.assertAlmostEqual(deriv[0], -2.055351334007383)
        self.assertAlmostEqual(deriv[1], -1.0151215581116324)
        self.assertAlmostEqual(deriv[2], -1.5082610203777322)
        self.assertAlmostEqual(deriv[3], -2.1759606944650822)

    def test_evaluateS1_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that likelihood score agrees with call
        # There are floating point deviations because in evaluateS1
        # log(sigma_tot) is for efficiency computed as -log(1/sigma_tot)
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that number of partials is correct
        self.assertAlmostEqual(deriv.shape, (4,))

        # Check that partials are computed correctly
        self.assertAlmostEqual(deriv[0], -2.055351334007383)
        self.assertAlmostEqual(deriv[1], -1.0151215581116324)
        self.assertAlmostEqual(deriv[2], -1.5082610203777322)
        self.assertAlmostEqual(deriv[3], -2.1759606944650822)

    def test_evaluateS1_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 0.5, 1.1, 1.0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that likelihood score agrees with call
        # There are floating point deviations because in evaluateS1
        # log(sigma_tot) is for efficiency computed as -log(1/sigma_tot)
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that number of partials is correct
        self.assertAlmostEqual(deriv.shape, (4,))

        # Check that partials are computed correctly
        self.assertAlmostEqual(deriv[0], -2.0553513340073835)
        self.assertAlmostEqual(deriv[1], -1.0151215581116324)
        self.assertAlmostEqual(deriv[2], -1.5082610203777322)
        self.assertAlmostEqual(deriv[3], -2.1759606944650822)

    def test_evaluateS1_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Evaluate likelihood for test parameters
        test_parameters = [
            2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that likelihood score agrees with call
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that number of partials is correct
        self.assertAlmostEqual(deriv.shape, (12,))

        # Check that partials are computed correctly
        self.assertAlmostEqual(deriv[0], 8.585990509232376)
        self.assertAlmostEqual(deriv[1], -1.6726936107293917)
        self.assertAlmostEqual(deriv[2], -0.6632862192355309)
        self.assertAlmostEqual(deriv[3], 5.547071959874058)
        self.assertAlmostEqual(deriv[4], -0.2868738955802226)
        self.assertAlmostEqual(deriv[5], 0.1813851785335695)
        self.assertAlmostEqual(deriv[6], 8.241803503682762)
        self.assertAlmostEqual(deriv[7], -1.82731103999105)
        self.assertAlmostEqual(deriv[8], 2.33264086991343)
        self.assertAlmostEqual(deriv[9], 11.890409042744405)
        self.assertAlmostEqual(deriv[10], -1.3181262877783717)
        self.assertAlmostEqual(deriv[11], 1.3018716574264304)

    def test_evaluateS1_gaussian_log_likelihood_agrees_single(self):
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, self.data_single)

        # Create CombinedGaussianLL and GaussianLL
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        gauss_log_likelihood = pints.GaussianLogLikelihood(problem)

        # Check that CombinedGaussianLL agrees with GaussianLoglikelihood when
        # sigma_rel = 0 and sigma_base = sigma
        test_parameters = [2.0, 0.5, 1.1, 0.0]
        gauss_test_parameters = [2.0, 0.5]
        score, deriv = log_likelihood.evaluateS1(test_parameters)
        gauss_score, gauss_deriv = gauss_log_likelihood.evaluateS1(
            gauss_test_parameters)

        # Check that scores are the same
        self.assertAlmostEqual(score, gauss_score)

        # Check that partials for model params and sigma_base agree
        self.assertAlmostEqual(deriv[0], gauss_deriv[0])
        self.assertAlmostEqual(deriv[1], gauss_deriv[1])

    def test_evaluateS1_gaussian_log_likelihood_agrees_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create CombinedGaussianLL and GaussianLL
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)
        gauss_log_likelihood = pints.GaussianLogLikelihood(problem)

        # Check that CombinedGaussianLL agrees with GaussianLoglikelihood when
        # sigma_rel = 0 and sigma_base = sigma
        test_parameters = [
            2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 1.1, 1.1, 1.1, 0.0, 0.0, 0.0]
        gauss_test_parameters = [2.0, 2.0, 2.0, 0.5, 0.5, 0.5]
        score, deriv = log_likelihood.evaluateS1(test_parameters)
        gauss_score, gauss_deriv = gauss_log_likelihood.evaluateS1(
            gauss_test_parameters)

        # Check that scores are the same
        self.assertAlmostEqual(score, gauss_score)

        # Check that partials for model params and sigma_base agree
        self.assertAlmostEqual(deriv[0], gauss_deriv[0])
        self.assertAlmostEqual(deriv[1], gauss_deriv[1])
        self.assertAlmostEqual(deriv[2], gauss_deriv[2])
        self.assertAlmostEqual(deriv[3], gauss_deriv[3])
        self.assertAlmostEqual(deriv[4], gauss_deriv[4])
        self.assertAlmostEqual(deriv[5], gauss_deriv[5])

    def test_evaluateS1_finite_difference_single(self):
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, self.data_single)

        # Create log-likelihood
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Compute derivatives with evaluateS1
        test_parameters = np.array([2.0, 0.5, 1.1, 1.0])
        _, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that finite difference approximately agrees with evaluateS1
        # Theta
        eps = np.array([1E-3, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[0], (score_after - score_before) / eps[0])

        # Sigma base
        eps = np.array([0, 1E-3, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[1], (score_after - score_before) / eps[1])

        # Eta
        eps = np.array([0, 0, 1E-4, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[2], (score_after - score_before) / eps[2])

        # Sigma rel
        eps = np.array([0, 0, 0, 1E-4])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[3], (score_after - score_before) / eps[3])

    def test_evaluateS1_finite_difference_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log-likelihood
        log_likelihood = pints.ConstantAndMultiplicativeGaussianLogLikelihood(
            problem)

        # Compute derivatives with evaluateS1
        test_parameters = [
            2.0, 1.9, 2.1, 0.5, 0.4, 0.6, 1.1, 1.0, 1.2, 1.0, 0.9, 1.1]
        _, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that finite difference approximately agrees with evaluateS1
        # Theta output 1
        eps = np.array([1E-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[0], (score_after - score_before) / eps[0])

        # Theta output 2
        eps = np.array([0, 1E-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[1], (score_after - score_before) / eps[1])

        # Theta output 3
        eps = np.array([0, 0, 1E-4, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[2], (score_after - score_before) / eps[2])

        # Sigma base output 1
        eps = np.array([0, 0, 0, 1E-4, 0, 0, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[3], (score_after - score_before) / eps[3])

        # Sigma base output 2
        eps = np.array([0, 0, 0, 0, 1E-4, 0, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[4], (score_after - score_before) / eps[4])

        # Sigma base output 3
        eps = np.array([0, 0, 0, 0, 0, 1E-4, 0, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[5], (score_after - score_before) / eps[5])

        # Eta output 1
        eps = np.array([0, 0, 0, 0, 0, 0, 1E-4, 0, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[6], (score_after - score_before) / eps[6])

        # Eta output 2
        eps = np.array([0, 0, 0, 0, 0, 0, 0, 1E-4, 0, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[7], (score_after - score_before) / eps[7])

        # Eta output 3
        eps = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1E-4, 0, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[8], (score_after - score_before) / eps[8])

        # Sigma rel ouput 1
        eps = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1E-4, 0, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(deriv[9], (score_after - score_before) / eps[9])

        # Sigma rel ouput 2
        eps = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1E-4, 0])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(
            deriv[10], (score_after - score_before) / eps[10])

        # Sigma rel ouput 3
        eps = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1E-4])
        score_before = log_likelihood(test_parameters - eps / 2)
        score_after = log_likelihood(test_parameters + eps / 2)
        self.assertAlmostEqual(
            deriv[11], (score_after - score_before) / eps[11])


class TestGaussianIntegratedUniformLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(4)

        # Generate test data
        cls.times = np.asarray([1, 2, 3])
        cls.n_times = len(cls.times)
        cls.data_single = np.asarray([1.0, -10.7, 15.5])
        cls.data_multi = np.asarray([
            [3.4, 4.3, 22.0, -7.3],
            [11.1, 12.2, 13.9, 5.0],
            [-0.4, -12.3, -8.3, -1.2]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.GaussianIntegratedUniformLogLikelihood(
            problem, 2, 4)

        # Evaluate likelihood for test parameters
        test_parameters = [0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -20.441037907121299)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.GaussianIntegratedUniformLogLikelihood(
            problem, 2, 4)

        # Evaluate likelihood for test parameters
        test_parameters = [0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -20.441037907121299)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.GaussianIntegratedUniformLogLikelihood(
            problem, 2, 4)

        # Evaluate likelihood for test parameters
        test_parameters = [0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -20.441037907121299)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.GaussianIntegratedUniformLogLikelihood(
            problem, 2, 4)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0, 0, 0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -75.443307614807225)

    def test_call_two_dim_array_multi_non_equal_priors(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.GaussianIntegratedUniformLogLikelihood(
            problem, [1, 0, 5, 2], [2, 4, 7, 8])

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0, 0, 0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -71.62076263891457)

    def test_bad_constructor_single(self):
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, self.data_single)

        # Check negative bound
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, -1, 2)

        # Check vanishing interval width
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, 0, 0)

        # Check higher lower bound than upper bound
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, 2, 1)

        # Check wrong prior dimensions
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, [1, 2], 2)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, 1, [2, 3])
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, [1, 2], [2, 3])

    def test_bad_constructor_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Check wrong prior dimensions
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, 2, 2)
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, [1, 2, 3], [2, 4])
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, [1, 2], [2, 4, 5])
        self.assertRaises(ValueError,
                          pints.GaussianIntegratedUniformLogLikelihood,
                          problem, [1, 3], [2, 2])


class TestGaussianKnownSigmaLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(3)

        # Generate test data
        cls.times = [1, 2, 3, 4]
        cls.n_times = len(cls.times)
        cls.data_single = np.arange(1, 5) / 5.0
        cls.data_multi = np.array([
            [10.7, 3.5, 3.8],
            [1.1, 3.2, -1.4],
            [9.3, 0.0, 4.5],
            [1.2, -3, -10]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)

        # Evaluate likelihood for test parameters
        test_parameters = [-1]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -7.3420590096957925)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)

        # Evaluate likelihood for test parameters
        test_parameters = [-1]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -7.3420590096957925)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)

        # Evaluate likelihood for test parameters
        test_parameters = [-1]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -7.3420590096957925)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0, 0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -196.9122623984561)

    def test_evaluate_pointwise_loglikelihoods_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)

        # Evaluate likelihood for test parameters
        parameters = [-1]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)

        # Evaluate likelihood for test parameters
        parameters = [-1]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)

        # Evaluate likelihood for test parameters
        parameters = [-1]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1)

        # Evaluate likelihood for test parameters
        parameters = [0, 0, 0]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, self.data_multi.shape)

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluateS1_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [3]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that score is computed correctly
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv.shape, (1, ))

        # Check that partials are comuted correctly
        self.assertAlmostEqual(deriv[0], -4.444444444444445)

    def test_evaluateS1_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [3]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that score is computed correctly
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv.shape, (1, ))

        # Check that partials are comuted correctly
        self.assertAlmostEqual(deriv[0], -4.444444444444445)

    def test_evaluateS1_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1.5)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [3]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that score is computed correctly
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv.shape, (1, ))

        # Check that partials are comuted correctly
        self.assertAlmostEqual(deriv[0], -4.444444444444445)

    def test_evaluateS1_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, 1)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [0, 0, 0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that score is computed correctly
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv.shape, (3, ))

        # Check that partials are comuted correctly
        self.assertAlmostEqual(deriv[0], 22.3)
        self.assertAlmostEqual(deriv[1], 2 * 3.7000000000000002)
        self.assertAlmostEqual(deriv[2], -9.3)

    def test_deprecated_alias(self):
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, self.data_single)

        # Create deprecated alias
        log_likelihood = pints.KnownNoiseLogLikelihood(problem, 0.1)

        # Check inheritance from current class
        self.assertIsInstance(
            log_likelihood, pints.GaussianKnownSigmaLogLikelihood)

    def test_bad_constructor(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_single, self.times, self.data_single)

        # Check wrong prior dimensions
        self.assertRaises(
            ValueError, pints.GaussianKnownSigmaLogLikelihood, problem, 0)
        self.assertRaises(
            ValueError,
            pints.GaussianKnownSigmaLogLikelihood, problem, [0.1, 0.2])
        self.assertRaises(
            ValueError, pints.GaussianKnownSigmaLogLikelihood, problem, -1)


class TestGaussianLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(3)

        # Generate test data
        cls.times = np.array([1, 2, 3, 4])
        cls.n_times = len(cls.times)
        cls.data_single = np.array([1, 2, 3, 4]) / 5.0
        cls.data_multi = np.array([
            [10.7, 3.5, 3.8],
            [1.1, 3.2, -1.4],
            [9.3, 0.0, 4.5],
            [1.2, -3, -10]])

        # Add noise to the data
        np.random.seed(42)
        cls.sigma = 0.1
        cls.data_single += np.random.normal(
            0, cls.sigma, cls.data_single.shape)
        cls.data_multi += np.random.normal(0, cls.sigma, cls.data_multi.shape)

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihoods with known and unknown sigma
        log_likelihood = pints.GaussianLogLikelihood(problem)
        log_likelihood_known = pints.GaussianKnownSigmaLogLikelihood(
            problem, self.sigma)

        # Evaluate likelihood for test parameters
        test_parameters = [2, self.sigma]
        score = log_likelihood(test_parameters)

        # Check that score between known and unknown sigma likelihoods agree
        self.assertAlmostEqual(
            log_likelihood(test_parameters),
            log_likelihood_known(test_parameters[:-1]))

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -421.8952711914118)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihoods with known and unknown sigma
        log_likelihood = pints.GaussianLogLikelihood(problem)
        log_likelihood_known = pints.GaussianKnownSigmaLogLikelihood(
            problem, self.sigma)

        # Evaluate likelihood for test parameters
        test_parameters = [2, self.sigma]
        score = log_likelihood(test_parameters)

        # Check that score between known and unknown sigma likelihoods agree
        self.assertAlmostEqual(
            log_likelihood(test_parameters),
            log_likelihood_known(test_parameters[:-1]))

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -421.8952711914118)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihoods with known and unknown sigma
        log_likelihood = pints.GaussianLogLikelihood(problem)
        log_likelihood_known = pints.GaussianKnownSigmaLogLikelihood(
            problem, self.sigma)

        # Evaluate likelihood for test parameters
        test_parameters = [2, self.sigma]
        score = log_likelihood(test_parameters)

        # Check that score between known and unknown sigma likelihoods agree
        self.assertAlmostEqual(
            log_likelihood(test_parameters),
            log_likelihood_known(test_parameters[:-1]))

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -421.8952711914118)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihoods with known and unknown sigma
        log_likelihood = pints.GaussianLogLikelihood(problem)
        log_likelihood_known = pints.GaussianKnownSigmaLogLikelihood(
            problem, self.sigma)

        # Check that score between known and unknown sigma likelihoods agree
        self.assertAlmostEqual(
            log_likelihood([0, 0, 0, 0.1, 0.1, 0.1]),
            log_likelihood_known([0, 0, 0]))

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0, 0, 3.5, 1, 12]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -50.75425117450455)

    def test_evaluate_pointwise_loglikelihoods_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.GaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [2, self.sigma]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.GaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [2, self.sigma]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.GaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [2, self.sigma]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        likelihood = pints.GaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [0, 0, 0, 0.1, 0.1, 0.1]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, self.data_multi.shape)

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluateS1_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihoods with known and unknown sigma
        log_likelihood = pints.GaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [7, 2.0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that score is computed correctly
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv.shape, (2, ))

        # Check that partials are comuted correctly
        self.assertAlmostEqual(deriv[0], -6.436770793841281)
        self.assertAlmostEqual(deriv[1], 18.75242861278283)

    def test_evaluateS1_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihoods with known and unknown sigma
        log_likelihood = pints.GaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [7, 2.0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that score is computed correctly
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv.shape, (2, ))

        # Check that partials are comuted correctly
        self.assertAlmostEqual(deriv[0], -6.436770793841281)
        self.assertAlmostEqual(deriv[1], 18.75242861278283)

    def test_evaluateS1_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihoods with known and unknown sigma
        log_likelihood = pints.GaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [7, 2.0]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that score is computed correctly
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv.shape, (2, ))

        # Check that partials are comuted correctly
        self.assertAlmostEqual(deriv[0], -6.436770793841281)
        self.assertAlmostEqual(deriv[1], 18.75242861278283)

    def test_evaluateS1_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihoods with known and unknown sigma
        log_likelihood = pints.GaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0, 0, 3.5, 1, 12]
        score, deriv = log_likelihood.evaluateS1(test_parameters)

        # Check that score is computed correctly
        self.assertAlmostEqual(score, log_likelihood(test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv.shape, (6, ))

        # Check that partials are comuted correctly
        self.assertAlmostEqual(deriv[0], 1.8053598646282394)
        self.assertAlmostEqual(deriv[1], 6.821148214206516)
        self.assertAlmostEqual(deriv[2], -0.06083031749704496)
        self.assertAlmostEqual(deriv[3], 3.5690214591801)
        self.assertAlmostEqual(deriv[4], 28.09526594310443)
        self.assertAlmostEqual(deriv[5], -0.25285170370039783)

    def test_deprecated_alias(self):
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, self.data_single)

        # Create deprecated alias
        log_likelihood = pints.UnknownNoiseLogLikelihood(problem)

        # Check inheritance from current class
        self.assertIsInstance(
            log_likelihood, pints.GaussianLogLikelihood)


class TestKnownNoiseLogLikelihood(unittest.TestCase):

    def test_known_noise_gaussian_single_and_multi(self):
        # Tests the output of single-series against multi-series known noise
        # log-likelihoods.

        # Define boring 1-output and 2-output models
        class NullModel1(pints.ForwardModel):
            def n_parameters(self):
                return 1

            def simulate(self, x, times):
                return np.zeros(times.shape)

        class NullModel2(pints.ForwardModel):
            def n_parameters(self):
                return 1

            def n_outputs(self):
                return 2

            def simulate(self, x, times):
                return np.zeros((len(times), 2))

        # Create two single output problems
        times = np.arange(10)
        np.random.seed(1)
        sigma1 = 3
        sigma2 = 5
        values1 = np.random.uniform(0, sigma1, times.shape)
        values2 = np.random.uniform(0, sigma2, times.shape)
        model1d = NullModel1()
        problem1 = pints.SingleOutputProblem(model1d, times, values1)
        problem2 = pints.SingleOutputProblem(model1d, times, values2)
        log1 = pints.GaussianKnownSigmaLogLikelihood(problem1, sigma1)
        log2 = pints.GaussianKnownSigmaLogLikelihood(problem2, sigma2)

        # Create one multi output problem
        values3 = np.array([values1, values2]).swapaxes(0, 1)
        model2d = NullModel2()
        problem3 = pints.MultiOutputProblem(model2d, times, values3)
        log3 = pints.GaussianKnownSigmaLogLikelihood(
            problem3, [sigma1, sigma2])

        # Check if we get the right output
        self.assertAlmostEqual(log1(0) + log2(0), log3(0))


class TestMultiplicativeGaussianLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(3)

        # Generate test data
        cls.times = np.array([1, 2, 3, 4])
        cls.n_times = len(cls.times)
        cls.data_single = np.asarray([1.9, 2.1, 1.8, 2.2])
        cls.data_multi = np.array([
            [10.7, 3.5, 3.8],
            [1.1, 3.2, -1.4],
            [9.3, 0.0, 4.5],
            [1.2, -3, -10]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create Gaussian and MuliplicativeGaussian LogLikelihood
        gaussian_log_likelihood = pints.GaussianLogLikelihood(problem)
        log_likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)

        # Check that likelihoods agree for eta=0
        self.assertAlmostEqual(
            log_likelihood([2.0, 0.0, 1.0]),
            gaussian_log_likelihood([2.0, 1.0]))

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 2.0, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -9.224056577298253)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create Gaussian and MuliplicativeGaussian LogLikelihood
        gaussian_log_likelihood = pints.GaussianLogLikelihood(problem)
        log_likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)

        # Check that likelihoods agree for eta=0
        self.assertAlmostEqual(
            log_likelihood([2.0, 0.0, 1.0]),
            gaussian_log_likelihood([2.0, 1.0]))

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 2.0, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -9.224056577298253)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create Gaussian and MuliplicativeGaussian LogLikelihood
        gaussian_log_likelihood = pints.GaussianLogLikelihood(problem)
        log_likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)

        # Check that likelihoods agree for eta=0
        self.assertAlmostEqual(
            log_likelihood([2.0, 0.0, 1.0]),
            gaussian_log_likelihood([2.0, 1.0]))

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 2.0, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -9.224056577298253)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create Gaussian and MuliplicativeGaussian LogLikelihood
        gaussian_log_likelihood = pints.GaussianLogLikelihood(problem)
        log_likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)

        # Check that likelihoods agree for eta=0
        gaussian_test_parameters = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
        test_parameters = [2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        self.assertAlmostEqual(
            log_likelihood(test_parameters),
            gaussian_log_likelihood(gaussian_test_parameters))

        # Evaluate likelihood for test parameters
        test_parameters = [2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]
        score = log_likelihood(test_parameters)

        # Check that likelihood returns expected value
        self.assertAlmostEqual(score, -46.324126706784014)

    def test_evaluate_pointwise_loglikelihoods_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [2.0, 2.0, 1.0]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [2.0, 2.0, 1.0]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [2.0, 2.0, 1.0]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        likelihood = pints.MultiplicativeGaussianLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, self.data_multi.shape)

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))


class TestScaledLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.LogisticModel()
        cls.model_multi = pints.toy.ConstantModel(2)

        # Generate test data
        cls.n_times = 10
        cls.times = np.linspace(0, 1000, cls.n_times)
        cls.data_single = cls.model_single.simulate(
            parameters=[0.015, 500], times=cls.times)
        cls.data_multi = cls.model_multi.simulate(
            parameters=[1, 2], times=cls.times)

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Evaluate likelihoods for test parameters
        test_parameters = [0.014, 501]
        score_not_scaled = log_likelihood_not_scaled(test_parameters)
        score_scaled = log_likelihood_scaled(test_parameters)

        # Check that unscaled likelihood returns expected value
        self.assertEqual(int(score_not_scaled), -1897896120)

        # Check that scaled likelihood returns expected value
        self.assertAlmostEqual(score_scaled * self.n_times, score_not_scaled)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Evaluate likelihoods for test parameters
        test_parameters = [0.014, 501]
        score_not_scaled = log_likelihood_not_scaled(test_parameters)
        score_scaled = log_likelihood_scaled(test_parameters)

        # Check that unscaled likelihood returns expected value
        self.assertEqual(int(score_not_scaled), -1897896120)

        # Check that scaled likelihood returns expected value
        self.assertAlmostEqual(score_scaled * self.n_times, score_not_scaled)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Evaluate likelihoods for test parameters
        test_parameters = [0.014, 501]
        score_not_scaled = log_likelihood_not_scaled(test_parameters)
        score_scaled = log_likelihood_scaled(test_parameters)

        # Check that unscaled likelihood returns expected value
        self.assertEqual(int(score_not_scaled), -1897896120)

        # Check that scaled likelihood returns expected value
        self.assertAlmostEqual(score_scaled * self.n_times, score_not_scaled)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Evaluate likelihoods for test parameters
        test_parameters = [2, 1]
        score_not_scaled = log_likelihood_not_scaled(test_parameters)
        score_scaled = log_likelihood_scaled(test_parameters)

        # Check that unscaled likelihood returns expected value
        self.assertEqual(int(score_not_scaled), -24999880)

        # Check that scaled likelihood returns expected value
        number_model_outputs = self.model_multi.n_outputs()
        self.assertAlmostEqual(
            score_scaled * self.n_times * number_model_outputs,
            score_not_scaled)

    def test_evaluateS1_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [0.014, 501]
        score_not_scaled, deriv_not_scaled = \
            log_likelihood_not_scaled.evaluateS1(test_parameters)
        score_scaled, deriv_scaled = log_likelihood_scaled.evaluateS1(
            test_parameters)

        # Check that score is computed correctly
        self.assertEqual(score_not_scaled, log_likelihood_not_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_not_scaled.shape, (2, ))

        # Check that score is computed correctly
        self.assertEqual(score_scaled, log_likelihood_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_scaled.shape, (2, ))

        # Check that partials of likelihoods agree
        unscaled_deriv = deriv_scaled * self.n_times
        self.assertAlmostEqual(deriv_not_scaled[0], unscaled_deriv[0])
        self.assertAlmostEqual(deriv_not_scaled[1], unscaled_deriv[1])

    def test_evaluateS1_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [0.014, 501]
        score_not_scaled, deriv_not_scaled = \
            log_likelihood_not_scaled.evaluateS1(test_parameters)
        score_scaled, deriv_scaled = log_likelihood_scaled.evaluateS1(
            test_parameters)

        # Check that score is computed correctly
        self.assertEqual(score_not_scaled, log_likelihood_not_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_not_scaled.shape, (2, ))

        # Check that score is computed correctly
        self.assertEqual(score_scaled, log_likelihood_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_scaled.shape, (2, ))

        # Check that partials of likelihoods agree
        unscaled_deriv = deriv_scaled * self.n_times
        self.assertAlmostEqual(deriv_not_scaled[0], unscaled_deriv[0])
        self.assertAlmostEqual(deriv_not_scaled[1], unscaled_deriv[1])

    def test_evaluateS1_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [0.014, 501]
        score_not_scaled, deriv_not_scaled = \
            log_likelihood_not_scaled.evaluateS1(test_parameters)
        score_scaled, deriv_scaled = log_likelihood_scaled.evaluateS1(
            test_parameters)

        # Check that score is computed correctly
        self.assertEqual(score_not_scaled, log_likelihood_not_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_not_scaled.shape, (2, ))

        # Check that score is computed correctly
        self.assertEqual(score_scaled, log_likelihood_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_scaled.shape, (2, ))

        # Check that partials of likelihoods agree
        unscaled_deriv = deriv_scaled * self.n_times
        self.assertAlmostEqual(deriv_not_scaled[0], unscaled_deriv[0])
        self.assertAlmostEqual(deriv_not_scaled[1], unscaled_deriv[1])

    def test_evaluateS1_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create a scaled and not scaled log_likelihood
        log_likelihood_not_scaled = pints.GaussianKnownSigmaLogLikelihood(
            problem, sigma=0.001)
        log_likelihood_scaled = pints.ScaledLogLikelihood(
            log_likelihood_not_scaled)

        # Compute derivatives for scaled and unscaled likelihood
        test_parameters = [2, 1]
        score_not_scaled, deriv_not_scaled = \
            log_likelihood_not_scaled.evaluateS1(test_parameters)
        score_scaled, deriv_scaled = log_likelihood_scaled.evaluateS1(
            test_parameters)

        # Check that score is computed correctly
        self.assertAlmostEqual(score_not_scaled, log_likelihood_not_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_not_scaled.shape, (2, ))

        # Check that score is computed correctly
        self.assertAlmostEqual(score_scaled, log_likelihood_scaled(
            test_parameters))

        # Check that partials have the correct shape
        self.assertEqual(deriv_scaled.shape, (2, ))

        # Check that partials of likelihoods agree
        number_model_outputs = self.model_multi.n_outputs()
        unscaled_deriv = deriv_scaled * self.n_times * number_model_outputs
        self.assertAlmostEqual(deriv_not_scaled[0], unscaled_deriv[0])
        self.assertAlmostEqual(deriv_not_scaled[1], unscaled_deriv[1])

    def test_bad_constructor(self):
        self.assertRaises(
            ValueError, pints.ScaledLogLikelihood, self.model_single)


class TestStudentTLogLikelihood(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test single output test model
        cls.model_single = pints.toy.ConstantModel(1)
        cls.model_multi = pints.toy.ConstantModel(4)

        # Generate test data
        cls.times = np.asarray([1, 2, 3])
        cls.n_times = len(cls.times)
        cls.data_single = np.asarray([1.0, -10.7, 15.5])
        cls.data_multi = np.asarray([
            [3.5, 7.6, 8.5, 3.4],
            [1.1, -10.3, 15.6, 5.5],
            [-10, -30.5, -5, 7.6]])

    def test_call_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.StudentTLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 3, 10]
        score = log_likelihood(test_parameters)

        # Check that scaled likelihood returns expected value
        self.assertAlmostEqual(score, -11.74010919785115)

    def test_call_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.StudentTLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 3, 10]
        score = log_likelihood(test_parameters)

        # Check that scaled likelihood returns expected value
        self.assertAlmostEqual(score, -11.74010919785115)

    def test_call_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        log_likelihood = pints.StudentTLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 3, 10]
        score = log_likelihood(test_parameters)

        # Check that scaled likelihood returns expected value
        self.assertAlmostEqual(score, -11.74010919785115)

    def test_call_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        log_likelihood = pints.StudentTLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        test_parameters = [0, 0, 0, 0, 2, 13, 1, 8, 2.5, 13.5, 3.4, 10.5]
        score = log_likelihood(test_parameters)

        # Check that scaled likelihood returns expected value
        self.assertAlmostEqual(score, -47.83720347766944)

    def test_evaluate_pointwise_loglikelihoods_list(self):
        # Convert data to list
        values = self.data_single.tolist()

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.StudentTLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [0, 3, 10]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_one_dim_array(self):
        # Convert data to array of shape (n_times,)
        values = np.reshape(self.data_single, (self.n_times,))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.StudentTLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [0, 3, 10]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_single(self):
        # Convert data to array of shape (n_times, 1)
        values = np.reshape(self.data_single, (self.n_times, 1))

        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(
            self.model_single, self.times, values)

        # Create log_likelihood
        likelihood = pints.StudentTLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [0, 3, 10]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, (self.n_times,))

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))

    def test_evaluate_pointwise_loglikelihoods_two_dim_array_multi(self):
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(
            self.model_multi, self.times, self.data_multi)

        # Create log_likelihood
        likelihood = pints.StudentTLogLikelihood(problem)

        # Evaluate likelihood for test parameters
        parameters = [0, 0, 0, 0, 2, 13, 1, 8, 2.5, 13.5, 3.4, 10.5]
        pointwise = likelihood.evaluate_pointwise_loglikelihoods(parameters)

        # Check that the loglikelihoods match the shape of the data
        self.assertEqual(pointwise.shape, self.data_multi.shape)

        # Check that the sum of the pointwise loglikelihoods agrees with call
        # There are floating point deviations because the summation occurs at
        # different points in __call__()
        self.assertAlmostEqual(np.sum(pointwise), likelihood(parameters))


if __name__ == '__main__':
    unittest.main()

#pragma once

extern "C"
{
	/**
	Expectation-maximisation algorithm with Gaussian distributions.

	@param number_components Number of Gaussian components.
	@param number_dimensions Number of features (dimensionality of data).
	@param sample_size Number of data points.
	@param data[in] Data points array (one after another) of size number_dimensions * sample_size.
	@param means_initialiser Optional name of means initialisation algorithm.
	@param responsibilities_initialiser Optional name of responsibilities initialisation algorithm.
	@param responsibilities[out] Calculated responsibilities array (one data point after another) of size number_components * sample_size. Ignored if NULL.
	@param means[out] Calculated means array (one component after another) of size number_components * sample_size. Ignored if NULL.
	@param covariances[out] Calculated covariances array (one covariance matrix after another) of size number_components * number_dimensions * number_dimensions. Ignored if NULL.
	@return Null pointer if success (converged), error message description otherwise.
	*/
	const char* em_fit(
		unsigned int number_components,
		unsigned int number_dimensions,
		unsigned int sample_size,
		const double* data,
		const char* means_initialiser,
		const char* responsibilities_initialiser,
		const unsigned int* seed,
		const double absolute_tolerance,
		const double relative_tolerance,
		unsigned int maximum_steps,
		const bool maximise_first,
		double* responsibilities,
		double* means,
		double* covariances
	);
}
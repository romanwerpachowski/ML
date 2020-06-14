#include <cstring>
#include "ML/EM.hpp"
#include "EM.h"

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
	)
{	
	ml::EM em(number_components);
	if (means_initialiser) {
		if (!strcmp(means_initialiser, "forgy")) {
			em.set_means_initialiser(std::make_shared<ml::EM::Forgy>());
		} else if (!strcmp(means_initialiser, "random_partition")) {
			em.set_means_initialiser(std::make_shared<ml::EM::RandomPartition>());
		} else if (!strcmp(means_initialiser, "kpp")) {
			em.set_means_initialiser(std::make_shared<ml::EM::KPP>());
		} else {
			return "Unknown means initialiser";
		}
	}
	if (responsibilities_initialiser) {
		if (!strcmp(responsibilities_initialiser, "closest_mean")) {
			em.set_responsibilities_initialiser(std::make_shared<ml::EM::ClosestMean>(em.means_initialiser()));
		} else {
			return "Unknown responsibilities initialiser";
		}
	}
	if (seed) {
		em.set_seed(*seed);
	}
	em.set_absolute_tolerance(absolute_tolerance);
	em.set_relative_tolerance(relative_tolerance);
	em.set_maximum_steps(maximum_steps);
	em.set_maximise_first(maximise_first);

	Eigen::Map<const Eigen::MatrixXd> X(data, number_dimensions, sample_size);
	const bool converged = em.fit(X);

	if (responsibilities) {
		memcpy(responsibilities, em.responsibilities().data(), sizeof(double) * sample_size * number_components);
	}
	if (means) {
		memcpy(means, em.means().data(), sizeof(double) * number_components * sample_size);
	}
	if (covariances) {
		const auto cov_len = sizeof(double) * number_dimensions * number_dimensions;
		for (unsigned int i = 0; i < number_components; ++i) {
			memcpy(covariances + i * cov_len, em.covariances()[i].data(), cov_len);
		}
	}

	if (converged) {
		return NULL;
	} else {
		return "Did not converge";
	}
}
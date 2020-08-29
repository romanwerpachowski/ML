/* (C) 2020 Roman Werpachowski. */
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include "ML/Clustering.hpp"
#include "ML/EM.hpp"

static constexpr double PI = 3.14159265358979323846;

void em_demo(void)
{
	std::default_random_engine rng;
	std::uniform_real_distribution<double> u01(0, 1);

	const unsigned int num_dimensions = 2;
	const unsigned int sample_size = 1000;
	const double face_radius = 1;
	const double ear_radius = 0.3;
	const unsigned int num_components = 3; // face, left ear, right ear
	const std::vector<double> radii{ face_radius, ear_radius, ear_radius };
	std::discrete_distribution<unsigned int> component_distr{ face_radius * face_radius, 2 * ear_radius * ear_radius, 2 * ear_radius * ear_radius };
	const double ear_angle = 45 * PI / 180.; // how far is ear from top of head
	std::vector<double> center_xs{ 0, -(face_radius + ear_radius) * std::sin(ear_angle), (face_radius + ear_radius) * std::sin(ear_angle) };
	std::vector<double> center_ys{ 0, (face_radius + ear_radius) * std::cos(ear_angle), (face_radius + ear_radius) * std::cos(ear_angle) };

	Eigen::MatrixXd data(num_dimensions, sample_size);
	std::vector<unsigned int> classes(sample_size);
	for (unsigned int i = 0; i < sample_size; ++i) {
		const unsigned int k = component_distr(rng);
		classes[i] = k;
		const double phi = 2 * PI * u01(rng);
		const double r = std::sqrt(u01(rng)) * radii[k];
		data(0, i) = center_xs[k] + r * std::cos(phi);
		data(1, i) = center_ys[k] + r * std::sin(phi);
	}

	ml::EM em(num_components);
	em.set_absolute_tolerance(1e-14);
	em.set_relative_tolerance(1e-14);
	em.set_means_initialiser(std::make_shared<ml::Clustering::KPP>());
	em.set_maximise_first(false);
	bool converged;
	for (int i = 0; i < 100; ++i) {
		converged = em.fit(data);
	}	

	std::cout << "Converged: " << converged << "\n";
	std::cout << "Log-likelihood: " << em.log_likelihood() << "\n";
	std::cout << "Means:\n" << em.means() << "\n";
	for (unsigned int k = 0; k < num_components; ++k) {
		std::cout << "Covariance[" << k << "]:\n" << em.covariance(k) << "\n";
	}

	std::ofstream file("mousie.csv");
	file << "X,Y,Class,P_face,P_left_ear,P_right_ear\n";
	for (unsigned int i = 0; i < sample_size; ++i) {
		file << data(0, i) << "," << data(1, i) << "," << classes[i] << "," << em.responsibilities()(i, 0) << "," << em.responsibilities()(i, 1) << "," << em.responsibilities()(i, 2) << "\n";
	}
}
/* (C) 2020 Roman Werpachowski. */
#include <benchmark/benchmark.h>
#include <random>
#include "ML/Clustering.hpp"
#include "ML/EM.hpp"

static constexpr double PI = 3.14159265358979323846;

static void em_mousie(benchmark::State& state)
{
	std::default_random_engine rng;
	std::uniform_real_distribution<double> u01(0, 1);

	const unsigned int num_dimensions = 2;
	const int64_t sample_size = state.range(0);
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
	for (int64_t i = 0; i < sample_size; ++i) {
		const unsigned int k = component_distr(rng);
		classes[i] = k;
		const double phi = 2 * PI * u01(rng);
		const double r = std::sqrt(u01(rng)) * radii[k];
		data(0, i) = center_xs[k] + r * std::cos(phi);
		data(1, i) = center_ys[k] + r * std::sin(phi);
	}
	
	// Benchmarked code.
	for (auto _ : state) {
		ml::EM em(num_components);
		em.set_absolute_tolerance(1e-14);
		em.set_relative_tolerance(1e-14);
		em.set_means_initialiser(std::make_shared<ml::Clustering::KPP>());
		em.set_maximise_first(false);
		em.fit(data);
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(em_mousie)->RangeMultiplier(10)->Range(100, 100000)->Complexity();
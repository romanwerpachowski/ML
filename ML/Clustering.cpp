#include <cassert>
#include <numeric>
#include "Clustering.hpp"

namespace ml
{
	namespace Clustering
	{
		MeansInitialiser::~MeansInitialiser()
		{}

		void Forgy::init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, const unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const
		{
			std::vector<Eigen::Index> all_indices(data.cols());
			std::iota(all_indices.begin(), all_indices.end(), 0);
			std::vector<Eigen::Index> sampled_indices;
			std::sample(all_indices.begin(), all_indices.end(), std::back_inserter(sampled_indices), number_components, prng);
			for (unsigned int i = 0; i < number_components; ++i) {
				means.col(i) = data.col(sampled_indices[i]);
			}
		}

		void RandomPartition::init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, const unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const
		{
			means.setZero();
			std::vector<unsigned int> counters(number_components, 0);
			std::uniform_int_distribution<unsigned int> dist(0, number_components - 1);
			for (Eigen::Index i = 0; i < data.cols(); ++i) {
				const auto k = dist(prng);
				means.col(k) += (data.col(i) - means.col(k)) / static_cast<double>(++counters[k]);
			}
			assert(std::accumulate(counters.begin(), counters.end(), 0) == data.cols());
		}

		void KPP::init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, const unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> means) const
		{
			std::vector<double> weights(data.cols());
			for (unsigned int n = 0; n < number_components; ++n) {
				if (n) {
					for (Eigen::Index i = 0; i < data.cols(); ++i) {
						double min_distance_squared = std::numeric_limits<double>::infinity();
						for (unsigned int k = 0; k < n; ++k) {
							const double distance_squared = (data.col(i) - means.col(k)).squaredNorm();
							min_distance_squared = std::min(min_distance_squared, distance_squared);
						}
						weights[i] = min_distance_squared;
					}
				} else {
					std::fill(weights.begin(), weights.end(), 1);
				}
				std::discrete_distribution<Eigen::Index> dist(weights.begin(), weights.end());
				const auto new_mean_idx = dist(prng);
				means.col(n) = data.col(new_mean_idx);
			}
		}

		ResponsibilitiesInitialiser::~ResponsibilitiesInitialiser()
		{}

		ClosestMean::ClosestMean(std::shared_ptr<const MeansInitialiser> means_initialiser)
			: means_initialiser_(means_initialiser)
		{
			if (!means_initialiser) {
				throw std::invalid_argument("Null means initialiser");
			}
		}

		void ClosestMean::init(Eigen::Ref<const Eigen::MatrixXd> data, std::default_random_engine& prng, unsigned int number_components, Eigen::Ref<Eigen::MatrixXd> responsibilities) const
		{
			Eigen::MatrixXd means(data.rows(), number_components);
			means_initialiser_->init(data, prng, number_components, means);
			responsibilities.setZero();
			for (Eigen::Index i = 0; i < data.cols(); ++i) {
				double min_distance_squared = (data.col(i) - means.col(0)).squaredNorm();
				unsigned int closest_mean_index = 0;
				for (unsigned int k = 1; k < number_components; ++k) {
					const double distance_squared = (data.col(i) - means.col(k)).squaredNorm();
					if (distance_squared < min_distance_squared) {
						min_distance_squared = distance_squared;
						closest_mean_index = k;
					}
				}
				responsibilities(i, closest_mean_index) = 1;
			}
		}
	}
}
/* (C) 2021 Roman Werpachowski. */
#include "KMeans.hpp"


namespace ml
{
    namespace Clustering
    {
		KMeans::KMeans(unsigned int number_clusters)
			: num_clusters_(number_clusters)
		{
			if (!number_clusters) {
				throw std::invalid_argument("KMeans: number of clusters cannot be zero");
			}
		}

		bool KMeans::fit(Eigen::Ref<const Eigen::MatrixXd> data)
		{
			const auto number_dimensions = static_cast<unsigned int>(data.rows());
			const auto sample_size = static_cast<unsigned int>(data.cols());
			if (!number_dimensions) {
				throw std::invalid_argument("KMeans: At least one dimension required");
			}
			if (sample_size < num_clusters_) {
				throw std::invalid_argument("KMeans: Not enough data ");
			}

			centroids_.resize(number_dimensions, num_clusters_);
			labels_.resize(sample_size);

			if (sample_size == num_clusters_) {
				// An exact deterministic fit is possible.
				// Center each cluster on a different sample point.
				for (unsigned int i = 0; i < sample_size; ++i) {
					centroids_.col(i) = data.col(i);
					labels_[i] = i;
				}
				return true;
			}

			return false;
		}

        void KMeans::set_seed(unsigned int seed)
        {
            prng_.seed(seed);
        }

		void KMeans::set_absolute_tolerance(double absolute_tolerance)
		{
			if (absolute_tolerance < 0) {
				throw std::domain_error("KMeans: Negative absolute tolerance");
			}
			absolute_tolerance_ = absolute_tolerance;
		}

		void KMeans::set_relative_tolerance(double relative_tolerance)
		{
			if (relative_tolerance < 0) {
				throw std::domain_error("KMeans: Negative relative tolerance");
			}
			relative_tolerance_ = relative_tolerance;
		}

		void KMeans::set_maximum_steps(unsigned int maximum_steps)
		{
			if (maximum_steps < 2) {
				throw std::invalid_argument("KMeans: At least two steps required for convergence test");
			}
			maximum_steps_ = maximum_steps;
		}

		void KMeans::set_centroids_initialiser(std::shared_ptr<const Clustering::CentroidsInitialiser> centroids_initialiser)
		{
			if (!centroids_initialiser) {
				throw std::invalid_argument("KMeans: Null centroids initialiser");
			}
			centroids_initialiser_ = centroids_initialiser;
		}
    }
}
/* (C) 2021 Roman Werpachowski. */
#include <cassert>
#include <iostream>
#include "KMeans.hpp"

namespace ml
{
    namespace Clustering
    {
		KMeans::KMeans(unsigned int number_clusters)
			: num_clusters_(number_clusters), work_vector_(number_clusters)
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
			old_centroids_.resize(number_dimensions, num_clusters_);
			labels_.resize(sample_size);
			old_labels_.resize(sample_size);			

			if (sample_size == num_clusters_) {
				// An exact deterministic fit is possible.
				// Center each cluster on a different sample point.
				for (unsigned int i = 0; i < sample_size; ++i) {
					centroids_.col(i) = data.col(i);
					labels_[i] = i;
				}
				return true;
			}

			centroids_initialiser_->init(data, prng_, num_clusters_, centroids_);

			// Main iteration loop.
			for (unsigned int step = 0; step < maximum_steps_; ++step) {
				// Aka "expectation step" in the E-M terminology.
				assignment_step(data);

				if (step > 0) {
					if (old_labels_ == labels_) {
						return true;
					}
				}

				// Aka "maximisation step" in the E-M terminology.
				update_step(data);

				if (verbose_) {
					std::cout << "Step " << step << "\n";
					for (unsigned int k = 0; k < num_clusters_; ++k) {
						std::cout << "Centroid[" << k << "] == " << centroids_.col(k).transpose() << "\n";
					}
					std::cout << std::endl;
				}

				if (step > 0) {					
					const double ssq_change = (centroids_ - old_centroids_).squaredNorm();
					if (ssq_change < absolute_tolerance_) {
						assignment_step(data);
						return true;
					}
				}
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

		void KMeans::assignment_step(Eigen::Ref<const Eigen::MatrixXd> data)
		{
			const auto sample_size = data.cols();
			assert(labels_.size() == static_cast<size_t>(sample_size));
			old_labels_.swap(labels_); // Save previoous labels.
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				double min_squared_distance = std::numeric_limits<double>::infinity();
				unsigned int label = 0;
				const auto point = data.col(i);
				for (unsigned int k = 0; k < num_clusters_; ++k) {
					const auto squared_distance = (point - centroids_.col(k)).squaredNorm();
					if (squared_distance < min_squared_distance) {
						min_squared_distance = squared_distance;
						label = k;
					}
				}
				labels_[i] = label;
			}
		}

		void KMeans::update_step(Eigen::Ref<const Eigen::MatrixXd> data)
		{
			work_vector_.setZero();
			old_centroids_.swap(centroids_);
			centroids_.setZero();
			const auto sample_size = data.cols();
			// Update centroids.
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				const unsigned int label = labels_[i];
				work_vector_[i] += 1.;
				centroids_.col(label) += (data.col(i) - centroids_.col(label)) / work_vector_[i];
			}
		}
    }
}
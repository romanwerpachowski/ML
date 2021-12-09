/* (C) 2021 Roman Werpachowski. */
#include <cassert>
#include <iostream>
#include "KMeans.hpp"

namespace ml
{
    namespace Clustering
    {
		KMeans::KMeans(unsigned int number_clusters)
			: work_vector_(number_clusters)
			, centroids_initialiser_(std::make_shared<Clustering::Forgy>())
			, absolute_tolerance_(1e-8)
			, maximum_steps_(1000)
			, num_inits_(1)
			, num_clusters_(number_clusters)
			, verbose_(false)
			, converged_(false)
		{
			if (!number_clusters) {
				throw std::invalid_argument("KMeans: number of clusters cannot be zero");
			}
		}

		bool KMeans::fit(Eigen::Ref<const Eigen::MatrixXd> data)
		{
			if (num_inits_ == 1) {
				return fit_once(data);
			} else {
				converged_ = false;
				double min_inertia = std::numeric_limits<double>::infinity();
				Eigen::MatrixXd best_centroids;				
				for (unsigned int i = 0; i < num_inits_; ++i) {
					if (fit_once(data)) {
						if (inertia_ < min_inertia) {
							min_inertia = inertia_;
							best_centroids = centroids_;
						}
						converged_ = true;
					}
				}
				if (converged_) {
					centroids_ = best_centroids;
					assignment_step(data);
				}
				return converged_;
			}
		}

		bool KMeans::fit_once(Eigen::Ref<const Eigen::MatrixXd> data)
		{
			converged_ = false;
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
				inertia_ = 0;
				converged_ = true;				
			} else {
				centroids_initialiser_->init(data, prng_, num_clusters_, centroids_);

				// Main iteration loop.
				for (unsigned int step = 0; step < maximum_steps_; ++step) {
					// Aka "expectation step" in the E-M terminology.
					assignment_step(data);

					if (step > 0) {
						if (old_labels_ == labels_) {
							converged_ = true;
							break;
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
						const double centroid_shift = (centroids_ - old_centroids_).squaredNorm();
						if (centroid_shift < absolute_tolerance_) {
							assignment_step(data);
							converged_ = true;
							break;
						}
					}
				}
			}

			return converged_;
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

		void KMeans::set_number_initialisations(unsigned int number_initialisations)
		{
			if (number_initialisations < 1) {
				throw std::invalid_argument("KMeans: At least 1 initialisation required");
			}
			num_inits_ = number_initialisations;
		}

		void KMeans::set_centroids_initialiser(std::shared_ptr<const Clustering::CentroidsInitialiser> centroids_initialiser)
		{
			if (!centroids_initialiser) {
				throw std::invalid_argument("KMeans: Null centroids initialiser");
			}
			centroids_initialiser_ = centroids_initialiser;
		}

		std::pair<unsigned int, double> KMeans::assign_label(const Eigen::Ref<const Eigen::VectorXd> x) const
		{
			double min_squared_distance = std::numeric_limits<double>::infinity();
			unsigned int label = 0;
			for (unsigned int k = 0; k < num_clusters_; ++k) {
				const auto squared_distance = (x - centroids_.col(k)).squaredNorm();
				if (squared_distance < min_squared_distance) {
					min_squared_distance = squared_distance;
					label = k;
				}
			}
			return std::make_pair(label, min_squared_distance);
		}

		void KMeans::assignment_step(Eigen::Ref<const Eigen::MatrixXd> data)
		{
			const auto sample_size = data.cols();
			assert(labels_.size() == static_cast<size_t>(sample_size));
			old_labels_.swap(labels_); // Save previoous labels.
			inertia_ = 0;
			for (Eigen::Index i = 0; i < sample_size; ++i) {
				const auto label_and_distance = assign_label(data.col(i));
				labels_[i] = label_and_distance.first;
				inertia_ += label_and_distance.second;
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
				const double num = (++work_vector_[label]);
				centroids_.col(label) += (data.col(i) - centroids_.col(label)) / num;
			}
		}
    }
}
#pragma once
/* (C) 2021 Roman Werpachowski. */
#include "Clustering.hpp"
#include <vector>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml
{
    namespace Clustering
    {
        /**
         * @brief K-means clustering method.
         * 
         * Converges if exactly the same cluster assignments are chosen twice, or if sum of squared differences between new and old centroids is lower than tolerance.
        */
        class KMeans : public Model
        {
        public:
            /** @brief Constructs a K-means model ready to fit.
            @param[in] number_clusters Number of clusters.
            @throw std::invalid_argument If `number_clusters == 0`.
            */
            DLL_DECLSPEC KMeans(unsigned int number_clusters);

            DLL_DECLSPEC bool fit(Eigen::Ref<const Eigen::MatrixXd> data) override;

            unsigned int number_clusters() const override
            {
                return num_clusters_;
            }

            const std::vector<unsigned int>& labels() const override
            {
                return labels_;
            }

            const Eigen::MatrixXd& centroids() const override
            {
                return centroids_;
            }

            /** @brief Sets PRNG seed.
            @param[in] seed PRNG seed.
            */
            DLL_DECLSPEC void set_seed(unsigned int seed);

            /** @brief Sets absolute tolerance for convergence test: || old centroids - new centroids ||^2 < absolute tolerance.
            @param[in] absolute_tolerance Absolute tolerance.
            @throw std::domain_error If `absolute_tolerance < 0`.
            */
            DLL_DECLSPEC void set_absolute_tolerance(double absolute_tolerance);

            /** @brief Sets maximum number of K-means steps.
            @param[in] maximum_steps Maximum number of steps.
            @throw std::invalid_argument If `maximum_steps < 1`.
            */
            DLL_DECLSPEC void set_maximum_steps(unsigned int maximum_steps);

            /** @brief Sets centroids initialiser.
            @param[in] centroids_initialiser Pointer to CentroidsInitialiser implementation.
            @throw std::invalid_argument If `centroids_initialiser` is null.
            */
            DLL_DECLSPEC void set_centroids_initialiser(std::shared_ptr<const CentroidsInitialiser> centroids_initialiser);

            /** @brief Switches between verbose and quiet mode.
            @param[in] verbose `true` if we want verbose output.
            */
            void set_verbose(bool verbose)
            {
                verbose_ = verbose;
            }

            /** @brief Given a data point x, assign it to its cluster and return the correct label.

            @param[in] x Data point with correct dimension.
            @throw std::invalid_argument If `x.size() != means().rows()`.
            */
            DLL_DECLSPEC unsigned int assign_label(Eigen::Ref<const Eigen::VectorXd> x) const;
        private:
            std::vector<unsigned int> labels_;
            std::vector<unsigned int> old_labels_;
            Eigen::MatrixXd centroids_;
            Eigen::MatrixXd old_centroids_;
            Eigen::VectorXd work_vector_;
            std::default_random_engine prng_;
            std::shared_ptr<const CentroidsInitialiser> centroids_initialiser_;
            double absolute_tolerance_;
            unsigned int maximum_steps_;
            unsigned int num_inits_;
            unsigned int num_clusters_;
            bool verbose_;

            /// Assigns points to centroids.
            void assignment_step(Eigen::Ref<const Eigen::MatrixXd> data);

            /// Updates positions of centroids.
            void update_step(Eigen::Ref<const Eigen::MatrixXd> data);
        };
    }
}
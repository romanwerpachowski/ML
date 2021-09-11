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
        */
        class KMeans : public Model
        {
        public:
            /** @brief Constructs a K-means model ready to fit.
            @param[in] number_clusters Number of clusters.
            @throw If `number_clusters == 0`.
            */
            DLL_DECLSPEC KMeans(unsigned int number_clusters);

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

            /** @brief Sets absolute tolerance for convergence test.
            @param[in] absolute_tolerance Absolute tolerance.
            @throw std::domain_error If `absolute_tolerance < 0`.
            */
            DLL_DECLSPEC void set_absolute_tolerance(double absolute_tolerance);

            /** @brief Sets relative tolerance for convergence test.
            @param[in] relative_tolerance Relative tolerance.
            @throw std::domain_error If `relative_tolerance < 0`.
            */
            DLL_DECLSPEC void set_relative_tolerance(double relative_tolerance);

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
        private:
            std::vector<unsigned int> labels_;
            Eigen::MatrixXd centroids_;
            std::default_random_engine prng_;
            std::shared_ptr<const CentroidsInitialiser> centroids_initialiser_;
            double absolute_tolerance_;
            double relative_tolerance_;
            unsigned int maximum_steps_;
            unsigned int num_inits_;
            unsigned int num_clusters_;
            bool verbose_;
        };
    }
}
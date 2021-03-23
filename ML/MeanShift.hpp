#pragma once
/* (C) 2021 Roman Werpachowski. */
#include <memory>
#include <Eigen/Core>
#include "Clustering.hpp"

namespace ml
{
    namespace Kernels
    {
        class DifferentiableRadialBasisFunction;
    }

    namespace Clustering
    {
        /**
         * @brief Mean shift clustering model.
         * 
         * Based on https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/TUZEL1/MeanShift.pdf and D. Comaniciu and P. Meer. Mean shift: A robust approach toward feature space analysis. IEEE Trans. Pattern Anal. Machine Intell., 24:603–619, 2002.
        */
        class MeanShift : public Model
        {
        public:
            /**
             * @brief Constructor.
             * 
             * @param rbf Radial basis function defining the kernel. Assumed to be monotonically decreasing with distance.
             * @param h Window radius.
             * 
             * @throw std::invalid_argument If `rbf` is null.
             * @throw std::domain_error If `h <= 0`.
            */
            DLL_DECLSPEC MeanShift(std::shared_ptr<const Kernels::DifferentiableRadialBasisFunction> rbf, double h);

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

            /**
             * @brief Fits the model to data.
             * @param data Matrix of feature vectors, with data points in columns.
             * @return True if the fit converged.
            */
            DLL_DECLSPEC bool fit(Eigen::Ref<const Eigen::MatrixXd> data) override;

            /**
             * @brief Number of clusters found.
            */
            unsigned int number_clusters() const override
            {
                return number_clusters_;
            }

            /**
             * @brief Const reference to cluster labels assigned to data points.
            */
            const std::vector<unsigned int>& labels() const override
            {
                return labels_;
            }
        private:
            std::vector<unsigned int> labels_;            
            std::shared_ptr<const Kernels::DifferentiableRadialBasisFunction> rbf_;
            double h_; /**< Window radius. */
            double h2_; /**< Window radius (squared). */
            double absolute_tolerance_;
            double relative_tolerance_;
            double perturbation_strength_;            
            double mode_identification_absolute_tolerance_;
            unsigned int number_clusters_;

            static bool close_within_tolerance(Eigen::Ref<const Eigen::VectorXd> x1, Eigen::Ref<const Eigen::VectorXd> x2, double absolute_tolerance, double relative_tolerance);

            bool close_within_tolerance(Eigen::Ref<const Eigen::VectorXd> x1, Eigen::Ref<const Eigen::VectorXd> x2) const;

            void calc_new_position(Eigen::Ref<const Eigen::MatrixXd> data, Eigen::Ref<const Eigen::VectorXd> old_pos, Eigen::Ref<Eigen::VectorXd> new_pos) const;

            void shift_until_stationary(Eigen::Ref<const Eigen::MatrixXd> data, Eigen::Ref<Eigen::VectorXd> pos, Eigen::Ref<Eigen::VectorXd> work) const;
        };
    }
}

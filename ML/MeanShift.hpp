#pragma once
/* (C) 2021 Roman Werpachowski. */
#include <memory>
#include <Eigen/Core>
#include "Clustering.hpp"

namespace ml
{
    namespace Kernels
    {
        class DoubleDifferentiableRadialBasisFunction;
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
             * @param rbf Radial basis function defining the kernel.
             * @param h Window radius.
             * 
             * @throw std::invalid_argument If `rbf` is null.
             * @throw std::domain_error If `h <= 0`.
            */
            MeanShift(std::shared_ptr<Kernels::DoubleDifferentiableRadialBasisFunction> rbf, double h);

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

            bool fit(Eigen::Ref<const Eigen::MatrixXd> data) override;

            unsigned int number_clusters() const override
            {
                return number_clusters_;
            }

            const std::vector<unsigned int>& labels() const override
            {
                return labels_;
            }
        private:
            std::vector<unsigned int> labels_;
            std::shared_ptr<Kernels::DoubleDifferentiableRadialBasisFunction> rbf_;
            double h2_; /**< Window radius. */            
            double absolute_tolerance_;
            double relative_tolerance_;
            unsigned int number_clusters_;
        };
    }
}

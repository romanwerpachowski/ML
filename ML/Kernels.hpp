#pragma once
/* (C) 2021 Roman Werpachowski. */
#include <memory>
#include <type_traits>
#include <Eigen/Core>
#include "dll.hpp"

namespace ml
{
    /**
     * @brief Methods and classes for working with kernels.
    */
    namespace Kernels
    {
        /**
         * @brief Abstract R^D kernel interface.
         * 
         * A kernel is a symmetric, positive-definite function \f$ R^D \times R^D \rightarrow R \f$.
        */
        class Kernel
        {
        public:
            /** @brief Virtual destructor. */
            DLL_DECLSPEC virtual ~Kernel();

            /**
             * @brief Value of the kernel \f$ K(\vec{x}_1, \vec{x}_2) \f$.
             * 
             * @param[in] x1 First feature vector.
             * @param[in] x2 Second feature vector.
             * @return Kernel value.
             * @throw std::invalid_argument If `x1.size() != #dim()` or `x2.size() != #dim()`.
            */
            DLL_DECLSPEC virtual double value(const Eigen::Ref<const Eigen::VectorXd> x1, const Eigen::Ref<const Eigen::VectorXd> x2) const = 0;

            /**
             * @brief Dimension of the feature space.
            */
            virtual unsigned int dim() const = 0;
        protected:
            DLL_DECLSPEC void validate_arguments(const Eigen::Ref<const Eigen::VectorXd> x1, const Eigen::Ref<const Eigen::VectorXd> x2) const;
        };

        /**
         * @brief Abstract differentiable R^D kernel interface.
        */
        class DifferentiableKernel: public virtual Kernel
        {
            /**
            * @brief Gradient of the kernel \f$ K(\vec{x}_1, \vec{x}_2) \f$ over the first feature vector. The gradient over the second vector can be calculated by swapping them, due to the symmetry of the kernel function.
            * @param[in] x1 First feature vector.
            * @param[in] x2 Second feature vector.
            * @param[out] Gradient over x1. Must have the same size as x1 and x2.
            * @throw std::invalid_argument If `x1.size() != #dim()`, `x2.size() != #dim()` or `dydx1.size() != #dim()`.
            */
            DLL_DECLSPEC virtual void gradient(const Eigen::Ref<const Eigen::VectorXd> x1, const Eigen::Ref<const Eigen::VectorXd> x2, Eigen::Ref<Eigen::VectorXd> dydx1) const = 0;
        };

        
        /**
         * @brief Abstract double differentiable R^D kernel interface.
        */
        class DoubleDifferentiableKernel : public virtual DifferentiableKernel
        {
            /**
             * @brief Hessian of the kernel \f$ K(\vec{x}_1, \vec{x}_2) \f$ over both vectors concatenated \f$ \vec{x}_1 \oplus \vec{x}_2 \f$.
             * @param[in] x1 First feature vector.
             * @param[in] x2 Second feature vector.
             * @param[out] H Hessian matrix. Upon return is filled with second derivatives.
             * @throw std::invalid_argument If `x1.size() != #dim()`, `x2.size() != #dim()` or `H.rows() != 2 * dim()` or `H.cols() != H.rows()`.
            */
            DLL_DECLSPEC virtual void hessian(const Eigen::Ref<const Eigen::VectorXd> x1, const Eigen::Ref<const Eigen::VectorXd> x2, Eigen::Ref<Eigen::MatrixXd> H) const = 0;
        };

        /**
         * @brief Radial basis function.
        */
        class RadialBasisFunction
        {
        public:
            /** @brief Virtual destructor. */
            DLL_DECLSPEC virtual ~RadialBasisFunction();

            /**
             * @brief Radial basis function of the RBF kernel.
             * @param r2 Square of the L2 norm of the difference between two feature vectors.
             * @return Kernel value.
             * @throw std::domain_error If r2 < 0.
            */
            DLL_DECLSPEC virtual double value(double r2) const = 0;
        };

        /**
         * @brief Differentiable radial basis function kernel.
        */
        class DifferentiableRadialBasisFunction : public RadialBasisFunction
        {
        public:
            /**
             * @brief Gradient of the radial basis function of the RBF kernel.
             * @param r2 Square of the L2 norm of the difference between two feature vectors.
             * @return Kernel value.
             * @throw std::domain_error If r2 < 0.
            */
            DLL_DECLSPEC virtual double gradient(double r2) const = 0;
        };

        /**
         * @brief Double differentiable radial basis function kernel.
        */
        class DoubleDifferentiableRadialBasisFunction : public DifferentiableRadialBasisFunction
        {
        public:
            /**
             * @brief Second derivative of the radial basis function of the RBF kernel.
             * @param r2 Square of the L2 norm of the difference between two feature vectors.
             * @return Kernel value.
             * @throw std::domain_error If r2 < 0.
            */
            DLL_DECLSPEC virtual double second_derivative(double r2) const = 0;
        };

        /**
         * @brief Gaussian radial basis function.
         * 
         * Given by the formula f(r2) = exp(-r2), where r2 is the SQUARE of the norm.
        */
        class GaussianRBF: public DoubleDifferentiableRadialBasisFunction
        {
        public:
            DLL_DECLSPEC double value(double r2) const override;
            DLL_DECLSPEC double gradient(double r2) const override;
            DLL_DECLSPEC double second_derivative(double r2) const override;
        };

        /**
         * @brief Radial basis function kernel.
         *
         * Kernel function of the form K(x1, x2) = s(||x1 - x2||^2).
         *
         * See https://en.wikipedia.org/wiki/Radial_basis_function_kernel
         * 
         * @tparam RBF RadialBasisFunction or its child class.
        */
        template <class RBF = RadialBasisFunction> class RBFKernel : public virtual Kernel
        {
            static_assert(std::is_base_of_v<RadialBasisFunction, RBF>);
        public:
            /**
             * @brief Constructor.
             * @param rbf Radial basis function, moved.
             * @param dim Dimension of the kernel, positive.
             * @throw std::invalid_argument If `rbf` is null.
             * @throw std::domain_error If `dim` is zero.
            */
            RBFKernel(std::unique_ptr<const RBF>&& rbf, unsigned int dim)
                : rbf_(std::move(rbf)), dim_(dim)
            {
                if (rbf_ == nullptr) {
                    throw std::invalid_argument("Null RBF object");
                }
                if (!dim) {
                    throw std::domain_error("Kernel dimension must be positive");
                }
            }

            RBFKernel(const RBFKernel& other) = delete;
            RBFKernel& operator=(const RBFKernel& other) = delete;

            DLL_DECLSPEC RBFKernel(RBFKernel&& other) = default;
            DLL_DECLSPEC RBFKernel& operator=(RBFKernel&& other) = default;

            double value(const Eigen::Ref<const Eigen::VectorXd> x1, const Eigen::Ref<const Eigen::VectorXd> x2) const override
            {
                validate_arguments(x1, x2);
                return rbf_->value((x1 - x2).squaredNorm());
            }

            unsigned int dim() const override
            {
                return dim_;
            }
        protected:
            std::unique_ptr<const RBF> rbf_;
        private:
            unsigned int dim_;
        };

        /**
         * @brief Differentiable radial basis function kernel.
         * @tparam DiffRBF DifferentiableRadialBasisFunction or its child class.
        */
        template <class DiffRBF = DifferentiableRadialBasisFunction> class DifferentiableRBFKernel : public virtual DifferentiableKernel, public RBFKernel<DiffRBF>
        {
            static_assert(std::is_base_of_v<DifferentiableRadialBasisFunction, DiffRBF>);
        public:
            using RBFKernel<DiffRBF>::RBFKernel;

            void gradient(const Eigen::Ref<const Eigen::VectorXd> x1, const Eigen::Ref<const Eigen::VectorXd> x2, Eigen::Ref<Eigen::VectorXd> dydx1) const override
            {
                this->validate_arguments(x1, x2);
                if (dydx1.size() != this->dim()) {
                    throw std::invalid_argument("Wrong dimension of dydx1");
                }
                const double rbf1der = this->rbf_->gradient((x1 - x2).squaredNorm());
                for (Eigen::Index i = 0; i < x1.size(); ++i) {
                    dydx1[i] = 2 * (x1[i] - x2[i]) * rbf1der;
                }
            }
        };        
    }
}

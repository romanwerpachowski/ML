#pragma once
/* (C) 2021 Roman Werpachowski */
#include <Eigen/Core>
#include "dll.hpp"

namespace ml
{
    /** @brief Binomial logistic regression algorithm.
    *
    * Based on Thomas P. Minka, "A comparison of numerical optimizers for logistic regression".
    * 
    * The model is:
    * 
    * \f$ P(y = +/-1 | \vec{x}, \vec{w}) = \frac{1}{1 + \exp(-y \vec{w} \cdot \vec{x} )} \f$
    * 
    * We perform a MAP estimation of \f$ \vec{w} \f$, using a prior \f$ p(\vec{w}) \sim N(0, \lambda^{-1} I) \f$.
    * 
    * Given a dataset \f$ [(\vec{x}_i, y_i)]_{i=1}^n \f$, the optimisation problem is maximising the function 
    * \f$ l(\vec{w}) = - \sum_{i=1}^n \ln(1 + \exp(- y_i \vec{w} \cdot \vec{x}_i)) - \frac{\lambda}{2} \vec{w} \cdot \vec{w} \f$.
    *
    */
    class LogisticRegression
    {
    public:        
        /**
         * @brief Result of binomial logistic regression.
        */
        struct Result
        {            
            Eigen::VectorXd w; /**< Fitted coefficients of the LR model. */
            unsigned int steps_taken; /**< Number of steps taken to converge. */
            bool converged; /**< Did it converge? */

            /**
             * @brief Predicts labels for features X given w.
             * @param X D x N matrix of X values, with data points in columns.
             * @param[out] y Y vector with length N.
             * @return Fills `y` with -1 or 1 values.
             * @throw std::invalid_argument If matrix or vector dimensions do not match.
            */
            DLL_DECLSPEC void predict(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXd> y) const;

            /**
             * @brief Predicts labels for features X given w. Version which returns a new vector.
             * @param X D x N matrix of X values, with data points in columns.
             * @return Vector with -1 or 1 values.
             * @throw std::invalid_argument If matrix or vector dimensions do not match.
            */
            Eigen::VectorXd predict(Eigen::Ref<const Eigen::MatrixXd> X) const
            {
                Eigen::VectorXd y(X.cols());
                predict(X, y);
                return y;
            }

            /**
             * @brief Predicts label for feature x given w.
             * @param x D-dimensional vector.
             * @return -1 or 1.
             * @throw std::invalid_argument If vector dimensions do not match.
            */
            DLL_DECLSPEC double predict_single(Eigen::Ref<const Eigen::VectorXd> x) const;

            /** @brief Formats the result as string. */
            DLL_DECLSPEC std::string to_string() const;
        };

        /**
         * @brief Virtual destructor.
        */
        DLL_DECLSPEC virtual ~LogisticRegression();

        /**
         * @brief Fits the model and returns the result.
         * 
         * If fitting with intercept is desired, include a row of 1's in the X values.
         * 
         * @param X D x N matrix of X values, with data points in columns.
         * @param y Y vector with length N. Values should be -1 or 1.
         * @throw std::invalid_argument if N or D are zero, or if dimensions of `X` and `y` do not match.
         * @return Result object.
        */
        DLL_DECLSPEC virtual Result fit(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y) const = 0;

        /**
         * @brief Calculates the probability of label given model weights and feature vector.
         * @param x Feature vector.
         * @param y Label (-1 or 1).
         * @param w Model weight vector, equal length to `x`.
         * @return P(y|x,w).
        */
        DLL_DECLSPEC static double probability(Eigen::Ref<const Eigen::VectorXd> x, double y, Eigen::Ref<const Eigen::VectorXd> w);

        /**
         * @brief Calculates the posterior log-likelihood of data given model weights.
         * @param X D x N matrix of X values, with data points in columns.
         * @param y Y vector with length N. Values should be -1 or 1.
         * @param w Model weight vector with length D.
         * @param lam Inverse variance of the Gaussian prior for `w`. Cannot be negative. Set it to 0 if you want to perform maximum likelihood estimation of `w`.
         * @return Log-likelihood.
         * @throw std::domain_error If `lam` is negative.
         * @throw std::invalid_argument If matrix or vector dimensions do not match.
        */
        DLL_DECLSPEC static double log_likelihood(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<const Eigen::VectorXd> w, double lam);

        /**
         * @brief Calculates the gradient of the posterior log-likelihood of data given model weights, over those weights.
         * @param X D x N matrix of X values, with data points in columns.
         * @param y Y vector with length N. Values should be -1 or 1.
         * @param w Model weight vector with length D.
         * @param lam Inverse variance of the Gaussian prior for `w`. Cannot be negative. Set it to 0 if you want to perform maximum likelihood estimation of `w`.
         * @param[out] g Vector with length D for the computed gradient of log-likelihood over weights `w`.
         * @throw std::domain_error If `lam` is negative.
         * @throw std::invalid_argument If matrix or vector dimensions do not match.
        */
        DLL_DECLSPEC static void grad_log_likelihood(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<const Eigen::VectorXd> w, double lam, Eigen::Ref<Eigen::VectorXd> g);

        /**
         * @brief Calculates the Hessian of the posterior log-likelihood of data given model weights, over those weights.
         * @param X D x N matrix of X values, with data points in columns.
         * @param y Y vector with length N. Values should be -1 or 1.
         * @param w Model weight vector with length D.
         * @param lam Inverse variance of the Gaussian prior for `w`. Cannot be negative. Set it to 0 if you want to perform maximum likelihood estimation of `w`.
         * @param[out] H D x D matrix for the computed Hessian of log-likelihood over weights `w`.
         * @throw std::domain_error If `lam` is negative.
         * @throw std::invalid_argument If matrix or vector dimensions do not match.
        */
        DLL_DECLSPEC static void hessian_log_likelihood(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<const Eigen::VectorXd> w, double lam, Eigen::Ref<Eigen::MatrixXd> H);        
    };

    /**
     * @brief Abstract implementation, sharing the common parameters and stopping criterion.
     * 
     * The stopping criterion is ||old_weights - new_weights||_2 <= absolute_tolerance + relative_tolerance * max(||old_weights||_2, ||new_weights||_2).
    */
    class AbstractLogisticRegression : public LogisticRegression
    {
    public:
        /**
         * @brief Constructor setting default parameter values.
        */
        DLL_DECLSPEC AbstractLogisticRegression();

        /**
         * @brief Returns the regularisation parameter.
         * @return Inverse variance of the Gaussian prior for `w`. Cannot be negative. Set it to 0 if you want to perform maximum likelihood estimation of `w`. 
        */
        double lam() const
        {
            return lam_;
        }

        /**
         * @brief Returns absolute tolerance for fitted weights.
        */
        double absolute_tolerance() const
        {
            return absolute_tolerance_;
        }

        /**
         * @brief Returns relative tolerance for fitted weights.
        */
        double relative_tolerance() const
        {
            return relative_tolerance_;
        }

        /**
         * @brief Returns maximum number of steps allowed.
        */
        unsigned int maximum_steps() const
        {
            return maximum_steps_;
        }

        /**
         * @brief Sets the regularisation parameter.
         * @param lam Inverse variance of the Gaussian prior for `w`. Cannot be negative. Set it to 0 if you want to perform maximum likelihood estimation of `w`.
         * @throw std::domain_error If `lam` is negative.
        */
        DLL_DECLSPEC void set_lam(double lam);

        /**
         * @brief Sets absolute tolerance for weight convergence.
         * @param absolute_tolerance Cannot be negative.
         * @throw std::domain_error If negative.
        */
        DLL_DECLSPEC void set_absolute_tolerance(double absolute_tolerance);

        /**
         * @brief Sets relative tolerance for weight convergence.
         * @param relative_tolerance Cannot be negative.
         * @throw std::domain_error If negative.
        */
        DLL_DECLSPEC void set_relative_tolerance(double relative_tolerance);

        /**
         * @brief Sets maximum number of steps.
        */
        DLL_DECLSPEC void set_maximum_steps(unsigned int maximum_steps);
    protected:
        /**
         * @brief Check if weight fitting converged.
         * @param old_weights Previous weight vector.
         * @param new_weights New weight vector.
         * @return True if converged, false otherwise.
        */
        bool weights_converged(Eigen::Ref<const Eigen::VectorXd> old_weights, Eigen::Ref<const Eigen::VectorXd> new_weights) const;
    private:        
        double lam_;
        double relative_tolerance_;
        double absolute_tolerance_;
        unsigned int maximum_steps_;
    };

    /**
     * @brief Conjugate gradient logistic regression, as described in Sec. 4 of Thomas P. Minka, "A comparison of numerical optimizers for logistic regression".
    */
    class ConjugateGradientLogisticRegression : public AbstractLogisticRegression
    {
    public:        
        DLL_DECLSPEC Result fit(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y) const override;
    };
}
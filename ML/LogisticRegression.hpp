#pragma once
/* (C) 2021 Roman Werpachowski */
#include <Eigen/Core>

namespace ml
{
    /** @brief Binomial logistic regression algorithm.
    *
    * Based on Thomas P. Minka, "A comparison of numerical optimizers for logistic regression", but modifying the labels to be 0 or 1, not +/-1.
    * 
    * The model is:
    * 
    * \f$ P(y = 0 or 1 | \vec{x}, \vec{w}) = \frac{1}{1 + \exp(-y \vec{w} \cdot \vec{x} )} \f$
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
            unsigned int n; /**< Number of data points. */
            Eigen::VectorXd w; /**< Fitted coefficients of the LR model. */
        };

        /**
         * @brief Virtual destructor.
        */
        virtual ~LogisticRegression();

        /**
         * @brief Fit the model and return the result.
         * 
         * If fitting with intercept is desired, include a row of 1's in the X values.
         * 
         * @param X D x N matrix of X values, with data points in columns.
         * @param y Y vector with length N. Values should be 0 or 1.
         * @return Result object.
        */
        virtual Result fit(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y) = 0;

    protected:
        /**
         * @brief Calculates the probability of label given model weights and feature vector.
         * @param x Feature vector.
         * @param y Label (0 or 1).
         * @param w Model weight vector, equal length to `x`.
         * @return P(y|x,w).
        */
        static double probability(Eigen::Ref<const Eigen::VectorXd> x, double y, Eigen::Ref<const Eigen::VectorXd> w);

        /**
         * @brief Calculates the data likelihood given model weights.
         * @param X D x N matrix of X values, with data points in columns.
         * @param y Y vector with length N. Values should be 0 or 1.
         * @param w Model weight vector with length D.
         * @param lam Regularisation parameter (inverse variance of the prior for `w`). Cannot be negative.
         * @return Likelihood.
        */
        static double likelihood(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y, Eigen::Ref<const Eigen::VectorXd> w, double lam);
    };
}
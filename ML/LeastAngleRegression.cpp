/* (C) 2020 Roman Werpachowski. */
#include <boost/format.hpp>
#include "LeastAngleRegression.hpp"
#include "RootFinding.hpp"

namespace ml
{
    namespace LinearRegression
    {
		template <> LeastAngleRegressionResult least_angle_regression<false>(Eigen::Ref<const Eigen::MatrixXd> X, Eigen::Ref<const Eigen::VectorXd> y)
		{
			// X is an q x N matrix and y is a N-size vector.
			// Rows of X are assumed to have mean 0 and population variance 1.
			const auto q = X.rows();
			const auto n = X.cols();
			if (n != static_cast<unsigned int>(y.size())) {
				throw std::invalid_argument("X and Y vectors have different sizes");
			}
			std::vector<Eigen::Index> selected_features;
			selected_features.reserve(q);
			const double intercept = y.mean();
			LeastAngleRegressionResult lar_result;
			lar_result.n = static_cast<unsigned int>(n);
			lar_result.dof = static_cast<unsigned int>(n - q - 1); // -1 for the intercept.
			lar_result.lasso_path.reserve(q);
			Eigen::VectorXd beta(Eigen::VectorXd::Zero(q));
			Eigen::VectorXd residuals(y.array() - y.mean());
			Eigen::VectorXd covariances_with_residuals(q);
			Eigen::VectorXd solution(Eigen::VectorXd::Zero(n));
			Eigen::MatrixXd X_sel(Eigen::MatrixXd::Zero(q, n));
			Eigen::VectorXd ls_solution(n);
			const Eigen::MatrixXd XXt = X * X.transpose();
			const Eigen::MatrixXd X_cov = XXt / static_cast<double>(n);
			for (Eigen::Index i = 0; i < q; ++i) {
				covariances_with_residuals[i] = residuals.dot(X.row(i)) / static_cast<double>(n);
			}
			const Eigen::VectorXd covariances_with_initial_residuals(covariances_with_residuals);
			Eigen::VectorXd covariances_with_ls_solution(q);
			std::vector<double> gammas(q, 0);
			// Selection of the first feature is a special case.
			const auto most_correlated_feature_it = std::max_element(covariances_with_residuals.data(), covariances_with_residuals.data() + q, [](const double& l, const double& r) { return std::abs(l) < std::abs(r); });
			selected_features.push_back(std::distance(covariances_with_residuals.data(), most_correlated_feature_it));
			X_sel.row(0) = X.row(selected_features.front());
			// +Inf means this feature was already selected.
			gammas[selected_features.front()] = std::numeric_limits<double>::infinity();
			// Select rest of the features.
			while (selected_features.size() < static_cast<size_t>(q)) {
				residuals = y.array() - y.mean();
				solution.setZero();
				const auto nbr_selected_features = selected_features.size();
				covariances_with_residuals = covariances_with_initial_residuals;
				for (size_t i = 0; i < nbr_selected_features; ++i) {
					const auto k = selected_features[i];
					residuals -= beta[i] * X.row(k);
					solution += beta[i] * X.row(k);
					covariances_with_residuals -= beta[i] * X_cov.col(k);
				}
				// TODO: optimize.
				const auto ls_result = multivariate(X_sel.topRows(nbr_selected_features), residuals);
				ls_solution = ls_result.predict(X_sel.topRows(nbr_selected_features));
				covariances_with_ls_solution.setZero();
				double cov_r_ls_solution = 0;
				double cov_r_solution = 0;
				for (size_t i = 0; i < nbr_selected_features; ++i) {
					const auto k = selected_features[i];
					covariances_with_ls_solution += ls_result.beta[i] * X_cov.col(k);
					cov_r_ls_solution += ls_result.beta[i] * covariances_with_residuals[k];
					cov_r_solution += beta[i] * covariances_with_residuals[k];
				}
				// Because residuals.mean() and solution.mean() are zero, we can use squared norm to compute
				// the variance.
				const double var_r = residuals.squaredNorm() / static_cast<double>(n);
				const double var_solution = solution.squaredNorm() / static_cast<double>(n);
				const double var_ls_solution = ls_solution.squaredNorm() / static_cast<double>(n);
				for (Eigen::Index k = 0; k < q; ++k) {
					if (std::isinf(gammas[k])) {
						// Skip already selected gammas.
						continue;
					}
					// Set up a quadratic equation for gamma, a * gamma^2 + b * gamma + c = 0.
					const double a = var_r * var_solution * std::pow(covariances_with_ls_solution[k], 2) - var_ls_solution * std::pow(cov_r_solution, 2);
					const double b = 2 * (
						std::pow(cov_r_solution, 2) * cov_r_ls_solution
						- covariances_with_residuals[k] * var_r * var_ls_solution * covariances_with_ls_solution[k]);
					const double c = var_r * (
						std::pow(covariances_with_residuals[k], 2) * var_solution
						- std::pow(cov_r_solution, 2));
					if (a != 0) {
						double gamma1, gamma2;
						const auto nbr_roots = ml::RootFinding::solve_quadratic(a, b, c, gamma1, gamma2);
						if (!nbr_roots) {
							throw std::runtime_error(boost::str(boost::format("No solutions found for gamma: A=%g, B=%g, C=%g") % a % b % c));
						} else if (nbr_roots == 1) {
							if (gamma1 < 0) {
								throw std::runtime_error(boost::str(boost::format("No non-negative solutions found for gamma: A=%g, B=%g, C=%g") % a % b % c));
							}
							gammas[k] = gamma1;
						} else {
							assert(nbr_roots == 2);
							if (gamma2 < gamma1) {
								std::swap(gamma1, gamma2);
							}
							if (gamma2 < 0) {
								throw std::runtime_error(boost::str(boost::format("No non-negative solutions found for gamma: A=%g, B=%g, C=%g") % a % b % c));
							} else {
								if (gamma1 < 0) {
									gammas[k] = gamma2;
								} else {
									gammas[k] = gamma1;
								}
							}
						}
					} else {
						// Solve linear equation b * gamma + c == 0.
						if (b != 0 || c != 0) {
							const double gamma = -c / b;
							if (gamma >= 0) {
								gammas[k] = 0;
							} else {
								throw std::runtime_error(boost::str(boost::format("No non-negative solutions found for gamma: A=%g, B=%g, C=%g") % a % b % c));
							}
						} else {
							gammas[k] = 0;
						}
					} // find gammas[k]
				} // loop over 0 <= k < q
				const auto selected_feature_idx = std::distance(gammas.begin(), std::min_element(gammas.begin(), gammas.end()));
				const auto gamma = gammas[selected_feature_idx];
				if (std::isinf(gamma)) {
					throw std::runtime_error("Cannot update the solution");
				}
				X_sel.row(nbr_selected_features) = X.row(selected_feature_idx);
				unsigned int nbr_nonzero_betas = 0;
				for (size_t i = 0; i < nbr_selected_features; ++i) {
					const auto k = selected_features[i];
					beta[k] += gamma * ls_result.beta[i];
					if (beta[k] != 0) {
						++nbr_nonzero_betas;
					}
				}
				selected_features.push_back(selected_feature_idx);
				LassoRegressionResult lasso_result;
				lasso_result.n = static_cast<unsigned int>(n);
				lasso_result.dof = lar_result.dof;
				lasso_result.beta.resize(q + 1);
				lasso_result.beta.head(q) = beta;
				lasso_result.beta[q] = intercept;
				lasso_result.effective_dof = static_cast<double>(n - nbr_nonzero_betas - 1); // -1 to account for the intercept.
			}
			return lar_result;
		}
    }
}
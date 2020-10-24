/* (C) 2020 Roman Werpachowski. */
#include <cassert>
#include <cmath>
#include <stdexcept>
#include "RootFinding.hpp"

namespace ml
{
    namespace RootFinding
    {
        unsigned int solve_quadratic(const double a, const double b, const double c, double& x1, double& x2)
        {
            if (!a) {
                throw std::domain_error("A must be nonzero");
            }
            const double delta = b * b - 4 * a * c;
            if (delta < 0) {
                return 0;
            } else if (delta == 0) {
                x1 = -b / (2 * a);
                return 1;
            } else {
                const double sqrt_delta = std::sqrt(delta);
                const double sign_b = b >= 0 ? 1 : -1;
                const double t = -0.5 * (b + sign_b * sqrt_delta);
                x1 = t / a;
                x2 = c / t;
                assert(x1 != x2);
                return 2;
            }
        }
    }
}
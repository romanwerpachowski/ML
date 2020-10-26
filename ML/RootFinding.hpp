/* (C) 2020 Roman Werpachowski. */
#pragma once
#include "dll.hpp"

namespace ml
{
    /** @brief Root finding methods. */
    namespace RootFinding
    {
        /** @brief Solves a quadratic equation a * x^2 + b * x + c == 0.
        * 
        * Algorithm from https://stackoverflow.com/a/900119/59557
        * 
        * @param[in] a Second order coefficient, nonzero.
        * @param[in] b First order coefficient.
        * @param[in] c Zeroth order coefficient.
        * @param[out] x1 First root. Set only if the equation has any real roots.
        * @param[out] x2 Second root. Set only if the equation has two real roots.
        * @return Number of real roots.
        * @throw std::domain_error If `a == 0`.
        */
        DLL_DECLSPEC unsigned int solve_quadratic(double a, double b, double c, double& x1, double& x2);
    }
}
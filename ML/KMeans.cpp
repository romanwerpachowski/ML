/* (C) 2021 Roman Werpachowski. */
#include "KMeans.hpp"


namespace ml
{
    namespace Clustering
    {
        void KMeans::set_seed(unsigned int seed)
        {
            prng_.seed(seed);
        }

		void KMeans::set_absolute_tolerance(double absolute_tolerance)
		{
			if (absolute_tolerance < 0) {
				throw std::domain_error("KMeans: Negative absolute tolerance");
			}
			absolute_tolerance_ = absolute_tolerance;
		}

		void KMeans::set_relative_tolerance(double relative_tolerance)
		{
			if (relative_tolerance < 0) {
				throw std::domain_error("KMeans: Negative relative tolerance");
			}
			relative_tolerance_ = relative_tolerance;
		}

		void KMeans::set_maximum_steps(unsigned int maximum_steps)
		{
			if (maximum_steps < 2) {
				throw std::invalid_argument("KMeans: At least two steps required for convergence test");
			}
			maximum_steps_ = maximum_steps;
		}

		void KMeans::set_centroids_initialiser(std::shared_ptr<const Clustering::CentroidsInitialiser> centroids_initialiser)
		{
			if (!centroids_initialiser) {
				throw std::invalid_argument("KMeans: Null centroids initialiser");
			}
			centroids_initialiser_ = centroids_initialiser;
		}
    }
}
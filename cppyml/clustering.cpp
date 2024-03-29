/* (C) 2020 Roman Werpachowski. */
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "ML/Clustering.hpp"
#include "ML/EM.hpp"
#include "ML/KMeans.hpp"
#include "types.hpp"


namespace py = pybind11;


namespace ml
{
    /** Version of ml::EM adapted for Python bindings. */
    class EMPy : public EM
    {
    public:
        EMPy(unsigned int number_components)
            : EM(number_components)
        {}

        /** Fits the model to data in row-major order.
        @param data Matrix (row-major order) with a data point in every row.
        */
        bool fit_row_major(Eigen::Ref<const MatrixXdR> data)
        {
            return fit(data.transpose());
        }

        /** @brief Given a data point x, calculate each component's responsibilities for x and return them.
        @see assign_responsibilities().
        */
        Eigen::VectorXd calculate_responsibilities(Eigen::Ref<const Eigen::VectorXd> x) const
        {
            Eigen::VectorXd u(number_components());
            EM::assign_responsibilities(x, u);
            return u;
        }        
    };

    namespace Clustering
    {
        /**
         * @brief Version of ml::Clustering::KMeans adapted for Python bindings.
        */
        class KMeansPy : public KMeans
        {
        public:
            KMeansPy(unsigned int number_clusters)
                : KMeans(number_clusters)
            {}

            /** Fits the model to data in row-major order.
            @param data Matrix (row-major order) with a data point in every row.
            */
            bool fit_row_major(Eigen::Ref<const MatrixXdR> data)
            {
                return fit(data.transpose());
            }

            /**
             * @brief Returns centroids in row-major order.
            */
            Eigen::Ref<const MatrixXdR> centroids_row_major() const
            {
                return this->centroids().transpose();
            }
        };
    }
}


void init_clustering(py::module& m) 
{
    auto m_clustering = m.def_submodule("clustering", "Clustering algorithms.");

    py::class_<ml::Clustering::CentroidsInitialiser, std::shared_ptr<ml::Clustering::CentroidsInitialiser>>(m_clustering, "CentroidsInitialiser")
        .doc() = "Abstract centroids initialiser.";

    py::class_<ml::Clustering::ResponsibilitiesInitialiser, std::shared_ptr<ml::Clustering::ResponsibilitiesInitialiser>>(m_clustering, "ResponsibilitiesInitialiser")
        .doc() = "Abstract responsibilities initialiser.";

    py::class_<ml::Clustering::Forgy, std::shared_ptr<ml::Clustering::Forgy>, ml::Clustering::CentroidsInitialiser>(m_clustering, "Forgy")
        .def(py::init<>())
        .doc() = "Forgy initialisation algorithm.";

    py::class_<ml::Clustering::RandomPartition, std::shared_ptr<ml::Clustering::RandomPartition>, ml::Clustering::CentroidsInitialiser>(m_clustering, "RandomPartition")
        .def(py::init<>())
        .doc() = "Random Partition initialisation algorithm.";

    py::class_<ml::Clustering::KPP, std::shared_ptr<ml::Clustering::KPP>, ml::Clustering::CentroidsInitialiser>(m_clustering, "KPP")
        .def(py::init<>())
        .doc() = "KMeans++ initialisation algorithm.";

    py::class_<ml::Clustering::ClosestCentroid, std::shared_ptr<ml::Clustering::ClosestCentroid>, ml::Clustering::ResponsibilitiesInitialiser>(m_clustering, "ClosestCentroid")
        .def(py::init<std::shared_ptr<ml::Clustering::CentroidsInitialiser>>(), py::arg("centroids_initialiser"))
        .doc() = "Assigns points to closest centroid.";

    py::class_<ml::EMPy, std::shared_ptr<ml::EMPy>>(m_clustering, "EM")
        .def(py::init<unsigned int>(), py::arg("number_components"), R"(Constructor.

Args:
    number_components: Number of Gaussian components to fit.
)")
        .def("set_seed", &ml::EMPy::set_seed, py::arg("seed"), "Sets PRNG seed.")
        .def("set_absolute_tolerance", &ml::EMPy::set_absolute_tolerance, py::arg("absolute_tolerance"), "Sets absolute tolerance.")
        .def("set_relative_tolerance", &ml::EMPy::set_relative_tolerance, py::arg("relative_tolerance"), "Sets relative tolerance.")
        .def("set_maximum_steps", &ml::EMPy::set_maximum_steps, py::arg("maximum_steps"), "Sets maximum number of iterations.")
        .def("set_means_initialiser", &ml::EMPy::set_means_initialiser, py::arg("means_initialiser"), "Sets the algorithm to initialise component means.")
        .def("set_responsibilities_initialiser", &ml::EMPy::set_responsibilities_initialiser, py::arg("responsibilities_initialiser"), "Sets the algorithm to initialise responsibilities for data points.")
        .def("set_verbose", &ml::EMPy::set_verbose, py::arg("verbose"), "Turns on/off the verbose mode.")
        .def("set_maximise_first", &ml::EMPy::set_maximise_first, py::arg("maximise_first"), "Turns on/off doing an initial maximisation step before the E-M iterations.")
        .def("fit", &ml::EMPy::fit_row_major, py::arg("data").noconvert(), 
            R"(Fits the components to the data.

Args:
    data: A 2D array with data points in rows.

Returns:
    True if EM algorithm converged.
)"
        )
        .def_property_readonly("number_components", &ml::EMPy::number_components, "Number of Gaussian components.")
        .def_property_readonly("means", &ml::EMPy::means, "Fitted means.")
        .def_property_readonly("responsibilities", &ml::EMPy::responsibilities, "Fitted responsibilities.")
        .def_property_readonly("log_likelihood", &ml::EMPy::log_likelihood, "Maximised log-likelihood.")
        .def_property_readonly("mixing_probabilities", &ml::EMPy::mixing_probabilities, "Mixing probabilities of components.")
        .def("covariance", &ml::EMPy::covariance, py::arg("k"), R"(Returns k-th covariance matrix.

Args:
    k: Gaussian component index.

Returns:
    2D square matrix with covariance coefficients.
)")
        .def("assign_responsibilities", &ml::EMPy::calculate_responsibilities, py::arg("x"),
            R"(Given a data point x, calculate each component's responsibilities for x and return them.

Args:
    x: Data point with correct dimension.

Returns:
    Array of components' responsibilities.
)")
        .doc() = "Gaussian Expectation-Maximisation algorithm.";

    py::class_<ml::Clustering::KMeansPy, std::shared_ptr<ml::Clustering::KMeansPy>>(m_clustering, "KMeans")
        .def(py::init<unsigned int>(), py::arg("number_clusters"), R"(Constructor.

Args:
    number_clusters: Number of clusters to fit.
)")
    .def("set_seed", &ml::Clustering::KMeansPy::set_seed, py::arg("seed"), "Sets the PRNG seed.")
    .def("set_absolute_tolerance", &ml::Clustering::KMeansPy::set_absolute_tolerance, py::arg("absolute_tolerance"), "Sets absolute tolerance.")
    .def("set_maximum_steps", &ml::Clustering::KMeansPy::set_maximum_steps, py::arg("maximum_steps"), "Sets maximum number of iterations.")
    .def("set_centroids_initialiser", &ml::Clustering::KMeansPy::set_centroids_initialiser, py::arg("centroids_initialiser"), "Sets the algorithm to initialise cluster centroids.")
    .def("set_number_initialisations", &ml::Clustering::KMeansPy::set_number_initialisations, py::arg("centroids_initialiser"), "Sets number of initialisations to try, to find the clusters with lowest inertia.")
    .def("set_verbose", &ml::Clustering::KMeansPy::set_verbose, py::arg("verbose"), "Turns on/off the verbose mode.")
    .def("fit", &ml::Clustering::KMeansPy::fit_row_major, py::arg("data").noconvert(),
    R"(Fits the components to the data.

Args:
    data: A 2D array with data points in rows.

Returns:
    True if the algorithm converged.
)"
)
    .def_property_readonly("number_clusters", &ml::Clustering::KMeansPy::number_clusters, "Number of clusters.")
    .def_property_readonly("centroids", &ml::Clustering::KMeansPy::centroids_row_major, "Fitted centroids.")
    .def_property_readonly("labels", &ml::Clustering::KMeansPy::labels, "Fitted labels.")
    .def_property_readonly("inertia", &ml::Clustering::KMeansPy::inertia, "Minimised inertia.")    
    .def("assign_label", &ml::Clustering::KMeansPy::assign_label, py::arg("x"),
    R"(Given a data point x, assigns it to the closest cluster.

Args:
    x: Data point with correct dimension.

Returns:
    Cluster label for point x.
)")
    .doc() = "Naive K-Means algorithm.";
}
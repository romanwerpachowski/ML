#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_clustering(py::module& m);
void init_decision_trees(py::module& m);

PYBIND11_MODULE(PyML, m)
{
	m.doc() = "PyML: a set of lovingly curated machine learning algorithms for Python.";
	init_clustering(m);
	init_decision_trees(m);
}

/* Main source for the cppyml extension.

IMPORTANT: compile it with the same optimisation options as ML.dll!
*/

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_clustering(py::module& m);
void init_decision_trees(py::module& m);
void init_linear_regression(py::module& m);

PYBIND11_MODULE(cppyml, m)
{
	m.doc() = R"(cppyml: Python bindings for efficient C++ implementations of selected ML algorithms.

cppyml provides Python programmers with a curated selection of popular ML algorithms implemented in C++.
The goal is to provide well-tested, highly optimised implementations.

(C) 2020 Roman Werpachowski. Available under GPG v3 license.
)";
	init_clustering(m);
	init_decision_trees(m);
	init_linear_regression(m);
}
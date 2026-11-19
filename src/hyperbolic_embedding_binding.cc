#define BUILD_PYTHON_MODULE

// Include Python headers before standard headers to set feature-test macros.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hyperbolic_embedding.hh"
#include "hyperbolic_embedding.cc"

#include <vector>
#include <array>

namespace py = pybind11;
using namespace std;

template <typename T>
py::array_t<T> poss_to_numpy(const vector<array<T, 3>>& poss) {
    size_t n = poss.size();
    py::array_t<T> result({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});
    auto buf = result.template mutable_unchecked<2>();
    for (size_t i = 0; i < n; ++i) {
        buf(i, 0) = poss[i][0];
        buf(i, 1) = poss[i][1];
        buf(i, 2) = poss[i][2];
    }
    return result;
}

PYBIND11_MODULE(_hyperbolic_embedding, m) {
    m.doc() = "Hyperbolic embedding / force-directed graph layout optimization module";

    m.def("set_seed", &set_seed, "Fix random seed for reproducibility (affects step_adam)", py::arg("seed"));

    py::class_<GradState>(m, "GradState")
        .def(py::init<int>(), "Create GradState with N nodes", py::arg("N") = 1)
        .def_readwrite("N", &GradState::N, "Number of nodes")
        .def_readwrite("max_level", &GradState::max_level, "Maximum Quadtree level")
        .def_readwrite("theta", &GradState::theta, "Barnes-Hut opening criterion")
        .def_readwrite("mu", &GradState::mu, "Distance threshold (mu) for hyperbolic embedding")
        .def_readwrite("beta", &GradState::beta, "Temperature parameter for hyperbolic embedding")
        .def_readwrite("beta1", &GradState::beta1, "Adam beta1 parameter")
        .def_readwrite("beta2", &GradState::beta2, "Adam beta2 parameter")
        .def_readwrite("epsilon", &GradState::epsilon, "Adam epsilon parameter")
        .def_readwrite("grad_clip_value", &GradState::grad_clip_value, "Gradient clipping value")
        .def("set_edgelist", [](GradState& self, py::object edgelist) {
            vector<pair<int, int>> edges;
            if (py::isinstance<py::list>(edgelist) || py::isinstance<py::tuple>(edgelist)) {
                py::list edge_list = py::cast<py::list>(edgelist);
                for (auto item : edge_list) {
                    if (py::isinstance<py::tuple>(item) || py::isinstance<py::list>(item)) {
                        py::tuple edge = py::cast<py::tuple>(item);
                        if (py::len(edge) == 2) {
                            int u = py::cast<int>(edge[0]);
                            int v = py::cast<int>(edge[1]);
                            edges.emplace_back(u, v);
                        }
                    }
                }
            }
            else if (py::isinstance<py::array>(edgelist)) {
                py::array_t<int> arr = py::cast<py::array_t<int>>(edgelist);
                auto buf = arr.request();
                if (buf.ndim == 2 && buf.shape[1] == 2) {
                    int* ptr = static_cast<int*>(buf.ptr);
                    for (py::ssize_t i = 0; i < buf.shape[0]; ++i) {
                        int u = ptr[i * 2 + 0];
                        int v = ptr[i * 2 + 1];
                        edges.emplace_back(u, v);
                    }
                } else {
                    throw runtime_error("Numpy array must be 2D with shape (E, 2)");
                }
            } else {
                throw runtime_error("edgelist must be a list of tuples or numpy array with shape (E, 2)");
            }
            self.set_edgelist(edges);
        }, "Set edges from Python list of tuples or numpy array (E x 2)", py::arg("edgelist"))
        .def("get_positions", [](GradState& self) {
            return poss_to_numpy(self.get_pos());
        }, "Get positions as numpy array (N x 3)")
        .def("set_positions", [](GradState& self, py::array_t<double> arr) {
            auto buf = arr.request();
            if (buf.ndim != 2 || buf.shape[1] != 3) {
                throw runtime_error("Array must be 2D with shape (N, 3)");
            }
            size_t n = buf.shape[0];
            if (n != static_cast<size_t>(self.N)) {
                throw runtime_error("Array size must match N");
            }
            vector<array<double, 3>> positions(n);
            double* ptr = static_cast<double*>(buf.ptr);
            for (size_t i = 0; i < n; ++i) {
                positions[i] = {ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]};
            }
            self.set_positions(positions);
        }, "Set positions from numpy array (N x 3)", py::arg("positions"))
        .def("set_pos", [](GradState& self, int node_idx, py::object r_or_pos, py::object theta_or_none = py::none(), py::object z_or_none = py::none()) {
            if (node_idx < 0 || node_idx >= self.N) {
                throw runtime_error("node_idx out of range");
            }
            double r, theta, z;
            if (!py::isinstance<py::none>(theta_or_none) && !py::isinstance<py::none>(z_or_none)) {
                r = py::cast<double>(r_or_pos);
                theta = py::cast<double>(theta_or_none);
                z = py::cast<double>(z_or_none);
            }
            else if (py::isinstance<py::tuple>(r_or_pos) || py::isinstance<py::list>(r_or_pos)) {
                py::tuple pos_tuple = py::cast<py::tuple>(r_or_pos);
                if (py::len(pos_tuple) != 3) {
                    throw runtime_error("Position must have 3 values (r, theta, z)");
                }
                r = py::cast<double>(pos_tuple[0]);
                theta = py::cast<double>(pos_tuple[1]);
                z = py::cast<double>(pos_tuple[2]);
            }
            else if (py::isinstance<py::array>(r_or_pos)) {
                py::array_t<double> arr = py::cast<py::array_t<double>>(r_or_pos);
                auto buf = arr.request();
                if (buf.ndim != 1 || buf.size != 3) {
                    throw runtime_error("Position array must be 1D with 3 elements");
                }
                double* ptr = static_cast<double*>(buf.ptr);
                r = ptr[0];
                theta = ptr[1];
                z = ptr[2];
            }
            else {
                throw runtime_error("Position must be (r, theta, z) as three arguments, tuple/list, or numpy array");
            }
            self.set_pos(node_idx, r, theta, z);
        }, "Set position of a single node. Usage: set_pos(node_idx, r, theta, z) or set_pos(node_idx, [r, theta, z])", 
           py::arg("node_idx"), py::arg("r_or_pos"), py::arg("theta") = py::none(), py::arg("z") = py::none())
        
        .def("get_forces", [](GradState& self) {
            return poss_to_numpy(self.get_forces());
        }, "Get forces as numpy array (N x 3)")
        .def("get_edges", [](GradState& self) {
            return self.get_edges();
        }, "Get list of edges as list of (u, v) tuples")
        .def("get_num_edges", [](GradState& self) {
            return self.get_edges().size();
        }, "Get number of edges")
        .def("set_theta", &GradState::set_theta, "Set Barnes-Hut opening criterion", py::arg("new_theta"))
        .def("set_mu", &GradState::set_mu, "Set distance threshold mu", py::arg("new_mu"))
        .def("set_beta", &GradState::set_beta, "Set temperature parameter beta", py::arg("new_beta"))
        .def("compute_repulsive_force_exact", &GradState::compute_repulsive_force_exact, "Compute repulsive forces exactly (all pairs)")
        .def("compute_repulsive_force_bh", &GradState::compute_repulsive_force_bh, "Compute repulsive forces using Barnes-Hut")
        .def("add_attractive_force", &GradState::add_attractive_force, "Add attractive (spring) forces")
        .def("add_l1_penalty_z", &GradState::add_l1_penalty_z, "Add L1 penalty to z coordinate", py::arg("penalty_z"))
        .def("add_l2_penalty_z", &GradState::add_l2_penalty_z, "Add L2 penalty to z coordinate", py::arg("penalty_z"))
        .def("project_equator", &GradState::project_equator, "Project all nodes to equator (z=0)")
        .def("step_adam", &GradState::step_adam, "Perform one Adam optimization step", py::arg("lr"))
        .def("step_adam_decay_z", &GradState::step_adam_decay_z, "Adam step with z-decay", py::arg("lr"), py::arg("lambda"))
        .def("get_lnp_exact", &GradState::get_lnp_exact, "Get log-probability (exact, all pairs)")
        .def("get_lnp_bh", &GradState::get_lnp_bh, "Get log-probability (Barnes-Hut approximation)");
}

#ifndef HYPERBOLIC_EMBEDDING_HH
#define HYPERBOLIC_EMBEDDING_HH

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <iomanip>
#include <array>

#include "quad_tree_hyperbolic.hh"

using namespace std;

using pos_t = array<double, 3>;  // (r, θ, z) in cylindrical coordinates

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr double EPSILON = 1e-10;
constexpr double MAX_FORCE = 100.0;

/** Return true if each coordinate difference |a[i] - b[i]| is <= tol (default 1e-10). */
template <typename Pos, size_t dim = 3>
inline bool is_same_point(const Pos& a, const Pos& b, double tol = 1e-10) {
    for (size_t i = 0; i < dim; ++i)
        if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

inline void zero_force(double& Fx, double& Fy, double& Fz) {
    Fx = 0;
    Fy = 0;
    Fz = 0;
}

double rnd();
void set_seed(unsigned int seed);
double distance_hyperbolic_2d(const pos_t& p1, const pos_t& p2);
template <typename Pos2D>
bool is_inside_2d(const Pos2D& p, const Pos2D& ll, const Pos2D& ur);

class GradState {
public:
    int N;
    int max_level = 15;
    double theta = 0.6;
    double mu = 0.1;
    double beta = 3.0;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    double grad_clip_value = 1.0;

    explicit GradState(int n = 1);
    void set_edgelist(const vector<pair<int, int>>& edgelist);
    void set_pos(int node_idx, double r, double theta, double z);
    void set_positions(const vector<pos_t>& positions);
    void set_theta(double new_theta);
    void set_mu(double new_mu);
    void set_beta(double new_beta);
    void compute_repulsive_force_exact();
    int compute_repulsive_force_bh();
    void add_attractive_force();
    void add_l1_penalty_z(double penalty_z);
    void add_l2_penalty_z(double penalty_z);
    void project_equator();
    void step_adam(double lr);
    void step_adam_decay_z(double lr, double lambda);
    double get_lnp_exact();
    double get_lnp_bh();

    double normalize_x(double dx) const;
    double lnp_repulsive_cylindrical(const pos_t& p1, const pos_t& p2) const;
    double lnp_attractive_cylindrical(const pos_t& p1, const pos_t& p2) const;
    void force_repulsive_cylindrical(const pos_t& p1, const pos_t& p2, double& Fx, double& Fy, double& Fz) const;
    void force_attractive_cylindrical(const pos_t& p1, const pos_t& p2, double& Fx, double& Fy, double& Fz) const;

    const vector<pos_t>& get_pos() const { return poss; }
    const vector<pos_t>& get_forces() const { return forces; }
    const vector<pair<int, int>>& get_edges() const { return edges; }

private:
    double direction_x(double x1, double x2) const;
    double distance_x(double x1, double x2) const;
    vector<pos_t> poss;
    vector<pos_t> forces;
    vector<pair<int, int>> edges;
    vector<pos_t> m;
    vector<pos_t> v;
    int t = 0;
};

#endif // HYPERBOLIC_EMBEDDING_HH

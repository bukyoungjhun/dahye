#include "hyperbolic_embedding.hh"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0.0, 1.0);

double rnd() { return dis(gen); }

void set_seed(unsigned int seed) {
    gen.seed(seed);
}

double GradState::direction_x(double x1, double x2) const {
    // in order to increase the distance, should x1 increase (+1.0) or decrease (-1.0)
    if (abs(x1-x2) < N - abs(x1-x2)) {
        if (x1 > x2) return 1.0;
        else return -1.0;
    }
    else {
        if (x1 > x2) return -1.0;
        else return 1.0;
    }
}

double GradState::distance_x(double x1, double x2) const {
    double dx = abs(x1 - x2);
    return min(dx, N-dx);
}

double GradState::normalize_x(double dx) const {
    double n = static_cast<double>(N);
    while (dx > 0.5 * n) dx -= n;
    while (dx < -0.5 * n) dx += n;
    return dx;
}

template <typename Pos2D>
bool is_inside_2d(const Pos2D& p, const Pos2D& ll, const Pos2D& ur) {
    return p[0] >= ll[0] && p[0] <= ur[0] && p[1] >= ll[1] && p[1] <= ur[1];
}

// log(1 + exp(t)) without overflow. Used for stable Fermi–Dirac log-likelihoods.
static double log1p_exp(double t) {
    if (t > 0.0) return t + log1p(exp(-t));
    return log1p(exp(t));
}

double GradState::lnp_repulsive_cylindrical(const pos_t& p1, const pos_t& p2) const {
    // log-likelihood ln(1-p_ij) = -log(1 + chi^{-beta})
    // Evaluate as -log(1 + exp(-beta log chi)) so chi^{-beta} never overflows
    // when beta is large and chi is tiny (near-coincident nodes).
    double x1 = p1[0], y1 = p1[1], z1 = p1[2];
    double x2 = p2[0], y2 = p2[1], z2 = p2[2];
    double dx = distance_x(x1, x2);
    double dz = z1 - z2;
    double dij = sqrt(dx*dx + dz*dz);
    dij = max(EPSILON, dij);
    double chi = dij / (mu * y1 * y2);
    return -log1p_exp(-beta * log(chi));
}

double GradState::lnp_attractive_cylindrical(const pos_t& p1, const pos_t& p2) const {
    // log-likelihood ln(p_ij) - ln(1-p_ij) = -beta log chi
    // (analytically exact; stays finite whenever chi is clamped away from 0)
    double x1 = p1[0], y1 = p1[1], z1 = p1[2];
    double x2 = p2[0], y2 = p2[1], z2 = p2[2];
    double dx = distance_x(x1, x2);
    double dz = z1 - z2;
    double dij = sqrt(dx*dx + dz*dz);
    dij = max(EPSILON, dij);
    double chi = dij / (mu * y1 * y2);
    return -beta * log(chi);
}

void GradState::force_repulsive_cylindrical(const pos_t& p1, const pos_t& p2, double& Fx, double& Fy, double& Fz) const {
    double x1 = p1[0], y1 = p1[1], z1 = p1[2];
    double x2 = p2[0], y2 = p2[1], z2 = p2[2];
    double dx = distance_x(x1, x2);
    double dz = z1 - z2;
    double dij = sqrt(dx*dx + dz*dz);

    if (dij < EPSILON) {
        zero_force(Fx, Fy, Fz);
        return;
    }

    double f1 = dij/mu/y1/y2;
    double f1_beta = pow(dij/mu/y1/y2, beta);

    double F1 = beta/mu/y1/y2 /f1 /(1 + f1_beta);

    Fx = F1 * dx/dij * direction_x(x1, x2);
    Fy = -F1 * dij/y1;
    Fz = F1 * dz/dij;

    Fx = max(-MAX_FORCE, min(MAX_FORCE, Fx));
    Fy = max(-MAX_FORCE, min(MAX_FORCE, Fy));
    Fz = max(-MAX_FORCE, min(MAX_FORCE, Fz));
}

void GradState::force_attractive_cylindrical(const pos_t& p1, const pos_t& p2, double& Fx, double& Fy, double& Fz) const {
    double x1 = p1[0], y1 = p1[1], z1 = p1[2];
    double x2 = p2[0], z2 = p2[2];
    double dx = distance_x(x1, x2);
    double dz = z1 - z2;
    double dij = sqrt(dx*dx + dz*dz);

    if (dij < EPSILON) {
        zero_force(Fx, Fy, Fz);
        return;
    }

    Fx = -beta/dij * dx/dij * direction_x(x1, x2);
    Fy = beta/y1;
    Fz = -beta/dij * dz/dij;

    Fx = max(-MAX_FORCE, min(MAX_FORCE, Fx));
    Fy = max(-MAX_FORCE, min(MAX_FORCE, Fy));
    Fz = max(-MAX_FORCE, min(MAX_FORCE, Fz));
}

GradState::GradState(int n) : N(n), poss(n), forces(n), m(n), v(n) {
    for (int i = 0; i < n; ++i) {
        m[i] = {0.0, 0.0, 0.0};
        v[i] = {0.0, 0.0, 0.0};
    }
}

void GradState::set_edgelist(const vector<pair<int, int>>& edgelist) {
    edges.clear();
    int max_node = -1;
    for (const auto& edge : edgelist) {
        int u = edge.first;
        int v = edge.second;
        
        if (u < 0 || v < 0) continue;
        max_node = max({max_node, u, v});
        if (u > v) swap(u, v);
        if (u != v)
            edges.emplace_back(u, v);
    }
    if (max_node >= N) {
        N = max_node + 1;
        poss.resize(N);
        forces.resize(N);
        m.resize(N);
        v.resize(N);
        for (int i = 0; i < N; ++i) {
            m[i] = {0.0, 0.0, 0.0};
            v[i] = {0.0, 0.0, 0.0};
        }
    }
}

void GradState::set_pos(int node_idx, double x, double y, double z) {
    if (node_idx < 0 || node_idx >= N) {
        cerr << "Error: node_idx " << node_idx << " out of range [0, " << N << ")\n";
        return;
    }
    while (x < 0) x += N;
    while (x >= N) x -= N;
    y = max(1e-10, y);
    poss[node_idx] = {x, y, z};
}

void GradState::set_positions(const vector<pos_t>& positions) {
    if (static_cast<int>(positions.size()) < N) {
        cerr << "Error: set_positions size " << positions.size() << " < N " << N << "\n";
        return;
    }
    int old_N = N;
    if (static_cast<int>(positions.size()) > N) {
        N = static_cast<int>(positions.size());
        poss.resize(N);
        forces.resize(N);
        m.resize(N);
        v.resize(N);
        for (int i = old_N; i < N; ++i) {
            m[i] = {0.0, 0.0, 0.0};
            v[i] = {0.0, 0.0, 0.0};
        }
    }
    poss = positions;
}

void GradState::set_theta(double new_theta) {
    theta = new_theta;
}

void GradState::set_mu(double new_mu) {
    mu = new_mu;
}

void GradState::set_beta(double new_beta) {
    beta = new_beta;
}

double GradState::get_lnp_exact() {
    double lnp = 0;
    #pragma omp parallel for reduction(+ : lnp) schedule(dynamic)
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < n; ++m) {
            lnp += lnp_repulsive_cylindrical(poss[n], poss[m]);
        }
    }

    for (const auto& [u, v] : edges) {
        lnp += lnp_attractive_cylindrical(poss[u], poss[v]);
    }

    return lnp;
}

double GradState::get_lnp_bh() {
    double lnp = 0;
    pos_t ll = {numeric_limits<double>::max(), numeric_limits<double>::max(), 0};
    pos_t ur = {-numeric_limits<double>::max(), -numeric_limits<double>::max(), 0};
    for (int n = 0; n < N; ++n) {
        ll[0] = min(ll[0], poss[n][0]);
        ll[1] = min(ll[1], poss[n][1]);
        ur[0] = max(ur[0], poss[n][0]);
        ur[1] = max(ur[1], poss[n][1]);
    }
    ll[0] = max(1e-10, ll[0]);
    using pos_2d_t = array<double, 2>;
    using pos_3d_t = array<double, 3>;
    pos_2d_t ll_2d = {ll[0], ll[1]};
    pos_2d_t ur_2d = {ur[0], ur[1]};

    QuadTree<double, double> qt(ll_2d, ur_2d, max_level, N);
    for (int n = 0; n < N; ++n) {
        pos_3d_t p_3d = {poss[n][0], poss[n][1], poss[n][2]};
        qt.put_pos(0, p_3d, 1);
    }

    int interaction_count = 0;
    #pragma omp parallel for reduction(+ : interaction_count, lnp) schedule(dynamic)
    for (int n = 0; n < N; ++n) {
        pos_3d_t cm_3d{0, 0, 0};
        pos_2d_t ll_2d, ur_2d;
        vector<int> stack;
        stack.reserve(N);
        stack.push_back(0);

        while (!stack.empty()) {
            int q = stack.back();
            stack.pop_back();

            auto& dleaves = qt.get_dense_leaves(q);
            if (!dleaves.empty()) {
                for (const auto& dleaf : dleaves) {
                    const pos_3d_t& p2_3d = get<0>(dleaf);
                    double w = get<1>(dleaf);
                    pos_t p2 = {p2_3d[0], p2_3d[1], p2_3d[2]};
                    if (is_same_point(poss[n], p2)) continue; // ignore self-interaction
                    lnp += w * lnp_repulsive_cylindrical(poss[n], p2);
                    ++interaction_count;
                }
            } else {
                qt[q].get_cm(cm_3d);
                qt[q].get_ll(ll_2d);
                qt[q].get_ur(ur_2d);
                double x_cm = cm_3d[0];
                double y_cm = cm_3d[1];
                if (x_cm < 0) x_cm += N;
                double z_cm = cm_3d[2];
                pos_t p2_cm = {x_cm, y_cm, z_cm};
                double cell_size = qt[q].get_w();
                pos_2d_t p_2d = {poss[n][0], poss[n][1]};
                double dx = distance_x(poss[n][0], x_cm);
                double err_f = cell_size / dx;
                bool subdivide = (qt[q].get_count() >= 2)
                    && (err_f > theta || is_inside_2d(p_2d, ll_2d, ur_2d));
                if (subdivide) {
                    size_t leaf = qt.get_leaves(q);
                    for (size_t i = 0; i < 4; ++i) {
                        if (qt[leaf].get_count() > 0)
                            stack.push_back(leaf);
                        ++leaf;
                    }
                } else {
                    double w = qt[q].get_count();
                    if (w == 1 && is_inside_2d(p_2d, ll_2d, ur_2d)) continue; // ignore self-interaction
                    lnp += w * lnp_repulsive_cylindrical(poss[n], p2_cm);
                    ++interaction_count;
                }
            }
        }
    }
    lnp /= 2.0; // each pair calculated twice
    
    for (const auto& [u, v] : edges) {
        lnp += lnp_attractive_cylindrical(poss[u], poss[v]);
    }

    return lnp;
}

void GradState::compute_repulsive_force_exact() {
    for (int n = 0; n < N; ++n) {
        forces[n][0] = 0;
        forces[n][1] = 0;
        forces[n][2] = 0;
    }
    #pragma omp parallel for schedule(dynamic)
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < N; ++m) {
            if (n == m) continue;
            double Fx, Fy, Fz;
            force_repulsive_cylindrical(poss[n], poss[m], Fx, Fy, Fz);
            forces[n][0] += Fx;
            forces[n][1] += Fy;
            forces[n][2] += Fz;
        }
    }
}

int GradState::compute_repulsive_force_bh() {
    pos_t ll = {numeric_limits<double>::max(), numeric_limits<double>::max(), 0};
    pos_t ur = {-numeric_limits<double>::max(), -numeric_limits<double>::max(), 0};
    for (int n = 0; n < N; ++n) {
        ll[0] = min(ll[0], poss[n][0]);
        ll[1] = min(ll[1], poss[n][1]);
        ur[0] = max(ur[0], poss[n][0]);
        ur[1] = max(ur[1], poss[n][1]);
    }
    ll[0] = max(1e-10, ll[0]);
    using pos_2d_t = array<double, 2>;
    using pos_3d_t = array<double, 3>;
    pos_2d_t ll_2d = {ll[0], ll[1]};
    pos_2d_t ur_2d = {ur[0], ur[1]};

    QuadTree<double, double> qt(ll_2d, ur_2d, max_level, N);
    for (int n = 0; n < N; ++n) {
        pos_3d_t p_3d = {poss[n][0], poss[n][1], poss[n][2]};
        qt.put_pos(0, p_3d, 1);
    }

    int interaction_count = 0;
    #pragma omp parallel for reduction(+ : interaction_count) schedule(dynamic)
    for (int n = 0; n < N; ++n) {
        pos_3d_t cm_3d{0, 0, 0};
        pos_2d_t ll_2d, ur_2d;
        vector<int> stack;
        stack.reserve(N);
        forces[n][0] = 0;
        forces[n][1] = 0;
        forces[n][2] = 0;
        stack.push_back(0);

        while (!stack.empty()) {
            int q = stack.back();
            stack.pop_back();

            auto& dleaves = qt.get_dense_leaves(q);
            if (!dleaves.empty()) {
                for (const auto& dleaf : dleaves) {
                    const pos_3d_t& p2_3d = get<0>(dleaf);
                    double w = get<1>(dleaf);
                    pos_t p2 = {p2_3d[0], p2_3d[1], p2_3d[2]};
                    if (is_same_point(poss[n], p2)) continue; // ignore self-interaction
                    double Fx, Fy, Fz;
                    force_repulsive_cylindrical(poss[n], p2, Fx, Fy, Fz);
                    forces[n][0] += w * Fx;
                    forces[n][1] += w * Fy;
                    forces[n][2] += w * Fz;
                    ++interaction_count;
                }
            } else {
                qt[q].get_cm(cm_3d);
                qt[q].get_ll(ll_2d);
                qt[q].get_ur(ur_2d);
                double x_cm = cm_3d[0];
                double y_cm = cm_3d[1];
                if (x_cm < 0) x_cm += N;
                double z_cm = cm_3d[2];
                pos_t p2_cm = {x_cm, y_cm, z_cm};
                double cell_size = qt[q].get_w();
                pos_2d_t p_2d = {poss[n][0], poss[n][1]};
                double dx = distance_x(poss[n][0], x_cm);
                double err_f = cell_size / dx;
                bool subdivide = (qt[q].get_count() >= 2)
                    && (err_f > theta || is_inside_2d(p_2d, ll_2d, ur_2d));
                if (subdivide) {
                    size_t leaf = qt.get_leaves(q);
                    for (size_t i = 0; i < 4; ++i) {
                        if (qt[leaf].get_count() > 0)
                            stack.push_back(leaf);
                        ++leaf;
                    }
                } else {
                    double w = qt[q].get_count();
                    if (w == 1 && is_inside_2d(p_2d, ll_2d, ur_2d)) continue; // ignore self-interaction
                    double Fx, Fy, Fz;
                    force_repulsive_cylindrical(poss[n], p2_cm, Fx, Fy, Fz);
                    forces[n][0] += w * Fx;
                    forces[n][1] += w * Fy;
                    forces[n][2] += w * Fz;
                    ++interaction_count;
                }
            }
        }
    }
    return interaction_count;
}

void GradState::add_attractive_force() {
    for (const auto& [u, v] : edges) {
        double Fx_u, Fy_u, Fz_u, Fx_v, Fy_v, Fz_v;
        force_attractive_cylindrical(poss[u], poss[v], Fx_u, Fy_u, Fz_u);
        forces[u][0] += Fx_u;
        forces[u][1] += Fy_u;
        forces[u][2] += Fz_u;
        force_attractive_cylindrical(poss[v], poss[u], Fx_v, Fy_v, Fz_v);
        forces[v][0] += Fx_v;
        forces[v][1] += Fy_v;
        forces[v][2] += Fz_v;
    }
}

void GradState::add_l1_penalty_z(double penalty_z) {
    for (int n = 0; n < N; ++n) {
        double z = poss[n][2];
        if (z > 0.0) {
            forces[n][2] -= penalty_z;
        } else if (z < 0.0) {
            forces[n][2] += penalty_z;
        }
    }
}

void GradState::add_l2_penalty_z(double penalty_z) {
    for (int n = 0; n < N; ++n) {
        double z = poss[n][2];
        forces[n][2] -= penalty_z * z;
    }
}

void GradState::project_equator() {
    for (int n = 0; n < N; ++n) {
        poss[n][2] = 0.0;
        m[n][2] = 0.0;
        v[n][2] = 0.0;
    }
}

void GradState::step_adam(double lr) {
    ++t;
    const double inv_b1 = 1.0 - pow(beta1, t);
    const double inv_b2 = 1.0 - pow(beta2, t);
    for (int n = 0; n < N; ++n) {
        double g_x = -forces[n][0];
        double g_y = -forces[n][1];
        double g_z = -forces[n][2];
        g_x = max(-grad_clip_value, min(grad_clip_value, g_x));
        g_y = max(-grad_clip_value, min(grad_clip_value, g_y));
        g_z = max(-grad_clip_value, min(grad_clip_value, g_z));
        m[n][0] = beta1 * m[n][0] + (1.0 - beta1) * g_x;
        m[n][1] = beta1 * m[n][1] + (1.0 - beta1) * g_y;
        m[n][2] = beta1 * m[n][2] + (1.0 - beta1) * g_z;
        v[n][0] = beta2 * v[n][0] + (1.0 - beta2) * g_x * g_x;
        v[n][1] = beta2 * v[n][1] + (1.0 - beta2) * g_y * g_y;
        v[n][2] = beta2 * v[n][2] + (1.0 - beta2) * g_z * g_z;
    }
    double rotation_angle = (rnd() * 2 - 1) * N;
    for (int n = 0; n < N; ++n) {
        double m_hat_x = m[n][0] / inv_b1;
        double m_hat_y = m[n][1] / inv_b1;
        double m_hat_z = m[n][2] / inv_b1;
        double v_hat_x = v[n][0] / inv_b2;
        double v_hat_y = v[n][1] / inv_b2;
        double v_hat_z = v[n][2] / inv_b2;
        double x_new = poss[n][0] - lr * m_hat_x / (sqrt(v_hat_x) + epsilon) + rotation_angle;
        double y_new = poss[n][1] - lr * m_hat_y / (sqrt(v_hat_y) + epsilon);
        double z_new = poss[n][2] - lr * m_hat_z / (sqrt(v_hat_z) + epsilon);
        y_new = max(1e-10, y_new);
        while (x_new < 0) x_new += N;
        while (x_new >= N) x_new -= N;
        poss[n][0] = x_new;
        poss[n][1] = y_new;
        poss[n][2] = z_new;
    }
}

void GradState::step_adam_decay_z(double lr, double lambda) {
    ++t;
    const double inv_b1 = 1.0 - pow(beta1, t);
    const double inv_b2 = 1.0 - pow(beta2, t);
    for (int n = 0; n < N; ++n) {
        double g_x = -forces[n][0];
        double g_y = -forces[n][1];
        double g_z = -forces[n][2];
        g_x = max(-grad_clip_value, min(grad_clip_value, g_x));
        g_y = max(-grad_clip_value, min(grad_clip_value, g_y));
        g_z = max(-grad_clip_value, min(grad_clip_value, g_z));
        m[n][0] = beta1 * m[n][0] + (1.0 - beta1) * g_x;
        m[n][1] = beta1 * m[n][1] + (1.0 - beta1) * g_y;
        m[n][2] = beta1 * m[n][2] + (1.0 - beta1) * g_z;
        v[n][0] = beta2 * v[n][0] + (1.0 - beta2) * g_x * g_x;
        v[n][1] = beta2 * v[n][1] + (1.0 - beta2) * g_y * g_y;
        v[n][2] = beta2 * v[n][2] + (1.0 - beta2) * g_z * g_z;
    }
    double rotation_angle = (rnd() * 2 - 1) * N;
    for (int n = 0; n < N; ++n) {
        double m_hat_x = m[n][0] / inv_b1;
        double m_hat_y = m[n][1] / inv_b1;
        double v_hat_x = v[n][0] / inv_b2;
        double v_hat_y = v[n][1] / inv_b2;
        double x_new = poss[n][0] - lr * m_hat_x / (sqrt(v_hat_x) + epsilon) + rotation_angle;
        double y_new = poss[n][1] - lr * m_hat_y / (sqrt(v_hat_y) + epsilon);
        double z_new = poss[n][2] * (1.0 - lambda * lr);
        y_new = max(1e-10, y_new);
        while (x_new < 0) x_new += N;
        while (x_new >= N) x_new -= N;
        poss[n][0] = x_new;
        poss[n][1] = y_new;
        poss[n][2] = z_new;
    }
}
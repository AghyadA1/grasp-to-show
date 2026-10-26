// Soft-finger grasp stability: friction utilisation of a two-finger grasp under a load.
// C++ version of grasp2show/stability.py::utilisation_wrench; derivation in docs/physics.md.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "vec.hpp"

namespace g2s {

struct StabilityParams {
    double mu = 0.5;            // sliding friction
    double mu_t = 0.005;        // torsional friction coefficient (m)
    double finger_force = 20.0; // N per finger
};

// Utilisation u for one grasp: contacts c1, c2 on the finger axis y (unit), centre of mass
// com, external force F and torque Tq about the COM acting on the object (object frame).
// u <= 1: the contacts can cancel the load; +inf: they cannot (a normal force would be <= 0).
inline double utilisation(const Vec3& c1, const Vec3& c2, const Vec3& y, const Vec3& com, const Vec3& F,
                          const Vec3& Tq, const StabilityParams& p, int iters = 40) {
    const double inf = std::numeric_limits<double>::infinity();
    const Vec3 c = 0.5 * (c1 + c2);
    const double w = norm(c2 - c1);
    const Vec3 f = -F;                                   // contact force needed
    const Vec3 tau = -(cross(com - c, F) + Tq);          // contact torque about c
    const double f_y = dot(f, y);
    const Vec3 f_t = f - f_y * y;
    const double tau_y = dot(tau, y);
    const Vec3 tau_p = tau - tau_y * y;
    const Vec3 d = (2.0 / std::max(w, 1e-9)) * cross(tau_p, y);
    const Vec3 t1 = 0.5 * (f_t - d), t2 = 0.5 * (f_t + d);
    const double n1 = p.finger_force + 0.5 * f_y, n2 = p.finger_force - 0.5 * f_y;
    if (!(n1 > 0.0 && n2 > 0.0 && w > 1e-6)) return inf;
    const double a1 = norm(t1) / p.mu, a2 = norm(t2) / p.mu;   // |t_i| / mu
    const double abs_tau_y = std::fabs(tau_y);

    auto feasible = [&](double u) {
        const double r1 = p.mu_t * std::sqrt(std::max((u * n1) * (u * n1) - a1 * a1, 0.0));
        const double r2 = p.mu_t * std::sqrt(std::max((u * n2) * (u * n2) - a2 * a2, 0.0));
        return r1 + r2 >= abs_tau_y;
    };
    double lo = std::max(a1 / n1, a2 / n2);
    if (feasible(lo)) return lo;
    double hi = lo + 1.0 + abs_tau_y / (p.mu_t * std::min(n1, n2));
    for (int i = 0; i < iters; ++i) {   // feasible() is monotone in u: bisect
        const double mid = 0.5 * (lo + hi);
        if (feasible(mid)) hi = mid; else lo = mid;
    }
    return hi;
}

}  // namespace g2s

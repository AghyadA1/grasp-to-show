// Analytic IK of the Franka Panda for a given q7 (up to 4 solutions), TCP frame of the Franka Hand.
// C++ version of grasp2show/franka_ik.py. Geometric solver of He & Liu (ICCMA 2021), following the
// implementation in https://github.com/CoMMALab/frantik (Apache-2.0), rewritten without Eigen and
// with two changes: the shoulder-singularity shortcut only at an exactly vertical joint-2 axis, and
// every solution verified by forward kinematics.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "vec.hpp"

namespace g2s {
namespace panda {

constexpr double D1 = 0.3330, D3 = 0.3160, D5 = 0.3840, D7E = 0.2104, A4 = 0.0825, A7 = 0.0880;
constexpr std::array<double, 7> Q_MIN = {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
constexpr std::array<double, 7> Q_MAX = {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
constexpr double NEAR_ONE = 1.0 - 1e-10;
constexpr double FK_TOL = 1e-6;

using Q = std::array<double, 7>;
using Pose = std::array<double, 16>;  // row-major 4x4

inline bool within(double q, int j) { return q > Q_MIN[j] && q < Q_MAX[j]; }

// Forward kinematics to the TCP (modified DH, Craig convention), row-major 4x4.
inline Pose fk(const Q& q) {
    const double a[7] = {0, 0, 0, A4, -A4, 0, A7};
    const double d[7] = {D1, 0, D3, 0, D5, 0, 0};
    const double al[7] = {0, -kPi / 2, kPi / 2, kPi / 2, -kPi / 2, kPi / 2, kPi / 2};
    double T[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    auto mul = [](double A[4][4], const double B[4][4]) {
        double R[4][4];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                R[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] + A[i][3] * B[3][j];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) A[i][j] = R[i][j];
    };
    for (int i = 0; i < 7; ++i) {
        const double ct = std::cos(q[i]), st = std::sin(q[i]), ca = std::cos(al[i]), sa = std::sin(al[i]);
        const double A[4][4] = {{ct, -st, 0, a[i]},
                                {st * ca, ct * ca, -sa, -sa * d[i]},
                                {st * sa, ct * sa, ca, ca * d[i]},
                                {0, 0, 0, 1}};
        mul(T, A);
    }
    const double c = std::cos(-kPi / 4), s = std::sin(-kPi / 4);
    const double F[4][4] = {{c, -s, 0, 0}, {s, c, 0, 0}, {0, 0, 1, D7E}, {0, 0, 0, 1}};
    mul(T, F);
    Pose out;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out[4 * i + j] = T[i][j];
    return out;
}

// All IK solutions for one TCP pose (row-major 4x4) and one q7. Invalid rows are NaN.
inline std::array<Q, 4> ik(const Pose& tf, double q7, double q1_singular = 0.0) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::array<Q, 4> out;
    for (auto& r : out) r.fill(nan);
    if (!within(q7, 6)) return out;

    const double LL24 = A4 * A4 + D3 * D3, LL46 = A4 * A4 + D5 * D5;
    const double L24 = std::sqrt(LL24), L46 = std::sqrt(LL46);
    const double thH46 = std::atan(D5 / A4), th342 = std::atan(D3 / A4), th46H = kPi / 2 - std::atan(D5 / A4);

    const Mat3 R = Mat3::from_cols({tf[0], tf[4], tf[8]}, {tf[1], tf[5], tf[9]}, {tf[2], tf[6], tf[10]});
    const Vec3 z_ee{tf[2], tf[6], tf[10]}, p_ee{tf[3], tf[7], tf[11]};
    const Vec3 p_7 = p_ee - D7E * z_ee;
    const Vec3 x_6 = normalized(R * Vec3{std::cos(q7 - kPi / 4), -std::sin(q7 - kPi / 4), 0.0});
    const Vec3 p_6 = p_7 - A7 * x_6;

    // q4
    const Vec3 p_2{0, 0, D1};
    const Vec3 V26 = p_6 - p_2;
    const double LL26 = dot(V26, V26), L26 = std::sqrt(LL26);
    if (L24 + L46 < L26 || L24 + L26 < L46 || L26 + L46 < L24) return out;
    const double th246 = std::acos(std::clamp((LL24 + LL46 - LL26) / (2 * L24 * L46), -1.0, 1.0));
    const double q4 = th246 + thH46 + th342 - 2 * kPi;
    if (!within(q4, 3)) return out;

    // q6 (two branches)
    const double th462 = std::acos(std::clamp((LL26 + LL46 - LL24) / (2 * L26 * L46), -1.0, 1.0));
    const double th26H = th46H + th462;
    const double D26 = -L26 * std::cos(th26H);
    const Vec3 Z_6 = cross(z_ee, x_6), Y_6 = cross(Z_6, x_6);
    const Mat3 R_6 = Mat3::from_cols(x_6, normalized(Y_6), normalized(Z_6));
    const Vec3 V_6_62 = R_6.tmul(-V26);
    const double Phi6 = std::atan2(V_6_62.y, V_6_62.x);
    const double Theta6 = std::asin(std::clamp(D26 / std::hypot(V_6_62.x, V_6_62.y), -1.0, 1.0));
    double q6[2] = {kPi - Theta6 - Phi6, Theta6 - Phi6};

    const double thP26 = 3 * kPi / 2 - th462 - th246 - th342;
    const double thP = kPi - thP26 - th26H;
    const double LP6 = L26 * std::sin(thP26) / std::sin(thP);

    for (int i = 0; i < 2; ++i) {
        if (q6[i] <= Q_MIN[5]) q6[i] += 2 * kPi;
        else if (q6[i] >= Q_MAX[5]) q6[i] -= 2 * kPi;
        if (!within(q6[i], 5)) continue;

        const Vec3 z_5 = R_6 * Vec3{std::sin(q6[i]), std::cos(q6[i]), 0.0};
        const Vec3 V2P = p_6 - LP6 * z_5 - p_2;
        const double L2P = norm(V2P);
        double q1[2], q2[2];
        if (std::fabs(V2P.z / L2P) > NEAR_ONE) {
            q1[0] = q1[1] = q1_singular;
            q2[0] = q2[1] = 0.0;
        } else {
            q1[0] = std::atan2(V2P.y, V2P.x);
            q2[0] = std::acos(std::clamp(V2P.z / L2P, -1.0, 1.0));
            q1[1] = q1[0] + (q1[0] < 0 ? kPi : -kPi);
            q2[1] = -q2[0];
        }
        for (int j = 0; j < 2; ++j) {
            if (!within(q1[j], 0) || !within(q2[j], 1)) continue;
            // q3
            const Vec3 z_3 = normalized(V2P);
            const Vec3 y_3 = normalized(-cross(V26, V2P));
            const Vec3 x_3 = cross(y_3, z_3);
            const double c1 = std::cos(q1[j]), s1 = std::sin(q1[j]), c2 = std::cos(q2[j]), s2 = std::sin(q2[j]);
            Mat3 R_2;
            R_2.m[0][0] = c1 * c2; R_2.m[0][1] = -c1 * s2; R_2.m[0][2] = -s1;
            R_2.m[1][0] = s1 * c2; R_2.m[1][1] = -s1 * s2; R_2.m[1][2] = c1;
            R_2.m[2][0] = -s2;     R_2.m[2][1] = -c2;      R_2.m[2][2] = 0.0;
            const Vec3 x_2_3 = R_2.tmul(x_3);
            const double q3 = std::atan2(x_2_3.z, x_2_3.x);
            if (!within(q3, 2)) continue;
            // q5
            const Vec3 VH4 = p_2 + D3 * z_3 + A4 * x_3 - p_6 + D5 * z_5;
            const double c6 = std::cos(q6[i]), s6 = std::sin(q6[i]);
            Mat3 R_5_6;
            R_5_6.m[0][0] = c6; R_5_6.m[0][1] = -s6;
            R_5_6.m[1][2] = -1.0;
            R_5_6.m[2][0] = s6; R_5_6.m[2][1] = c6;
            const Mat3 R_5 = R_6 * R_5_6.transpose();
            const Vec3 V_5_H4 = R_5.tmul(VH4);
            const double q5 = -std::atan2(V_5_H4.y, V_5_H4.x);
            if (!within(q5, 4)) continue;

            const Q sol = {q1[j], q2[j], q3, q4, q5, q6[i], q7};
            const Pose T = fk(sol);   // safety net: keep only exact solutions
            double err = 0;
            for (int k = 0; k < 16; ++k) err = std::max(err, std::fabs(T[k] - tf[k]));
            if (err <= FK_TOL) out[2 * i + j] = sol;
        }
    }
    return out;
}

}  // namespace panda
}  // namespace g2s

// Python bindings (nanobind): batched versions of the stability model and the analytic IK.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <stdexcept>

#include "franka_ik.hpp"
#include "stability.hpp"

namespace nb = nanobind;
using namespace nb::literals;

template <nb::ssize_t... Shape>
using In = nb::ndarray<const double, nb::numpy, nb::shape<Shape...>, nb::c_contig, nb::device::cpu>;
using Out = nb::ndarray<nb::numpy, double>;

namespace {

g2s::Vec3 row(const In<-1, 3>& a, size_t i) { return {a(i, 0), a(i, 1), a(i, 2)}; }

Out utilisation_wrench(In<-1, 3> c1, In<-1, 3> c2, In<-1, 3> y, In<3> com, In<-1, 3> F, In<-1, 3> Tq,
                       double mu, double mu_t, double finger_force, int iters) {
    const size_t n = c1.shape(0);
    if (c2.shape(0) != n || y.shape(0) != n || F.shape(0) != n || Tq.shape(0) != n)
        throw std::invalid_argument("all (N, 3) inputs must have the same N");
    const g2s::StabilityParams p{mu, mu_t, finger_force};
    const g2s::Vec3 cm{com(0), com(1), com(2)};
    double* u = new double[n];
    {
        nb::gil_scoped_release release;
        for (size_t i = 0; i < n; ++i)
            u[i] = g2s::utilisation(row(c1, i), row(c2, i), row(y, i), cm, row(F, i), row(Tq, i), p, iters);
    }
    nb::capsule owner(u, [](void* ptr) noexcept { delete[] static_cast<double*>(ptr); });
    return Out(u, {n}, owner);
}

Out ik(In<-1, 4, 4> T, In<-1, -1> q7, double q1_singular) {
    const size_t n = T.shape(0), k = q7.shape(1);
    if (q7.shape(0) != n) throw std::invalid_argument("q7 must have shape (N, K)");
    double* q = new double[n * k * 4 * 7];
    {
        nb::gil_scoped_release release;
        for (size_t i = 0; i < n; ++i) {
            g2s::panda::Pose tf;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) tf[4 * r + c] = T(i, r, c);
            for (size_t j = 0; j < k; ++j) {
                const auto sols = g2s::panda::ik(tf, q7(i, j), q1_singular);
                double* dst = q + ((i * k + j) * 4) * 7;
                for (int b = 0; b < 4; ++b)
                    for (int a = 0; a < 7; ++a) dst[7 * b + a] = sols[b][a];
            }
        }
    }
    nb::capsule owner(q, [](void* ptr) noexcept { delete[] static_cast<double*>(ptr); });
    return Out(q, {n, k, size_t(4), size_t(7)}, owner);
}

}  // namespace

NB_MODULE(g2s_cpp, m) {
    m.doc() = "grasp2show C++ kernels: soft-finger stability and analytic Franka IK";
    m.def("utilisation_wrench", &utilisation_wrench, "c1"_a, "c2"_a, "y"_a, "com"_a, "F"_a, "Tq"_a,
          "mu"_a, "mu_t"_a, "finger_force"_a, "iters"_a = 40,
          "Friction utilisation u (N,) of two-finger soft contacts under external loads F, Tq.");
    m.def("ik", &ik, "T"_a, "q7"_a, "q1_singular"_a = 0.0,
          "All analytic IK solutions q (N, K, 4, 7) for TCP poses T (N, 4, 4) and q7 (N, K); NaN = invalid.");
}

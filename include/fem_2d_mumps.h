#ifndef FEM_2D_MUMPS_H
#define	FEM_2D_MUMPS_H

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

#include "mumps_solver.h"
#include "base_functions.h"
#include "quadratures.h"
#include "bspline_fem_solver.h"

template<class T, int degree>
class fem_2d_mumps : public bspline_fem_solver<T> {
    typedef typename bspline_fem_solver<T>::param_fun param_fun;

    param_fun _a, _b, _c, _f;
    T _beta, _gamma;

    int _N;
    T _h, _h2;

    std::vector<int> irn, jcn;
    std::vector<T> A;
    std::vector<T> RHS;
    std::vector<T> tt;

    /**
     * Knot vector (open uniform)
     * @param i
     * @return
     */
    T
    t(int i)
    {
        if (i <= degree) return T(0);
        if (i <= _N + degree) return (i - degree) * _h;
        return T(1);
    }

    T
    N(T x, int i)
    {
        return ::N<T, degree > (x, i, &tt[0]);
    }

    T
    dN(T x, int i)
    {
        return ::dN<T, degree > (x, i, &tt[0]);
    }

    T
    fun_b(T x1, T x2, int f1x, int f1y, int f2x, int f2y)
    {
        return dN(x1, f1x) * N(x2, f1y) * dN(x1, f2x) * N(x2, f2y)
                + N(x1, f1x) * dN(x2, f1y) * N(x1, f2x) * dN(x2, f2y);
    }

    T
    fun_rhs(T x1, T x2, int fx, int fy)
    {
        return _f(x1, x2) * N(x1, fx) * N(x2, fy);
    }

    T
    eval_b(int f1x, int f1y, int f2x, int f2y)
    {
        int ax = std::min(f1x, f2x);
        int bx = std::max(f1x, f2x);
        int ay = std::min(f1y, f2y);
        int by = std::max(f1y, f2y);

        T sum(0);

        for (int px = bx; px <= ax + degree; ++px)
        {
            for (int py = by; py <= ay + degree; ++py)
            {

                sum += integrate_2d(boost::bind(&fem_2d_mumps<T, degree>::fun_b,
                                                this, _1, _2, f1x, f1y, f2x, f2y),
                                    t(px), t(px + 1),
                                    t(py), t(py + 1));

            }
        }
        return sum;
    }

    T
    eval_rhs(int j1, int j2)
    {
        T sum(0);
        for (int i1 = 0; i1 <= degree; ++i1)
            for (int i2 = 0; i2 <= degree; ++i2)
            {
                sum += integrate_2d(boost::bind(&fem_2d_mumps<T, degree>::fun_rhs,
                                                this, _1, _2, j1, j2),
                                    t(j1 + i1), t(j1 + i1 + 1),
                                    t(j2 + i2), t(j2 + i2 + 1));
            }
        return sum + (j1 == _N + degree - 1 ? _gamma : 0);
    }

    mumps_solver solver;

public:

    fem_2d_mumps(int N)
    :
    _a(zero), _b(zero), _c(zero), _N(N)
    {
        _h = T(1.0) / _N;
        _h2 = _h * _h;
    }

    int
    xy_to_idx(int x, int y, int n)
    {
        return y * n + x;
    }

    void
    init(param_fun aa = zero,
         param_fun bb = zero,
         param_fun cc = zero,
         param_fun ff = zero,
         T beta = 0,
         T gamma = 0)
    {
        _a = aa;
        _b = bb;
        _c = cc;
        _f = ff;
        _beta = beta;
        _gamma = gamma;
        //        irn.push_back(1);
        //        jcn.push_back(1);
        //        A.push_back(1);
        //        RHS.push_back(0);
        for (int i = 0; i < _N + 2 * degree + 1; ++i)
            tt.push_back(t(i));

        for (int x1 = 0; x1 < _N + degree; ++x1)
        {
            for (int x2 = 0; x2 < _N + degree; ++x2)
            {
                int fun1_idx = xy_to_idx(x1, x2, _N + degree);
                for (int k1 = -degree; k1 <= degree; ++k1)
                {
                    int j1 = x1 + k1;
                    if (j1 < 0 || j1 >= _N + degree) continue;

                    for (int k2 = -degree; k2 <= degree; ++k2)
                    {
                        int j2 = x2 + k2;
                        if (j2 < 0 || j2 >= _N + degree) continue;

                        int fun2_idx = xy_to_idx(j1, j2, _N + degree);

                        irn.push_back(fun1_idx + 1);
                        jcn.push_back(fun2_idx + 1);

                        A.push_back(eval_b(x1, x2, j1, j2));
                    }
                }
                RHS.push_back(eval_rhs(x1, x2));
            }
        }
    }

    void
    eliminate()
    {
        solver.initialize((_N + degree)*(_N + degree), A.size(), &irn[0],
                          &jcn[0], &A[0], &RHS[0]);
        solver.analysis();
    }

    void
    solve_last_equation()
    {
        solver.factorization();
    }

    void
    backward_substitution()
    {
        solver.solve();
    }

    void
    print()
    {
        print_A();
        print_rhs();

        for (int i = 0; i <= degree; ++i)
        {
            std::cout << "\n\n ->> " << t(1 + i) << ' ' << t(1 + i + 1) << ' '
                    << integrate(boost::bind(&fem_2d_mumps<T, degree>::fun_rhs,
                                             this, _1, 1),
                                 t(1 + i), t(1 + i + 1)) << '\n';
        }
    }

    void
    print_A()
    {
        T aaa[9][9];
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 9; ++j)
                aaa[i][j] = 0;
        for (unsigned i = 0; i < A.size(); ++i)
            aaa[irn[i] - 1][jcn[i] - 1] = A[i];
        std::cout << "------A-------\n";
        for (int i = 0; i < 9; ++i)
        {
            for (int j = 0; j < 9; ++j)
            {
                std::cout << boost::format("%9g ") % aaa[i][j];
            }
            std::cout << '\n';
        }
    }

    void
    print_rhs()
    {
        std::cout << "------RHS-------\n";
        for (unsigned i = 0; i < RHS.size(); ++i)
        {
            std::cout << boost::format("%9g\n") % RHS[i];
        }
    }

    void
    debug()
    {
        //        print_A();
        print_rhs();
    }

    void
    solve() { }

    void
    print_result(std::ostream &ostr)
    {
        for (double x = 0; x < 1; x += 0.005)
        {
            ostr << x << ' ' << get_val(x) << '\n';
        }
    }

    T
    get_val(T x)
    {
        T sum(0);
        for (unsigned j = 0; j < RHS.size(); ++j)
        {
            sum += RHS[j] * N(x, j);
        }
        return sum;
    }

    T
    error(param_fun)
    {
        return 0;
    }
};


#endif	/* FEM_2D_H */


/*
 * File:   FEM_1D_mumps.h
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 */

#ifndef FEM_1D_MUMPS_H
#define	FEM_1D_MUMPS_H

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
class FEM_1D_mumps : public bspline_fem_solver<T>{
    typedef typename bspline_fem_solver<T>::param_fun param_fun;

    param_fun _a, _b, _c, _f;
    T _beta, _gamma;

    int _N;
    T _h, _h2;

    std::vector<int> irn, jcn;
    std::vector<T> A;
    std::vector<std::vector<T> > RHSes;
    std::vector<T> tt;

    mumps_solver solver_;

    /**
     * Knot vector (open uniform)
     * @param i
     * @return
     */
    T
    t(int i)
    {
        // temporary implementation for linear B splines
        // TODO generalize for quadratic and cubic (maybe more)
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
    fun_b(T x, int i, int j)
    {
        return _a(x) * dN(x, i) * dN(x, j)
                + _b(x) * dN(x, i) * N(x, j)
                + _c(x) * N(x, i) * N(x, j);
    }

    T
    fun_rhs(T x, int i)
    {
        return _f(x) * N(x, i);
    }

    T
    eval_b(int i, int j)
    {
        int a = std::min(i, j);
        int b = std::max(i, j);
        T sum(0);
        for (int p = b; p <= a + degree; ++p)
        {
            sum += integrate(boost::bind(&FEM_1D_mumps<T, degree>::fun_b,
                                         this, _1, i, j),
                             t(p), t(p + 1));
        }
        return sum + (i == j && j == _N + degree - 1 ? _beta : 0);
    }

    T
    eval_rhs(int j)
    {
        T sum(0);
        for (int i = 0; i <= degree; ++i)
        {
            sum += integrate(boost::bind(&FEM_1D_mumps<T, degree>::fun_rhs,
                                         this, _1, j),
                             t(j + i), t(j + i + 1));
        }
        return sum + (j == _N + degree - 1 ? _gamma : 0);
    }

  public:

    FEM_1D_mumps(int N, int rhs_cnt = 1)
    :
    _a(zero), _b(zero), _c(zero), _N(N), RHSes(rhs_cnt)
    {
        _h = T(1.0) / _N;
        _h2 = _h * _h;
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
        irn.push_back(1);
        jcn.push_back(1);
        A.push_back(1);
        for (int i = 0; i < _N + 2 * degree + 1; ++i)
            tt.push_back(t(i));
        for (int i = 1; i < _N + degree; ++i)
        {
            for (int k = -degree; k <= degree; ++k)
            {
                int j = i + k;
                if (j < 0 || j >= _N + degree) continue;
                irn.push_back(i + 1);
                jcn.push_back(j + 1);
                A.push_back(eval_b(j, i));
            }
        }
        for (unsigned r = 0 ; r < RHSes.size() ; ++r)
        {
            std::vector<T> &RHS = RHSes[r];
            RHS.push_back(0);
            for (int i = 1; i < _N + degree; ++i)
            {
                RHS.push_back(eval_rhs(i));
            }
        }
    }

    void
    print()
    {
        print_A();
        print_rhs();

        for (int i = 0; i <= degree; ++i)
        {
            std::cout << "\n\n ->> " << t(1 + i) << ' ' << t(1 + i + 1) << ' '
                    << integrate(boost::bind(&FEM_1D_mumps<T, degree>::fun_rhs,
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
        std::vector<T> &RHS = RHSes[0];
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
    factorize_matrix()
    {
        solver_.initialize(_N + degree, A.size(), &irn[0],
                          &jcn[0], &A[0]);
        solver_.factorize();
    }

    void
    solve()
    {
        for (unsigned i = 0; i<RHSes.size() ; ++i)
            solver_.solve(&RHSes[i][0]);
        solver_.finalize();
    }

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
        std::vector<T> &RHS = RHSes[0];
        T sum(0);
        for (unsigned j = 0; j < RHS.size(); ++j)
        {
            sum += RHS[j] * N(x, j);
        }
        return sum;
    }

    T
    error(param_fun fun)
    {
        T diff(0), sum(0);
        for (double x = 0; x < 1; x += 0.005)
        {
            diff += pow(fun(x) - get_val(x), 2);
            sum += pow(fun(x), 2);
        }
        return diff / sum;
    }
};


#endif	/* FEM_1D_H */


/*
 * File:   bspline_fem_solver.h
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 */

#ifndef BSPLINE_FEM_SOLVER_H
#define	BSPLINE_FEM_SOLVER_H

#include <iostream>
#include <string>

template<class T>
T
zero(T)
{
    return T(0);
}

template<class T>
class bspline_fem_solver {
  public:
    typedef T(*param_fun)(T);

    virtual void init(param_fun aa = zero,
                      param_fun bb = zero,
                      param_fun cc = zero,
                      param_fun ff = zero,
                      T beta = 0,
                      T gamma = 0) = 0;

    virtual void factorize_matrix() = 0;

    virtual void solve() = 0;

    virtual T error(param_fun ideal_solution) = 0;

    virtual void print_result(std::ostream &ostr) = 0;

    virtual void debug() = 0;

    virtual ~bspline_fem_solver() {};
};

#endif	/* BSPLINE_FEM_SOLVER_H */


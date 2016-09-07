/*
 * File:   CUDA_solver.h
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 */

#ifndef CUDA_SOLVER_H
#define	CUDA_SOLVER_H

#include <bspline_fem_solver.h>
#include <CUDA_interface.h>

template<class T, unsigned degree>
class CUDA_solver : public bspline_fem_solver<T> {
    typedef typename bspline_fem_solver<T>::param_fun param_fun;

    unsigned _N;
    unsigned _RHS_cnt;
  public:

    CUDA_solver(unsigned n, unsigned rhs_cnt) : _N(n), _RHS_cnt(rhs_cnt)
    {
        CUDA_prepare_device<T>(degree, _N, _RHS_cnt);
    }

    virtual
    ~CUDA_solver()
    {
        cleanup_device();
    }

    virtual void
    init(param_fun,
         param_fun,
         param_fun,
         param_fun,
         T,
         T)
    {
        CUDA_init_fronts<T > (degree, _N, _RHS_cnt);
    }

    virtual void
    factorize_matrix()
    {
        CUDA_factorize_matrix<T>(degree, _N);
    }

    virtual void
    solve()
    {
        CUDA_solve<T > (degree, _N, _RHS_cnt);
    }

    virtual T
    error(param_fun)
    {
        return CUDA_error<T > (degree, _N);
    }

    virtual void
    print_result(std::ostream &ostr)
    {
        CUDA_print_result<T > (degree, _N, ostr);
    }

    virtual void
    debug()
    {
        CUDA_debug<T>(degree, _N);
    }
};

#endif	/* CUDA_SOLVER_H */


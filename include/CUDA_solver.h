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
  public:

    CUDA_solver(unsigned n) : _N(n)
    {
        CUDA_prepare_device(degree, _N);
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
        CUDA_init_fronts(degree, _N);
    }

    virtual void
    eliminate()
    {
        CUDA_eliminate(degree, _N);
    }

    virtual void
    solve_last_equation()
    {
        CUDA_solve_last_equation(degree, _N);
    }
    
    virtual void
    backward_substitution()
    {
        CUDA_backward_substitution(degree, _N);
    }
    
    virtual T
    error(param_fun)
    {
        return CUDA_error(degree, _N);
    }

    virtual void
    print_result(std::ostream &ostr)
    {
        CUDA_print_result(degree, _N, ostr);
    }
    
    virtual void
    debug()
    {
        CUDA_debug(degree, _N);
    }
};

#endif	/* CUDA_SOLVER_H */


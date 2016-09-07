/*
 * File:   solver_factory.h
 * Author: Krzysztof Kuźnik <kmkuznik at gmail.com>
 */

#ifndef SOLVER_FACTORY_H
#define	SOLVER_FACTORY_H

#include <bspline_fem_solver.h>
#include <FEM_1D_mumps.h>
#include <CUDA_solver.h>

template<class T, unsigned degree>
bspline_fem_solver<T>
*create_solver(const std::string &name, unsigned intervals, int rhs_cnt)
{
    if(name == "MUMPS")
        return new FEM_1D_mumps<T,degree>(intervals, rhs_cnt);
    if(name == "CUDA")
        return new CUDA_solver<T, degree>(intervals, rhs_cnt);

    return 0;
}



#endif	/* SOLVER_FACTORY_H */


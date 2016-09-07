/*
 * File:   mumps_solver.h
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 */

#ifndef MUMPS_SOLVER_H
#define	MUMPS_SOLVER_H

#include "dmumps_c.h"

#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

#define ICNTL(I) icntl[(I)-1]

class mumps_solver {
  private:
    DMUMPS_STRUC_C id;

public:
    void initialize(int N,
                    int NZ,
                    int *irn,
                    int *jcn,
                    double *A);

    void
    factorize()
    {
        id.job = 1;
        dmumps_c(&id);
        id.job = 2;
        dmumps_c(&id);
    }

    double *
    solve(double *RHS)
    {
        id.rhs = RHS;
        id.job = 3;
        dmumps_c(&id);

        return id.rhs;
    }

    void
    finalize() {
        id.job = JOB_END;
        dmumps_c(&id);
    }
};


#endif	/* MUMPS_SOLVER_H */


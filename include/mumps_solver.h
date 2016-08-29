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

    void
    start()
    {
        id.job = 3;
        dmumps_c(&id);
    }

    void
    stop()
    {
        id.job = JOB_END;
        dmumps_c(&id);
    }


public:
    void initialize(int N,
                    int NZ,
                    int *irn,
                    int *jcn,
                    double *A,
                    double *RHS);

    void
    analysis()
    {
        id.job = 1;
        dmumps_c(&id);
    }

    void
    factorization()
    {
        id.job = 2;
        dmumps_c(&id);
    }

    double *
    solve()
    {
        this->start();
        this->stop();
        return id.rhs;
    }

};


#endif	/* MUMPS_SOLVER_H */


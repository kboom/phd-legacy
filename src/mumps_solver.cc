/*
 * File:   mumps_solver.cc
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 */

#include <mumps_solver.h>

void mumps_solver::initialize(int N,
                              int NZ,
                              int* irn,
                              int* jcn,
                              double* A){
    id.job = JOB_INIT;
    id.par = 1;
    id.sym = 0;
    id.comm_fortran = USE_COMM_WORLD;
    dmumps_c(&id);

    id.n = N;
    id.nz = NZ;
    id.irn = irn;
    id.jcn = jcn;
    id.a = A;

    /* No outputs */
    id.ICNTL(1) = -1;
    id.ICNTL(2) = -1;
    id.ICNTL(3) = -1;
    id.ICNTL(4) = 0;
    id.ICNTL(5) = 0;
    id.ICNTL(18) = 0;
}

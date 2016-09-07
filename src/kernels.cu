
#include <config.h>
#include <CUDA_interface.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <boost/format.hpp>
#include <device/utils.cuh>
#include <device/base_functions.cuh>
#include <device/matrix_utils.cuh>
#include "quadratures.h"

#if __CUDA_ARCH__ >= 200
#define THREADS_PER_BLOCK 16
#else
#define THREADS_PER_BLOCK 16
#endif

__constant__ int QPC;
__constant__ FEM_PRECISION Qp[8];
__constant__ FEM_PRECISION Qw[8];

__constant__ int QPC_e;
__constant__ FEM_PRECISION Qp_e[8];
__constant__ FEM_PRECISION Qw_e[8];

__device__ FEM_PRECISION *d_B[2];
__device__ FEM_PRECISION *d_RHS;
__device__ FEM_PRECISION *d_assembled;
__device__ FEM_PRECISION *d_not_assembled;


__device__ FEM_PRECISION *d_Nvals[2];
__device__ FEM_PRECISION *d_dNvals[2];

__device__ FEM_PRECISION *d_Nvals_err[2];
__device__ FEM_PRECISION *d_dNvals_err[2];

__device__ FEM_PRECISION *d_abssicas;
__device__ FEM_PRECISION *d_fun_vals;
__device__ FEM_PRECISION *d_der_vals;

#define gpuAssert(ans) { gpuAssertCheck((ans), __FILE__, __LINE__); }
inline void gpuAssertCheck(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      const char* msg = cudaGetErrorString(code);
      fprintf(stderr,"GPUassert: %s %s %d\n", msg, file, line);
      if (abort) exit(code);
   }
}


void
check_error(const char* str, cudaError_t err_code)
{
    if (err_code != ::cudaSuccess)
        std::cerr << str << " -- " << cudaGetErrorString(err_code) << "\n";
}

/**
 * Exact solution
 * @param x
 * @return
 */
template<class T>
inline T
__device__ __host__
_u(T x)
{
    return sin(15 * x) * cos(24 * x) * x;
//    return x * x;
}

// <editor-fold defaultstate="collapsed" desc="equation parameters">

template<class T>
__device__ __host__
T
_a(T x)
{
    return sin(x);
//    return 1;
}

template<class T>
__device__ __host__
T
_b(T x)
{
    return x;
//    return 0;
}

template<class T>
__device__ __host__
T
_c(T x)
{
    return -x;
//    return 0;
}

template <class T>
inline T
__device__ __host__
_du(T x)
{
    return (-sin(9 * x)
            + sin(39 * x)
            - 9 * x * cos(9 * x)
            + 39 * x * cos(39 * x)) / 2.0;
//    return 2 * x;
}

template <class T>
__device__ __host__
inline T
_adu(T x)
{
    return _a(x) * _du(x);
}

template <class T>
__device__ __host__
inline T
_dadu(T x)
{
    return (cos(x)*(-sin(9 * x)
            + sin(39 * x)
            - 9 * x * cos(9 * x)
            + 39 * x * cos(39 * x))
            + 3 * sin(x)*(27 * x * sin(9 * x)
            - 507 * x * sin(39 * x)
            - 6 * cos(9 * x)
            + 26 * cos(39 * x))) / 2.0;
//    return 2;
}

template<class T>
__device__ __host__
T
_beta()
{
    return 0;
}

template<class T>
__device__ __host__
T
_gamma()
{
    return _adu<T > (1) + _beta<T > () * _u<T > (1);
}

template<class T>
__device__ __host__
inline T
_f(T x)
{
//    return cos(M_PI * x) * cos(M_PI * y);
    return -(_dadu(x)) + _b(x) * _du(x) + _c(x) * _u(x);
//    return -2;
}

// </editor-fold>

/**
 * Returns index that can be used for retrieving value of previously calculated
 * values of base functions (d_Nval) and their derivatives (d_dNval).
 * @param point_i - index of quadrature point [0, QUADRATURE_POINTS_CNT-1]
 * @param fun_i - index of a function [0, fun_cnt-1]
 * @param interval_i - number of interval
 * @param fun_cnt - total number of functions
 * @return index that can be used for d_Nvals and d_dNvals
 */
template<int qpc>
__device__
inline int
point_mem_idx(int point_i, int fun_i, int interval_i, int fun_cnt)
{
    return (interval_i * qpc + point_i) * fun_cnt + fun_i;
}

template<int degree, class T>
__device__
inline T
get_N(int point_i, int fun_i, int fun_part, int _n)
{
    return d_Nvals[degree & 1][point_mem_idx<degree + 1>(point_i, fun_i, fun_part, _n)];
}

template<int degree, class T>
__device__
inline T
get_dN(int point_i, int fun_i, int interval_i, int _n)
{
    return d_dNvals[degree & 1][point_mem_idx<degree + 1>(point_i, fun_i, interval_i, _n)];
}

template<int degree, class T>
__device__
inline T
get_N_err(int point_i, int fun_i, int fun_part, int _n)
{
    return d_Nvals_err[degree & 1][point_mem_idx<degree + 2>(point_i, fun_i, fun_part, _n)];
}

template<int degree, class T>
__device__
inline T
get_dN_err(int point_i, int fun_i, int interval_i, int _n)
{
    return d_dNvals_err[degree & 1][point_mem_idx<degree + 2>(point_i, fun_i, interval_i, _n)];
}

template<int degree, class T>
__device__
inline T
fun_L(T x, int point_idx, int fun, int interval, int _n)
{
//    return get_N<degree, T>(point_idx, fun, interval - fun, _n);
    return _f<T>(x) * get_N<degree, T>(point_idx, fun, interval - fun, _n);
}

template<int degree, class T>
__device__
inline T
fun_B(T x, int point_idx, int i, int j, int element_id, int _n)
{
    return _a<T > (x) * get_dN<degree, T > (point_idx, i, element_id - i + degree, _n)
            * get_dN<degree, T > (point_idx, j, element_id - j + degree, _n)
            + _b<T > (x) * get_dN<degree, T > (point_idx, i, element_id - i + degree, _n)
            * get_N<degree, T > (point_idx, j, element_id - j + degree, _n)
            + _c<T > (x) * get_N<degree, T > (point_idx, i, element_id - i + degree, _n)
            * get_N<degree, T > (point_idx, j, element_id - j + degree, _n);
}

template<int degree, class T>
__device__
T
eval_L(int fun, int interval, int _n)
{
    T sum(0);
    T aa, bb;
    T a = d_knot_vector[interval], b = d_knot_vector[interval + 1];
    aa = (b - a) / 2.0;
    bb = (b + a) / 2.0;

    for (int i = 0; i < QPC; ++i)
    {
        sum += Qw[i] * fun_L<degree>(aa * Qp[i] + bb, i, fun, interval, _n);
    }
    return aa*sum;
}

/**
 *
 * @param i
 * @param j
 * @param element_idx
 * @param _n - number of functions
 * @return
 */
template<int degree, class T>
__device__
inline T
eval_B(int i, int j, int knot_i, int _n)
{
    T sum(0);
    T aa, bb;
    T a = d_knot_vector[knot_i];
    T b = d_knot_vector[knot_i + 1];
    aa = (b - a) / 2.0;
    bb = (b + a) / 2.0;
    for (int idx = 0; idx < QPC; ++idx)
    {
        T x = aa * Qp[idx] + bb;
        sum += Qw[idx] * fun_B<degree > (x, idx, i, j, knot_i - degree, _n);
    }
    return aa*sum;
}

/**
 * Temporarily implemented only for degree 1, and _n being power of 2.
 * @param _c - front cell number (0,1,...)
 * @param _n - total number of fronts in one part
 * @param _p - number of part being evaluated (0,1,...)
 */
template<int degree, class T>
__global__
void
init_B(int _c, int _n, int _p)
{
    // find number of front being initiated
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    // is n greater than total number of fronts in one part?
    if (n >= _n) return;

    int element_id = n * (degree + 1) + _p;
    int y = _c / (degree + 1) + element_id;
    int x = _c % (degree + 1) + element_id;
    int idx = _n * (_p * (degree + 1)*(degree + 1) + _c) + n;

    if (y == 0)
    {
        if (x == 0)
        {
            d_B[0][0] = 1;
        }
        else
        {
            d_B[0][idx] = 0;
        }
    }
    else
    {
        d_B[0][idx] = eval_B<degree, T> (x, y, element_id + degree, _n * (degree + 1) + degree); // TODO uuu this is ugly
    }

    if (idx == _n * (degree + 1)*(degree + 1)*(degree + 1) - 1)
        d_B[0][idx] += _beta<T> ();
}

/**
 * Initializes RHS vector
 * @param Ndof number of basis functions
 */
template<int degree, class T>
__global__
void
init_RHS(int Ndof, int Nrhs)
{
    int rhs_num = blockDim.x * blockIdx.x + threadIdx.x;
    int n = blockDim.y * blockIdx.y + threadIdx.y;
    if (n >= Ndof || rhs_num >= Nrhs) return;

    T x = 0;

    for (int part = 0; part <= degree; ++part)
        x += eval_L<degree, T>(n, n + part, Ndof);

    if (n==0) x = 0;
    if (n == Ndof - 1) x += _gamma<T > ();
    d_RHS[(n*Nrhs) + rhs_num] = x;
}

/**
 *
 * @param _n - number of functions
 */
template<int degree, int qpc, class T>
__global__
void
init_basis_functions_and_derivatives(int _n, T **N, T **dN)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= _n) return;

    #pragma unroll
    for (int i = 0; i < (degree + 1) * qpc; ++i)
    {
        N[0][i * _n + n] = 0;
        N[1][i * _n + n] = 0;
        dN[0][i * _n + n] = 0;
        dN[1][i * _n + n] = 0;
    }

    #pragma unroll
    for (int i = 0; i < qpc; ++i)
    {
        N[0][i * _n + n] = 1;
    }
}

template<class T>
__device__
inline T
interval(int a, int b)
{
    return d_knot_vector[b] - d_knot_vector[a];
}

template<class T>
__device__
inline T
interval(int a)
{
    return interval<T > (a, a + 1);
}

template<int qpc, class T>
__global__
void
update_base(int _n, int idx, T **N, T **dN, T *Q_points)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x; // function number
    if (n >= _n) return;

    T h1 = interval<T > (n, n + idx);
    T h2 = interval<T > (n + 1, n + idx + 1);

    if (is_zero(h1))
    {
        for (int i = 0; i < qpc; ++i)
        {
            // function
            N[idx & 1][point_mem_idx<qpc>(i, n, 0, _n)] = 0;
            //derivative
            dN[idx & 1][point_mem_idx<qpc>(i, n, 0, _n)] = 0;
        }
    }
    else
    {
        for (int i = 0; i < qpc; ++i)
        {
            T x = (Q_points[i] / 2.0 + 0.5) * interval<T > (n);
            // function
            N[idx & 1][point_mem_idx<qpc>(i, n, 0, _n)] = x * N[(~idx)&1][point_mem_idx<qpc>(i, n, 0, _n)] / h1;
            //derivative
            dN[idx & 1][point_mem_idx<qpc>(i, n, 0, _n)] = (x * dN[(~idx)&1][point_mem_idx<qpc>(i, n, 0, _n)] + N[(~idx)&1][point_mem_idx<qpc>(i, n, 0, _n)]) / h1;
        }
    }

    for (int j = 1; j < idx; ++j)
    {
        for (int i = 0; i < qpc; ++i)
        {
            T sum_fun, sum_der;
            T x = (Q_points[i] / 2.0 + 0.5) * interval<T > (n + j) + d_knot_vector[n + j];
            if (is_zero(h1))
            {
                sum_fun = 0;
                sum_der = 0;
            }
            else
            {
                sum_fun = (x - d_knot_vector[n]) * N[(~idx)&1][point_mem_idx<qpc>(i, n, j, _n)] / h1;
                sum_der = ((x - d_knot_vector[n]) * dN[(~idx)&1][point_mem_idx<qpc>(i, n, j, _n)] + N[(~idx)&1][point_mem_idx<qpc>(i, n, j, _n)]) / h1;
            }

            if (is_zero(h2))
            {
                sum_fun += 0;
                sum_der += 0;
            }
            else
            {
                sum_fun += (d_knot_vector[n + idx + 1] - x) * N[(~idx)&1][point_mem_idx<qpc>(i, n + 1, j - 1, _n)] / h2;
                sum_der += ((d_knot_vector[n + idx + 1] - x) * dN[(~idx)&1][point_mem_idx<qpc>(i, n + 1, j - 1, _n)] - N[(~idx)&1][point_mem_idx<qpc>(i, n + 1, j - 1, _n)]) / h2;
            }
            N[idx & 1][point_mem_idx<qpc>(i, n, j, _n)] = sum_fun;
            dN[idx & 1][point_mem_idx<qpc>(i, n, j, _n)] = sum_der;
        }
    }

    if (is_zero(h2))
    {
        for (int i = 0; i < qpc; ++i)
        {
            N[idx & 1][point_mem_idx<qpc>(i, n, idx, _n)] = 0;
            dN[idx & 1][point_mem_idx<qpc>(i, n, idx, _n)] = 0;
        }
    }
    else
    {
        for (int i = 0; i < qpc; ++i)
        {
            T x = (Q_points[i] / 2.0 + 0.5) * interval<T > (n + idx) + d_knot_vector[n + idx];
            N[idx & 1][point_mem_idx<qpc>(i, n, idx, _n)] =
                    (d_knot_vector[n + idx + 1] - x) * N[(~idx)&1][point_mem_idx<qpc>(i, n + 1, idx - 1, _n)] / h2;
            dN[idx & 1][point_mem_idx<qpc>(i, n, idx, _n)] =
                    ((d_knot_vector[n + idx + 1] - x) * dN[(~idx)&1][point_mem_idx<qpc>(i, n + 1, idx - 1, _n)] - N[(~idx)&1][point_mem_idx<qpc>(i, n + 1, idx - 1, _n)]) / h2;
        }
    }

}

template<int degree, class T>
__device__
inline void
store_new_matrix(int next_merges_cnt, T BB[][2 * degree + 1][2 * degree + 1])
{
    int divisor = min(blockDim.x / 2, next_merges_cnt);

    int gm_idx = blockIdx.x * blockDim.x / 2;
    if (threadIdx.x < divisor)
    {
        for (int x = 0; x < 2 * degree; ++x)
            for (int y = 0; y < 2 * degree; ++y)
                d_B[1][(x * (2 * degree) + y) * next_merges_cnt + gm_idx + threadIdx.x] = BB[2 * threadIdx.x][x + 1][y + 1];
    }
    else
    {
        int tid = threadIdx.x - divisor;
        int group2_offset = next_merges_cnt * (2 * degree) * (2 * degree);
        for (int x = 0; x < 2 * degree; ++x)
            for (int y = 0; y < 2 * degree; ++y)
                d_B[1][group2_offset + (x * (2 * degree) + y) * next_merges_cnt + gm_idx + tid] = BB[2 * tid + 1][x + 1][y + 1];
    }
}

template<int degree, class T>
__device__
inline void
store_new_matrix2(int _n, T BB[][3 * degree][3 * degree], int step)
{
    int divisor = min(blockDim.x / 2, _n);

    int gm_idx = blockIdx.x * blockDim.x / 2;
    if (threadIdx.x < divisor)
    {
        for (int x = 0; x < 2 * degree; ++x)
        {
            for (int y = 0; y < 2 * degree; ++y)
            {
                d_B[(~step)&1][(x * (2 * degree) + y) * _n + gm_idx + threadIdx.x] = BB[2 * threadIdx.x][x + degree][y + degree];
            }
        }
    }
    else
    {
        int tid = threadIdx.x - divisor;
        int group2_offset = _n * (2 * degree) * (2 * degree);
        for (int x = 0; x < 2 * degree; ++x)
        {
            for (int y = 0; y < 2 * degree; ++y)
            {
                d_B[(~step)&1][group2_offset + (x * (2 * degree) + y) * _n + gm_idx + tid] = BB[2 * tid + 1][x + degree][y + degree];
            }
        }
    }
}

/**
 * @param merges_cnt - number of merges i.e. total number of elements / (degree + 1)
 */
template<int degree, class T, int TPB>
__global__
void
first_merge(int merges_cnt)
{
    __shared__ T BB[TPB][2 * degree + 1][2 * degree + 1];

    T assembled[2 * degree + 1];
    T X;

    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= merges_cnt) return;

    // initializing shared memory
    for (int x = 0; x < 2 * degree + 1; ++x)
        for (int y = 0; y < 2 * degree + 1; ++y)
            BB[threadIdx.x][x][y] = T(0);

    // load data to shared memory
    for (int i = 0; i < degree + 1; ++i)
        for (int x = 0; x < degree + 1; ++x)
            for (int y = 0; y < degree + 1; ++y)
                BB[threadIdx.x][x + i][y + i]
                    += d_B[0][((i * (degree + 1) + x) * (degree + 1) + y) * merges_cnt + n];

    // pivoting
    pivot_rows_cyclic<degree>(BB);
    pivot_columns_cyclic<degree>(BB);

    // calculate first row
    X = BB[threadIdx.x][0][0];
    for (int i = 1; i < 2 * degree + 1; ++i)
        assembled[i] = BB[threadIdx.x][0][i] / X;
    assembled[0] = X;

    // store first row which is already factorized
    for (int i = 0; i < 2 * degree + 1; ++i)
        d_assembled[i * merges_cnt + n] = assembled[i];

    // store first column needed for forward substitution
    for (int i = 1; i <= 2 * degree; ++i)
        d_not_assembled[(i-1) * merges_cnt + n] = BB[threadIdx.x][i][0];

    // elimination
    for (int i = 1; i < 2 * degree + 1; ++i)
    {
        T lead = BB[threadIdx.x][i][0];
        for (int j = 1; j <= 2 * degree; ++j)
            BB[threadIdx.x][i][j] -= assembled[j] * lead;
    }

    __syncthreads();
    store_new_matrix<degree, T > (merges_cnt / 2, BB);
}

/**
 * Merges fronts of size (2*degree)x(2*degree)
 * @param merges_cnt number of merging processes in this step
 * @param offset offset for factorized rows
 */
template<int degree, class T, int TPB>
__global__
void
merge(int merges_cnt, int assembled_offset, int not_assembled_offset, int step)
{
    __shared__ T BB[TPB][3 * degree][3 * degree];
    T assembled[3 * degree];

    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    const int row_len = 3 * degree;
    if (n >= merges_cnt) return;

    // initializing shared memory
    for (int x = 0; x < 3 * degree; ++x)
        for (int y = 0; y < 3 * degree; ++y)
            BB[threadIdx.x][x][y] = T(0);

    // load data to shared memory
    for (int x = 0; x < 2 * degree; ++x)
        for (int y = 0; y < 2 * degree; ++y)
        {
            BB[threadIdx.x][x][y] += d_B[step&1][(x * 2 * degree + y) * merges_cnt + n];
            BB[threadIdx.x][x + degree][y + degree]
                    += d_B[step&1][(2 * degree)* (2 * degree) * merges_cnt + (x * 2 * degree + y) * merges_cnt + n];
        }

    // pivoting
    for (int i = 0; i < degree; ++i)
        for (int x = 0; x < row_len; ++x)
        {
            T tmp = BB[threadIdx.x][i][x];
            BB[threadIdx.x][i][x] = BB[threadIdx.x][i + degree][x];
            BB[threadIdx.x][i + degree][x] = tmp;
        }
    for (int i = 0; i < degree; ++i)
        for (int x = 0; x < row_len; ++x)
        {
            T tmp = BB[threadIdx.x][x][i];
            BB[threadIdx.x][x][i] = BB[threadIdx.x][x][i + degree];
            BB[threadIdx.x][x][i + degree] = tmp;
        }

    // elimination (we eliminate |degree| rows)
    for (int i = 0; i < degree; ++i)
    {
        T X = BB[threadIdx.x][i][i];
        for (int j = i + 1; j < row_len; ++j)
            assembled[j] = BB[threadIdx.x][i][j] / X;
        for (int j = 0; j <= i; ++j)
            assembled[j] = BB[threadIdx.x][i][j];

        // store i-th row in global memory
        for (int j = 0; j < row_len; ++j)
            d_assembled[assembled_offset + (i * row_len + j) * merges_cnt + n] = assembled[j];

        // store i-th column in global memory
        for (int j = 0; j < 2 * degree; ++j)
            d_not_assembled[not_assembled_offset + (i * (2 * degree) + j) * merges_cnt + n] = BB[threadIdx.x][j + degree][i];

        // eliminate i-th row
        for (int x = i + 1; x < 3 * degree; ++x)
        {
            T lead = BB[threadIdx.x][x][i];
            for (int y = i + 1; y < 3 * degree; ++y)
            {
                BB[threadIdx.x][x][y] -= assembled[y] * lead;
            }
        }
    }

    // store new matrices in global memory
    __syncthreads();
    store_new_matrix2<degree, T > (merges_cnt / 2, BB, step);
}

template<int degree, class T>
__global__
__launch_bounds__(1)
void
last_merge(int offset, int step)
{
    __shared__ T BB[3 * degree][3 * degree];

    for (int x = 0; x < 3 * degree; ++x)
        for (int y = 0; y < 3 * degree; ++y)
            BB[x][y] = 0;

    // load data from global memory
    for (int x = 0; x < 2 * degree; ++x)
        for (int y = 0; y < 2 * degree; ++y)
            BB[x][y] = d_B[step&1][x * 2 * degree + y];

    for (int x = 0; x < 2 * degree; ++x)
        for (int y = 0; y < 2 * degree; ++y)
            BB[x + degree][y + degree] += d_B[step&1][(2 * degree) * (2 * degree) + x * (2 * degree) + y];

    for (int i = 0; i < 3 * degree; ++i)
    {
        T lead = BB[i][i];
        // BB[i][i] = 1; it is implicitly considered to be 1 later
        for (int x = i + 1; x < 3 * degree; ++x)
            BB[i][x] /= lead;

        for (int y = i + 1; y < 3 * degree; ++y)
        {
            lead = BB[y][i];
            // BB[y][i] = 0; it is implicitly considered to be 0 later
            for (int x = i + 1; x < 3 * degree; ++x)
                BB[y][x] -= BB[i][x] * lead;
        }
    }

    // row first order
    for (int y=0 ; y<3*degree ; ++y)
        for (int x=0 ; x<3*degree ; ++x)
        {
            d_assembled[offset + (y * (3 * degree) + x)] = BB[y][x];
        }
}

template<int degree, class T, int tpb>
__global__
void first_forward_substitution(int merges_cnt, int rhs_cnt)
{
    int merge_num = blockDim.y * blockIdx.y + threadIdx.y;
    int rhs_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (merge_num >= merges_cnt || rhs_num >= rhs_cnt) return;

    vertical_vec<T> RHS(d_RHS + rhs_num, rhs_cnt);

    __shared__ T s_div[tpb];
    T &div = s_div[threadIdx.y];
    if (threadIdx.x == 0)
        div = d_assembled[merge_num];
    __syncthreads();

    int row_num = (degree + 1) * (merge_num + 1) - 1;
    RHS[row_num] /= div;
}

template<int degree, class T, int tpb>
__global__
void first_forward_substitution_update_left(int merges_cnt, int rhs_cnt)
{
    int merge_num = blockDim.y * blockIdx.y + threadIdx.y;
    int rhs_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (merge_num >= merges_cnt || rhs_num >= rhs_cnt) return;

    __shared__ T s_B[tpb][degree];
    T *B = s_B[threadIdx.y];
    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (int i=0 ; i<degree ; ++i)
            B[i] = d_not_assembled[(i * merges_cnt) + merge_num];
    }
    __syncthreads();

    vertical_vec<T> RHS(d_RHS + rhs_num, rhs_cnt);
    int row_num = (degree + 1) * (merge_num + 1) - 1;

    T x = RHS[row_num];

    #pragma unroll
    for (int i=0 ; i<degree ; ++i)
        RHS[row_num - degree + i] -= B[i] * x;
}

template<int degree, class T, int tpb>
__global__
void first_forward_substitution_update_right(int merges_cnt, int rhs_cnt)
{
    int merge_num = blockDim.y * blockIdx.y + threadIdx.y;
    int rhs_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (merge_num >= merges_cnt || rhs_num >= rhs_cnt) return;

    __shared__ T s_B[tpb][degree];
    T *B = s_B[threadIdx.y];
    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (int i=degree ; i< 2 * degree ; ++i)
            B[i-degree] = d_not_assembled[(i * merges_cnt) + merge_num];
    }
    __syncthreads();

    vertical_vec<T> RHS(d_RHS + rhs_num, rhs_cnt);
    int row_num = (degree + 1) * (merge_num + 1) - 1;

    T x = RHS[row_num];

    #pragma unroll
    for (int i=0 ; i<degree ; ++i)
        RHS[row_num + i + 1] -= B[i] * x;
}

template<int degree, class T, int tpb>
__global__
void forward_substitution(int merges_cnt, int rhs_cnt, int offset, int step)
{
    int merge_num = blockDim.y * blockIdx.y + threadIdx.y;
    int rhs_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (merge_num >= merges_cnt || rhs_num >= rhs_cnt) return;

    vertical_vec<T> global_RHS(d_RHS + rhs_num, rhs_cnt);
    T RHS[degree];

    int stride = (1 << step) * (degree + 1);
    int row_len = 3 * degree;
    int g_idx = (stride >> 1) + merge_num * stride;

    __shared__ T s_B[tpb][degree][degree];
    T (*B)[degree] = s_B[threadIdx.y];
    if (threadIdx.x == 0)
        for (int i = 0; i < degree; ++i)
            for (int j = 0; j <= i; ++j)
                B[i][j] = d_assembled[offset + (i * row_len + j) * merges_cnt + merge_num];
    __syncthreads();

    for (int i = 0; i < degree; ++i)
        RHS[i] = global_RHS[g_idx + i];

    for (int i = 0; i < degree; ++i)
    {
        RHS[i] /= B[i][i];
        for (int j = i + 1; j < degree; ++j)
            RHS[j] -= RHS[i] * B[j][i];
    }

    for (int i = 0; i < degree; ++i)
        global_RHS[g_idx + i] = RHS[i];
}

template<int degree, class T, int tpb>
__global__
void forward_substitution_update_left(int merges_cnt, int rhs_cnt, int not_assembled_offset, int step)
{
    int merge_num = blockDim.y * blockIdx.y + threadIdx.y;
    int rhs_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (merge_num >= merges_cnt || rhs_num >= rhs_cnt) return;

    vertical_vec<T> global_RHS(d_RHS + rhs_num, rhs_cnt);
    T RHS[2 * degree];

    int stride = (1 << step) * (degree + 1);
    int middle_idx = (stride >> 1) + merge_num * stride;

    __shared__ T s_B[tpb][degree][degree];
    T (*B)[degree] = s_B[threadIdx.y];
    if (threadIdx.x == 0)
        for (int col = 0; col < degree; ++col)
            for (int j = 0; j < degree; ++j)
                B[j][col] = d_not_assembled[not_assembled_offset + (col * (2 * degree) + j) * merges_cnt + merge_num];
    __syncthreads();

    int small_stride = stride >> 1;
    for (int i = 0; i < degree; ++i)
        RHS[i] = global_RHS[middle_idx + i];
    for (int i = 0; i < degree; ++i)
        RHS[i + degree] = global_RHS[middle_idx - small_stride + i];

    for (int i = 0; i < degree; ++i)
        for (int j = 0; j < degree; ++j)
            RHS[j + degree] -= RHS[i] * B[j][i];

    for (int i = 0; i < degree; ++i)
        global_RHS[middle_idx - small_stride + i] = RHS[i + degree];
}

template<int degree, class T, int tpb>
__global__
void forward_substitution_update_right(int merges_cnt, int rhs_cnt, int not_assembled_offset, int step)
{
    int merge_num = blockDim.y * blockIdx.y + threadIdx.y;
    int rhs_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (merge_num >= merges_cnt || rhs_num >= rhs_cnt) return;

    vertical_vec<T> global_RHS(d_RHS + rhs_num, rhs_cnt);
    T RHS[2 * degree];

    int stride = (1 << step) * (degree + 1);
    int middle_idx = (stride >> 1) + merge_num * stride;

    __shared__ T s_B[tpb][degree][degree];
    T (*B)[degree] = s_B[threadIdx.y];
    if (threadIdx.x == 0)
        for (int col = 0; col < degree; ++col)
            for (int j = degree; j < 2 * degree; ++j)
                B[j - degree][col] = d_not_assembled[not_assembled_offset + (col * (2 * degree) + j) * merges_cnt + merge_num];
    __syncthreads();

    int small_stride = stride >> 1;
    for (int i = 0; i < degree; ++i)
        RHS[i] = global_RHS[middle_idx + i];
    for (int i = 0; i < degree; ++i)
        RHS[i + degree] = global_RHS[middle_idx + small_stride + i];

    for (int i = 0; i < degree; ++i)
        for (int j = 0; j < degree; ++j)
            RHS[j + degree] -= RHS[i] * B[j][i];

    for (int i = 0; i < degree; ++i)
        global_RHS[middle_idx + small_stride + i] = RHS[i + degree];
}

template<int degree, class T>
__global__
void
last_forward_first_backward_substitution(int ne, int rhs_cnt, int offset)
{
    int rhs_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (rhs_num >= rhs_cnt) return;

    vertical_vec<T> global_RHS(d_RHS + rhs_num, rhs_cnt);
    T RHS[3 * degree];

    __shared__ T BB[3 * degree][3 * degree];
    if (threadIdx.x == 0)
        // row first order
        for (int y = 0; y < 3 * degree; ++y)
            for (int x = 0; x < 3 * degree; ++x)
                BB[y][x] = d_assembled[offset + (y * (3 * degree) + x)];
    __syncthreads();

    for (int i = 0; i < degree; ++i)
        RHS[i] = global_RHS[i];
    for (int i = degree; i < 2 * degree; ++i)
        RHS[i] = global_RHS[ne / 2 - degree + i];
    for (int i = 2 * degree; i < 3 * degree; ++i)
        RHS[i] = global_RHS[ne - 2 * degree + i];

    for (int i = 0; i < 3 * degree; ++i)
    {
        RHS[i] /= BB[i][i];
        for (int j = i + 1; j < 3 * degree; ++j)
            RHS[j] -= RHS[i] * BB[j][i];
    }

    // we skip last row as it is already solved
    for (int i = 3 * degree - 2; i >= 0 ; --i)
        for (int j = i + 1 ; j < 3 * degree ; ++j)
            RHS[i] -= BB[i][j] * RHS[j];

    for (int i = 0; i < degree; ++i)
        global_RHS[i] = RHS[i];
    for (int i = degree; i < 2 * degree; ++i)
        global_RHS[ne / 2 - degree + i] = RHS[i];
    for (int i = 2 * degree; i < 3 * degree; ++i)
        global_RHS[ne - 2 * degree + i] = RHS[i];
}

template<int degree, class T, int tpb>
__global__
void
backward_substitution(int merges_cnt, int rhs_cnt, int offset, int step)
{
    int merge_num = blockDim.y * blockIdx.y + threadIdx.y;
    int rhs_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (merge_num >= merges_cnt || rhs_num >= rhs_cnt) return;

    vertical_vec<T> global_RHS(d_RHS + rhs_num, rhs_cnt);
    T RHS[3 * degree];

    int stride = (1 << step) * (degree + 1);
    int row_len = 3 * degree;
    int g_idx = (stride >> 1) + merge_num * stride;

    __shared__ T s_B[tpb][degree][3 * degree];
    T (*B)[3 * degree] = s_B[threadIdx.y];
    if (threadIdx.x == 0)
        for (int i = 0; i < degree; ++i)
            for (int j = i+1; j < 3 * degree; ++j)
                B[i][j] = d_assembled[offset + (i * row_len + j) * merges_cnt + merge_num];
    __syncthreads();

    stride >>= 1;
    for (int i = 0; i < degree; ++i)
        RHS[i] = global_RHS[g_idx + i];
    for (int i = 0; i < degree; ++i)
        RHS[i + degree] = global_RHS[g_idx - stride + i];
    for (int i = 0; i < degree; ++i)
        RHS[i + 2 * degree] = global_RHS[g_idx + stride + i];

    for (int i = degree - 1; i >= 0; --i)
        for (int j = i + 1; j < 3 * degree; ++j)
            RHS[i] -= RHS[j] * B[i][j];

    for (int i = 0; i < degree; ++i)
        global_RHS[g_idx + i] = RHS[i];
}

/**
 * @param _n
 */
template<int degree, class T, int tpb>
__global__
void
last_backward_substitution(int merges_cnt, int rhs_cnt)
{
    int merge_num = blockDim.y * blockIdx.y + threadIdx.y;
    int rhs_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (merge_num >= merges_cnt || rhs_num >= rhs_cnt) return;

    __shared__ T s_B[tpb][2 * degree];
    T *B = s_B[threadIdx.y];
    if (threadIdx.x == 0)
        for (int i = 1; i <= 2 * degree; ++i)
            B[i - 1] = d_assembled[i * merges_cnt + merge_num];
    __syncthreads();

    vertical_vec<T> global_RHS(d_RHS + rhs_num, rhs_cnt);
    unsigned row_num = (degree + 1) * (merge_num + 1) - 1;
    T RHS = global_RHS[row_num];

    for (int i = -degree; i < 0; ++i)
        RHS -= B[i + degree] * global_RHS[row_num + i];

    for (int i = 1; i <= degree; ++i)
        RHS -= B[i + degree - 1] * global_RHS[row_num + i];

    global_RHS[row_num] = RHS;
}

template<int degree, class T>
__global__
void
evaluate(int _n, T* x, int length)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= length) return;

    T sum = 0;
    T XX = x[n];
    for (int i = 0; i < _n + degree; ++i)
    {
        sum += d_RHS[i] * N<degree, T > (XX, i);
    }
    x[n] = sum;
}

template<int degree, class T>
__device__
T
evaluate(T x, int elements_cnt)
{
    T sum = 0;
    for (int i = 0; i < elements_cnt + degree; ++i)
    {
        sum += d_RHS[i] * N<degree > (x, i);
    }
    return sum;
}

template<int degree, class T, int qpc>
__global__
void
calculate_norm(int elm_cnt, T *sum)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= elm_cnt) return;

    T tmp = 0;

    for (int i = 0; i < qpc; ++i)
    {
        tmp += Qw_e[i]
                * pow(_u(d_abssicas[i * elm_cnt + n]) - d_fun_vals[i * elm_cnt + n], 2);
    }

    for (int i = 0; i < qpc; ++i)
    {
        tmp += Qw_e[i]
                * pow((_du(d_abssicas[i * elm_cnt + n]) - d_der_vals[i * elm_cnt + n]), 2);
    }

    sum[n] = interval<T>(n + degree) * tmp / 2.0;
}

template<int degree, class T>
__global__
void
calculate_function_in_quadrature_points(int elements_cnt, int Nrhs)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    const int rhs_num = 0;
    if (n >= elements_cnt) return;

    for (int i = 0; i < QPC_e; ++i)
        for (int j = 0; j <= degree; ++j)
            d_fun_vals[i * elements_cnt + n] +=
                d_RHS[(n + j)*Nrhs + rhs_num] * get_N_err<degree, T > (i, n + j, degree - j, basis_funs_cnt<degree > (elements_cnt));
}

template<int degree, class T>
__global__
void
calculate_derivative_in_quadrature_points(int elements_cnt, int Nrhs)
{
    const int rhs_num = 0;
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= elements_cnt) return;

    for (int i = 0; i < QPC_e; ++i)
        for (int j = 0; j <= degree; ++j)
            d_der_vals[i * elements_cnt + n] +=
                d_RHS[(n + j)*Nrhs + rhs_num] * get_dN_err<degree, T > (i, n + j, degree - j, basis_funs_cnt<degree > (elements_cnt));
}

template<int degree, class T>
__global__
void
calculate_abscissas(int elements_cnt)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= elements_cnt) return;

    T a = d_knot_vector[n + degree], b = d_knot_vector[n + degree + 1];
    T aa = (b - a) / 2.0;
    T bb = (b + a) / 2.0;
    for (int i = 0; i < QPC_e; ++i)
    {
        d_abssicas[i * elements_cnt + n] = aa * Qp_e[i] + bb;
    }
}

template<class T>
T
t_gen(int i, int N, int degree)
{
    if (i <= degree) return T(0);
    if (i < N + degree) return T(i - degree) / N;
    return T(1);
}

template<class T>
struct vector_scattered {
    T *p;
    int skip;
    vector_scattered (T *p, int skip) : p(p), skip(skip) { }

    T &
    operator[](int idx)
    {
        return *(p + idx * skip);
    }
};

template<class T, int degree>
void
debug_device(int n);


template<class T, int degree>
void
print_local(int n);


template<int degree, int qpc, class T>
void
calculate_basis(int n, T **N, T **dN, T *Q)
{
    int tpb = 256;
    int block_count = ((n + degree) / tpb) + ((n + degree) % tpb ? 1 : 0);

    init_basis_functions_and_derivatives<degree, qpc><<<block_count, tpb >>>(basis_funs_cnt<degree>(n), N, dN);

    check_error("init_base_functions_and_derivatives", cudaGetLastError());
    for (int i = 1; i <= degree; ++i)
    {
        update_base<qpc><<<block_count, tpb>>>(n + degree, i, N, dN, Q);
        check_error("update_base", cudaGetLastError());
    }
}

template<class T, int degree>
void
calculate_basis_solver(int n)
{
    void *tmp;
    T **N, **dN, *Q;
    gpuAssert(cudaGetSymbolAddress(&tmp, d_Nvals));
    N = reinterpret_cast<T**>(tmp);
    gpuAssert(cudaGetSymbolAddress(&tmp, d_dNvals));
    dN = reinterpret_cast<T**>(tmp);
    gpuAssert(cudaGetSymbolAddress(&tmp, Qp));
    Q = reinterpret_cast<T*>(tmp);

    calculate_basis<degree, degree + 1>(n, N, dN, Q);
}

template<class T, int degree>
void
calculate_basis_error(int n)
{
    void *tmp;
    T **N, **dN, *Q;
    gpuAssert(cudaGetSymbolAddress(&tmp, d_Nvals_err));
    N = reinterpret_cast<T**>(tmp);
    gpuAssert(cudaGetSymbolAddress(&tmp, d_dNvals_err));
    dN = reinterpret_cast<T**>(tmp);
    gpuAssert(cudaGetSymbolAddress(&tmp, Qp_e));
    Q = reinterpret_cast<T*>(tmp);

    // error_QPC = degree + 2
    calculate_basis<degree, degree + 2>(n, N, dN, Q);
}

/**
 * Allocates necessary memory on device
 * @param Ne - number of elements in [0, 1] interval
 * @param Nrhs - number of right hand sides
 */
template<class T, int degree>
void
prepare_device(int Ne, int Nrhs)
{
    T *tmp, *t;
    T * dev_ptrs[2];

    const int solver_QPC = degree + 1;
    const int error_QPC = degree + 2;
    // initialize quadratures for solver
    {
        T *p = get_gauss_legendre_points<T, solver_QPC>();
        T *w = get_gauss_legendre_weights<T, solver_QPC>();
        gpuAssert(cudaMemcpyToSymbol(QPC, &solver_QPC, sizeof (solver_QPC)));
        gpuAssert(cudaMemcpyToSymbol(Qp, p, sizeof (T) * solver_QPC));
        gpuAssert(cudaMemcpyToSymbol(Qw, w, sizeof (T) * solver_QPC));
    }
    // initialize quadratures for error calculation
    {
        T *p = get_gauss_legendre_points<T, error_QPC>();
        T *w = get_gauss_legendre_weights<T, error_QPC>();
        gpuAssert(cudaMemcpyToSymbol(QPC_e, &error_QPC, sizeof (error_QPC)));
        gpuAssert(cudaMemcpyToSymbol(Qp_e, p, sizeof (T) * error_QPC));
        gpuAssert(cudaMemcpyToSymbol(Qw_e, w, sizeof (T) * error_QPC));
    }

    t = new T[Ne + 2 * degree + 1];
    for (int i = 0; i <= Ne + 2 * degree; ++i)
        t[i] = t_gen<T > (i, Ne, degree);

    int size;
    int mem_size=0, total_mem_size=0;

    // Allocate knot vector
    gpuAssert(cudaMalloc(&tmp, sizeof (T) * knots_cnt<degree > (Ne)));
    gpuAssert(cudaMemcpy(tmp, t, sizeof (T) * knots_cnt<degree > (Ne),
                             cudaMemcpyHostToDevice));
    gpuAssert(cudaMemcpyToSymbol(d_knot_vector, &tmp, sizeof (tmp)));
    delete[] t;

    // Allocate fronts (B part)
    size = std::max(Ne * (degree + 1) * (degree + 1),
                    (Ne / (degree + 1)) * (2 * degree) * (2 * degree));
    gpuAssert(cudaMalloc(&dev_ptrs[0], sizeof (T) * size));
    gpuAssert(cudaMalloc(&dev_ptrs[1], sizeof (T) * size));
    gpuAssert(cudaMemcpyToSymbol(d_B, &dev_ptrs, sizeof(dev_ptrs)));

    // allocate RHS
    gpuAssert(cudaMalloc(&tmp, Nrhs * sizeof (T) * basis_funs_cnt<degree>(Ne)));
    gpuAssert(cudaMemcpyToSymbol(d_RHS, &tmp, sizeof (tmp)));

    // allocate d_assembled
    gpuAssert(cudaMalloc(&tmp, sizeof (T) * basis_funs_cnt<degree>(Ne) * 3 * degree));
    gpuAssert(cudaMemcpyToSymbol(d_assembled, &tmp, sizeof (tmp)));

    // allocate d_not_assembled
    {
        int M = Ne / (degree + 1);
        gpuAssert(cudaMalloc(&tmp, sizeof (T) * 2 * M * (degree * degree + degree)));
        gpuAssert(cudaMemcpyToSymbol(d_not_assembled, &tmp, sizeof (tmp)));
    }
    // Allocate matrices for accumulative base function evaluation
    // functions
    gpuAssert(cudaMalloc(&dev_ptrs[0], sizeof (T) * basis_funs_cnt<degree > (Ne) * (degree + 1) * solver_QPC));
    gpuAssert(cudaMalloc(&dev_ptrs[1], sizeof (T) * basis_funs_cnt<degree > (Ne) * (degree + 1) * solver_QPC));
    gpuAssert(cudaMemcpyToSymbol(d_Nvals, dev_ptrs, sizeof (dev_ptrs)));

    // derivatives
    gpuAssert(cudaMalloc(&dev_ptrs[0], sizeof (T) * basis_funs_cnt<degree > (Ne) * (degree + 1) * solver_QPC));
    gpuAssert(cudaMalloc(&dev_ptrs[1], sizeof (T) * basis_funs_cnt<degree > (Ne) * (degree + 1) * solver_QPC));
    gpuAssert(cudaMemcpyToSymbol(d_dNvals, dev_ptrs, sizeof (dev_ptrs)));

    // FOR ERROR CALCULATION
    // functions
    mem_size = sizeof (T) * basis_funs_cnt<degree > (Ne) * (degree + 1) * error_QPC;
    total_mem_size += mem_size;
    gpuAssert(cudaMalloc(&dev_ptrs[0], mem_size));
    total_mem_size += mem_size;
    gpuAssert(cudaMalloc(&dev_ptrs[1], mem_size));
    gpuAssert(cudaMemcpyToSymbol(d_Nvals_err, dev_ptrs, sizeof (dev_ptrs)));

    // derivatives
    mem_size = sizeof (T) * basis_funs_cnt<degree > (Ne) * (degree + 1) * error_QPC;
    total_mem_size += mem_size;
    gpuAssert(cudaMalloc(&dev_ptrs[0], mem_size));
    total_mem_size += mem_size;
    gpuAssert(cudaMalloc(&dev_ptrs[1], mem_size));
    gpuAssert(cudaMemcpyToSymbol(d_dNvals_err, dev_ptrs, sizeof (dev_ptrs)));


    // Allocate space for result
    gpuAssert(cudaMalloc(&tmp, sizeof (T) * error_QPC * Ne));
    gpuAssert(cudaMemset(tmp, 0, sizeof (T) * error_QPC * Ne));
    gpuAssert(cudaMemcpyToSymbol(d_fun_vals, &tmp, sizeof (tmp)));

    gpuAssert(cudaMalloc(&tmp, sizeof (T) * error_QPC * Ne));
    gpuAssert(cudaMemset(tmp, 0, sizeof (T) * error_QPC * Ne));
    gpuAssert(cudaMemcpyToSymbol(d_der_vals, &tmp, sizeof (tmp)));

    gpuAssert(cudaMalloc(&tmp, sizeof (T) * error_QPC * Ne));
    gpuAssert(cudaMemcpyToSymbol(d_abssicas, &tmp, sizeof (tmp)));

    cudaThreadSynchronize();
}

/**
 *
 * @param n number of elements
 */
template<int degree, class T>
void
prepare_result(int n, int Nrhs)
{
    int tpb = 256;
    int block_cnt = n / tpb + (n % tpb ? 1 : 0);

    calculate_abscissas<degree, T> << <block_cnt, tpb >> >(n);
    check_error("calculate_abscissas", cudaGetLastError());
    calculate_function_in_quadrature_points<degree, T> << <block_cnt, tpb >> >(n, Nrhs);
    check_error("calculate_function_in_quadrature_points", cudaGetLastError());
    calculate_derivative_in_quadrature_points<degree, T> << <block_cnt, tpb >> >(n, Nrhs);
    check_error("calculate_derivative_in_quadrature_points", cudaGetLastError());
}

void
cleanup_device()
{
    void *tmp;
    void *dev_ptrs[2];

    gpuAssert(cudaMemcpyFromSymbol(&dev_ptrs, d_B, sizeof (dev_ptrs)));
    gpuAssert(cudaFree(dev_ptrs[0]));
    gpuAssert(cudaFree(dev_ptrs[1]));

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_knot_vector, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_RHS, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_assembled, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_not_assembled, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));

    // Free matrices for accumulative base function evaluation
    // functions
    gpuAssert(cudaMemcpyFromSymbol(dev_ptrs, d_Nvals, sizeof (dev_ptrs)));
    gpuAssert(cudaFree(dev_ptrs[0]));
    gpuAssert(cudaFree(dev_ptrs[1]));

    // derivatives
    gpuAssert(cudaMemcpyFromSymbol(dev_ptrs, d_dNvals, sizeof (dev_ptrs)));
    gpuAssert(cudaFree(dev_ptrs[0]));
    gpuAssert(cudaFree(dev_ptrs[1]));

    // Free result memory
    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_fun_vals, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));
    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_der_vals, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));
    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_abssicas, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));

    // FOR ERROR CALCULATION
    // functions
    gpuAssert(cudaMemcpyFromSymbol(dev_ptrs, d_Nvals_err, sizeof (dev_ptrs)));
    gpuAssert(cudaFree(dev_ptrs[0]));
    gpuAssert(cudaFree(dev_ptrs[1]));

    // derivatives
    gpuAssert(cudaMemcpyFromSymbol(dev_ptrs, d_dNvals_err, sizeof (dev_ptrs)));
    gpuAssert(cudaFree(dev_ptrs[0]));
    gpuAssert(cudaFree(dev_ptrs[1]));
}

template<class T, int degree>
void
init_fronts(int Ne, int Nrhs)
{
    calculate_basis_solver<T, degree>(Ne);
    check_error("calculate_basis_solver", cudaGetLastError());
    calculate_basis_error<T, degree>(Ne);
    check_error("calculate_basis_error", cudaGetLastError());

    int N = Ne / (degree + 1);
    int threads_per_block = 32;
    int block_count = div_ceil(N, threads_per_block);
    for (int part = 0; part <= degree; ++part)
    {
        for (int i = 0; i < (degree + 1)*(degree + 1); ++i)
        {
            init_B<degree, T><<<block_count, threads_per_block>>>(i, N, part);
            check_error("B", cudaGetLastError());
        }
    }

    const int Ndof = basis_funs_cnt<degree>(Ne);
    int threads_per_block_per_rhs = 8;
    int rhs_per_block = 16;
    int blocks_per_rhs = div_ceil(Ndof, threads_per_block_per_rhs);
    int rhs_blocks = div_ceil(Nrhs, rhs_per_block);

    dim3 b_grid(rhs_blocks, blocks_per_rhs);
    dim3 t_grid(rhs_per_block, threads_per_block_per_rhs);

    init_RHS<degree, T><<<b_grid, t_grid>>>(Ndof, Nrhs);
    check_error("init_RHS", cudaGetLastError());

    cudaThreadSynchronize();
}

/**
 * Calculates offset in global d_assembled for rows being factorized in
 * provided step
 * @param n - number of elements
 * @param step - current step
 * @return offset
 */
template<int degree>
int
assembled_offset_for_step(int Ne, int step)
{
    if (step <= 0) return 0;

    const int rows_per_merge = degree;
    int merges_cnt = Ne / (degree + 1);
    int offset = 0;
    offset += merges_cnt * (2 * degree + 1);
    while (--step)
    {
        merges_cnt >>= 1; // div 2
        offset += (3 * degree) * merges_cnt * rows_per_merge;
    }
    return offset;
}

/**
 * Calculates offset in global d_not_assembled for columns for not assembled
 * rows
 * @param n - number of elements
 * @param step - current step
 * @return offset
 */
template<int degree>
int
not_assembled_offset_for_step(int Ne, int step)
{
    if (step <= 0) return 0;

    int merges_cnt = Ne / (degree + 1);
    int offset = 0;
    offset += merges_cnt * (2 * degree);
    while (--step)
    {
        merges_cnt >>= 1; // div 2
        offset += (2 * degree * degree) * merges_cnt;
    }
    return offset;
}

template<class T, int degree>
void
launch_matrix_factorization(int ne)
{
    const int tpb = degree >= 4 ? THREADS_PER_BLOCK / 2 : THREADS_PER_BLOCK;

    int block_grid;
    int merges_cnt;

    merges_cnt = merges_cnt_for_step<degree>(ne, 0);
    block_grid = prepare_block_grid(merges_cnt, tpb);

    first_merge<degree, T, tpb> <<<block_grid, tpb>>>(merges_cnt);
    check_error("first merge", cudaGetLastError());
//    print_local<T, degree>(ne);

    int max_step = steps_cnt<degree>(ne);
    for (int step = 1; step < max_step ; ++step)
    {
        merges_cnt = merges_cnt_for_step<degree>(ne, step);
        block_grid = prepare_block_grid(merges_cnt, tpb);
        merge<degree, T, tpb><<<block_grid, tpb>>>(merges_cnt,
                                                   assembled_offset_for_step<degree>(ne, step),
                                                   not_assembled_offset_for_step<degree>(ne, step),
                                                   step);
        check_error("merge", cudaGetLastError());
    }

    last_merge<degree, T><<<1, 1>>>(assembled_offset_for_step<degree>(ne, max_step), max_step);
    check_error("last merge", cudaGetLastError());
}

template<class T, int degree>
void
launch_forward_substitution(int ne, int Nrhs)
{
    const int tpb = degree >= 4 ? 8 : 16;
    const int RHSes_per_block = 16;

    int merges_cnt;

    merges_cnt = merges_cnt_for_step<degree>(ne, 0);

    dim3 t_grid(RHSes_per_block, tpb);
    dim3 b_grid = calculate_blocks(dim3(Nrhs, merges_cnt), t_grid);

//    debug_device<T, degree>(ne);
    first_forward_substitution<degree, T, RHSes_per_block><<<b_grid, t_grid>>>(merges_cnt, Nrhs);
    check_error("first_forward_substitution", cudaGetLastError());
//    debug_device<T, degree>(ne);
    first_forward_substitution_update_left<degree, T, RHSes_per_block><<<b_grid, t_grid>>>(merges_cnt, Nrhs);
    check_error("first_forward_substitution_update_left", cudaGetLastError());
//    debug_device<T, degree>(ne);
    first_forward_substitution_update_right<degree, T, RHSes_per_block><<<b_grid, t_grid>>>(merges_cnt, Nrhs);
    check_error("first_forward_substitution_update_right", cudaGetLastError());
//    debug_device<T, degree>(ne);


    int max_step = steps_cnt<degree>(ne);
    for (int step = 1; step < max_step ; ++step)
    {
        merges_cnt = merges_cnt_for_step<degree>(ne, step);
        b_grid = calculate_blocks(dim3(Nrhs, merges_cnt), t_grid);

        forward_substitution<degree, T, RHSes_per_block><<<b_grid, t_grid>>>(merges_cnt, Nrhs, assembled_offset_for_step<degree>(ne, step), step);
        check_error("forward_substitution", cudaGetLastError());
//        debug_device<T, degree>(ne);
        forward_substitution_update_left<degree, T, RHSes_per_block><<<b_grid, t_grid>>>(merges_cnt, Nrhs, not_assembled_offset_for_step<degree>(ne, step), step);
        check_error("forward_substitution_update_left", cudaGetLastError());
//        debug_device<T, degree>(ne);
        forward_substitution_update_right<degree, T, RHSes_per_block><<<b_grid, t_grid>>>(merges_cnt, Nrhs, not_assembled_offset_for_step<degree>(ne, step), step);
        check_error("forward_substitution_update_right", cudaGetLastError());
//        debug_device<T, degree>(ne);
    }

    t_grid = dim3(RHSes_per_block, 1);
    b_grid = calculate_blocks(dim3(Nrhs, 1), t_grid);
    last_forward_first_backward_substitution<degree, T><<<b_grid, t_grid>>>(ne, Nrhs, assembled_offset_for_step<degree>(ne, max_step));
    check_error("last_forward_first_backward_substitution", cudaGetLastError());
}

template<class T, int degree>
void
launch_backward_substitution(int ne, int Nrhs)
{
    const int tpb = degree >= 4 ? 8 : 16;
    const int RHSes_per_block = 16;

    int merges_cnt;

    dim3 t_grid(RHSes_per_block, tpb);
    dim3 b_grid;

    int max_step = steps_cnt<degree>(ne);
    for (int step = max_step - 1; step > 0 ; --step)
    {
        merges_cnt = merges_cnt_for_step<degree>(ne, step);
        b_grid = calculate_blocks(dim3(Nrhs, merges_cnt), t_grid);
        backward_substitution<degree, T, RHSes_per_block><<<b_grid, t_grid>>>(merges_cnt, Nrhs, assembled_offset_for_step<degree>(ne, step), step);
        check_error("backward_substitution", cudaGetLastError());
    }

    merges_cnt = merges_cnt_for_step<degree>(ne, 0);
    b_grid = calculate_blocks(dim3(Nrhs, merges_cnt), t_grid);

    last_backward_substitution<degree, T, RHSes_per_block><<<b_grid, t_grid>>>(merges_cnt, Nrhs);
    check_error("last_backward_substitution", cudaGetLastError());
}

template<class T, int degree>
void
factorize_matrix(int Ne)
{
//    debug_device<T, degree>(ne);
    launch_matrix_factorization<T, degree>(Ne);
    cudaThreadSynchronize();
}

template<class T, int degree>
void
solve_equation(int Ne, int Nrhs)
{
    launch_forward_substitution<T, degree>(Ne, Nrhs);

//    debug_device<T, degree>(ne);

    launch_backward_substitution<T, degree>(Ne, Nrhs);
//    debug_device<T, degree>(ne);

    prepare_result<degree, T>(Ne, Nrhs);

    cudaThreadSynchronize();
}

template<class T, int degree>
void
print_result(int n, std::ostream &ostr)
{
    T *x, *y, *tmp;
    const int QPC = degree + 2;
    x = new T[n * QPC];
    y = new T[n * QPC];

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_abssicas, sizeof (tmp)));
    gpuAssert(cudaMemcpy(x, tmp, sizeof (T) * QPC * n, cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_fun_vals, sizeof (tmp)));
    gpuAssert(cudaMemcpy(y, tmp, sizeof (T) * QPC * n, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n * QPC; ++i)
        ostr << x[i] << ' ' << y[i] << '\n';

    delete []x;
    delete []y;

//    debug_device<T, degree>(n);
}

template<class T, int degree>
T
calculate_error(int N)
{
    T *tmp;
    T *result = new T[N];

    gpuAssert(cudaMalloc(&tmp, sizeof (T) * N));

    int tpb = 256;
    int block_cnt = N / tpb + (N % tpb ? 1 : 0);

    calculate_norm<degree, T, degree + 2><<<block_cnt, tpb>>>(N, tmp);
    check_error("calculate_norm", cudaGetLastError());

    gpuAssert(cudaMemcpy(result, tmp, sizeof (T) * N, cudaMemcpyDeviceToHost));

    T sum = 0;
    for (int i = 0; i < N; ++i)
    {
        sum += result[i];
    }

    delete []result;

    gpuAssert(cudaFree(tmp));
    return sqrt(sum);
}

template<class T, int degree>
void
print_local(int n)
{
    T *bb;
    void *tmp[2];
    int merges_cnt = merges_cnt_for_step<degree > (n, 1);
    int size = merges_cnt * (2 * degree + 1) * (2 * degree + 1);
    bb = new T[size];
    gpuAssert(cudaMemcpyFromSymbol(tmp, d_B, sizeof (tmp)));
    gpuAssert(cudaMemcpy(bb, tmp[0], sizeof (T) * size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < merges_cnt; ++i)
    {
        for (int x = 0; x < 2 * degree + 1; ++x)
        {
            for (int y = 0; y < 2 * degree + 1; ++y)
            {
                std::cout << bb[(x * 2 * degree + y) * merges_cnt + i] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    delete[] bb;
//    bb = new T[basis_funs_cnt<degree>(n)];
//    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_RHS, sizeof (tmp)));
//    gpuAssert(cudaMemcpy(bb, tmp, sizeof (T) * basis_funs_cnt<degree>(n), cudaMemcpyDeviceToHost));
//
//
//    for (int i = 0 ; i < basis_funs_cnt<degree>(n) ; ++i)
//        std::cout << bb[i] << "\n";
//
//    std::cout << "------------------------\n------------------------\n";
//
//
//    delete[] bb;
}

template<class T, int degree>
void
debug_device(int n)
{
//    T *bb, *tmp;
//    int QPC = degree + 2;
//    bb = new T[(n + degree) * (degree + 1) * QPC];
//    T * dev_ptrs[2];
//    gpuAssert(cudaMemcpyFromSymbol(dev_ptrs, d_Nvals_err, sizeof (dev_ptrs)));
//    gpuAssert(cudaMemcpy(bb, dev_ptrs[degree & 1],
//                             sizeof (T) * basis_funs_cnt<degree > (n) * (degree + 1) * QPC,
//                             cudaMemcpyDeviceToHost));
//
//    std::cout << "------------------------\n------------------------\n";
//    for (int i = 0; i < basis_funs_cnt<degree > (n) * (degree + 1) * QPC; ++i)
//    {
//        std::cout << bb[i] << '\n';
//    }
//    delete [] bb;
//
//    gpuAssert(cudaMemcpy(bb, dev_ptrs[1], sizeof (T) * (n + degree) * (degree + 1) * QPC, cudaMemcpyDeviceToHost));
//
//    std::cout << "------------------------\n------------------------\n";
//    for (int i = 0; i < (n + degree) * (degree + 1) * QPC; ++i)
//    {
//        std::cout << bb[i] << '\n';
//    }
//
//    delete [] bb;
//    int Ndof = basis_funs_cnt<degree>(n);
//    bb = new T[RHSC * Ndof];
//    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_RHS, sizeof (tmp)));
//    gpuAssert(cudaMemcpy(bb, tmp, sizeof (T) * Ndof * RHSC, cudaMemcpyDeviceToHost));
//
//    for (int i=0 ; i<Ndof ; ++i)
//    {
//        for (int r = 0 ; r < RHSC ; ++r)
//        {
//            std::cerr << bb[i*RHSC + r] << "\t\t";
//        }
//        std::cerr << "\n";
//    }
//
//    delete[] bb;
}

typedef void (*device_fun_i)(int);
typedef void (*device_fun_ii)(int, int);
// <editor-fold defaultstate="collapsed" desc="interface functions (float)">
//
//template<>
//void
//CUDA_prepare_device<float>(int degree, int n)
//{
//    static device_fun prepares[] = {
//        prepare_device<float, 1 >,
//        prepare_device<float, 2 >,
//        prepare_device<float, 3 >,
//        prepare_device<float, 4 >,
//        prepare_device<float, 5 >
//    };
//
//    prepares[degree - 1](n);
//}
//
//template<>
//void
//CUDA_init_fronts<float>(int degree, int n)
//{
//    static device_fun initializers[] = {
//        init_fronts<float, 1 >,
//        init_fronts<float, 2 >,
//        init_fronts<float, 3 >,
//        init_fronts<float, 4 >,
//        init_fronts<float, 5 >
//    };
//
//    initializers[degree - 1](n);
//}
//
//template<>
//void
//CUDA_solve<float>(int degree, int n)
//{
//    static device_fun solvers[] = {
//        solve_equation<float, 1 >,
//        solve_equation<float, 2 >,
//        solve_equation<float, 3 >,
//        solve_equation<float, 4 >,
//        solve_equation<float, 5 >
//    };
//
//    solvers[degree - 1](n);
//}
//
//template<>
//float
//CUDA_error<float>(int degree, int n)
//{
//    typedef float (*error_fun)(int);
//    static error_fun calculators[] = {
//        calculate_error<float, 1 >,
//        calculate_error<float, 2 >,
//        calculate_error<float, 3 >,
//        calculate_error<float, 4 >,
//        calculate_error<float, 5 >
//    };
//
//    return calculators[degree - 1](n);
//}
//
//template<>
//void
//CUDA_print_result<float>(int degree, int n, std::ostream &ostr)
//{
//    typedef void (*print_fun)(int, std::ostream &);
//    static print_fun printers[] = {
//        print_result<float, 1 >,
//        print_result<float, 2 >,
//        print_result<float, 3 >,
//        print_result<float, 4 >,
//        print_result<float, 5 >
//    };
//
//    printers[degree - 1](n, ostr);
//}
//
//template<>
//void
//CUDA_debug<float>(int degree, int n)
//{
//    typedef void (*print_fun)(int);
//    static print_fun debuggers[] = {
//        debug_device<float, 1 >,
//        debug_device<float, 2 >,
//        debug_device<float, 3 >,
//        debug_device<float, 4 >,
//        debug_device<float, 5 >
//    };
//
//    debuggers[degree - 1](n);
//}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="interface functions (double)">

template<>
void
CUDA_prepare_device<double>(int degree, int n, int rhs_cnt)
{
    static device_fun_ii prepares[] = {
        prepare_device<double, 1 >,
        prepare_device<double, 2 >,
        prepare_device<double, 3 >,
        prepare_device<double, 4 >,
        prepare_device<double, 5 >
    };

    prepares[degree - 1](n, rhs_cnt);
}

template<>
void
CUDA_init_fronts<double>(int degree, int n, int rhs_cnt)
{
    static device_fun_ii initializers[] = {
        init_fronts<double, 1 >,
        init_fronts<double, 2 >,
        init_fronts<double, 3 >,
        init_fronts<double, 4 >,
        init_fronts<double, 5 >
    };

    initializers[degree - 1](n, rhs_cnt);
}

template<>
void
CUDA_factorize_matrix<double>(int degree, int n)
{
    static device_fun_i factorize[] = {
        factorize_matrix<double, 1 >,
        factorize_matrix<double, 2 >,
        factorize_matrix<double, 3 >,
        factorize_matrix<double, 4 >,
        factorize_matrix<double, 5 >
    };

    factorize[degree - 1](n);
}

template<>
void
CUDA_solve<double>(int degree, int n, int rhs_cnt)
{
    static device_fun_ii solvers[] = {
        solve_equation<double, 1 >,
        solve_equation<double, 2 >,
        solve_equation<double, 3 >,
        solve_equation<double, 4 >,
        solve_equation<double, 5 >
    };

    solvers[degree - 1](n, rhs_cnt);
}

template<>
double
CUDA_error<double>(int degree, int n)
{
    typedef double (*error_fun)(int);
    static error_fun calculators[] = {
        calculate_error<double, 1 >,
        calculate_error<double, 2 >,
        calculate_error<double, 3 >,
        calculate_error<double, 4 >,
        calculate_error<double, 5 >
    };

    return calculators[degree - 1](n);
}

template<>
void
CUDA_print_result<double>(int degree, int n, std::ostream &ostr)
{
    typedef void (*print_fun)(int, std::ostream &);
    static print_fun printers[] = {
        print_result<double, 1 >,
        print_result<double, 2 >,
        print_result<double, 3 >,
        print_result<double, 4 >,
        print_result<double, 5 >
    };

    printers[degree - 1](n, ostr);
}

template<>
void
CUDA_debug<double>(int degree, int n)
{
    typedef void (*print_fun)(int);
    static print_fun debuggers[] = {
        debug_device<double, 1 >,
        debug_device<double, 2 >,
        debug_device<double, 3 >,
        debug_device<double, 4 >,
        debug_device<double, 5 >
    };

    debuggers[degree - 1](n);
}
// </editor-fold>

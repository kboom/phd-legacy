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
#include <device/merge_utils.cuh>
#include <utils.h>
#include <magma.h>

// Quadrature points count
// IMPORTANT do not set less than 2
#define QPC 4
#define MAX_MERGES 23

// this must not be greater than maximal threads per block
#define MAX_SHARED_ROW 1024

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

__constant__ FEM_PRECISION points[QPC] = {-0.861136311594052,
    -0.339981043584856,
    0.339981043584856,
    0.861136311594052};

__constant__ FEM_PRECISION weights[QPC] = {0.347854845137454,
    0.652145154862546,
    0.652145154862546,
    0.347854845137454};

__device__ FEM_PRECISION *d_Bh;
__device__ FEM_PRECISION *d_Lh;
__device__ FEM_PRECISION *d_Bv;
__device__ FEM_PRECISION *d_Lv;
__device__ FEM_PRECISION *d_assembled[MAX_MERGES];

__device__ FEM_PRECISION *d_Nvals[2];
__device__ FEM_PRECISION *d_dNvals[2];

__device__ p_2D<FEM_PRECISION> *d_coords;
__device__ FEM_PRECISION *d_fun_vals;
__device__ FEM_PRECISION *d_der_vals;

/**
 * Exact solution
 * @param x
 * @return
 */
template<class T>
inline T
__device__ __host__
_u(T /*x1*/, T x2)
{
    return x2 * x2;
}

// <editor-fold defaultstate="collapsed" desc="equation parameters">

template<class T>
__device__ __host__
inline T
_f(T /*x1*/, T /*x2*/)
{
    // A_u = _f
    return T(-2.0);
}

/**
 * Exact solution
 * @param x
 * @return
 */
template<class T>
inline T
__device__ __host__
_grad_u(T /*x1*/, T x2)
{
    return 2 * x2;
}

template<class T>
__device__ __host__
inline T
_shift(T /*x1*/, T x2)
{
    return x2;
}

template<class T>
__device__ __host__
inline T
_grad_shift(T /*x1*/, T /*x2*/)
{
    return 1;
}

// </editor-fold>

/**
 * Returns index that can be used for retrieving value of previously calculated
 * values of base functions (d_Nval) and their derivatives (d_dNval).
 * @param point_i - index of quadrature point [0, QUADRATURE_POINTS_CNT-1]
 * @param fun_i - index of a function [0, fun_cnt-1]
 * @param interval_i - index of interval (function support) [0, degree-1]
 * @param fun_cnt - total number of functions
 * @return index that can be used for d_Nvals and d_dNvals
 */
__device__
inline int
point_mem_idx(int point_i, int fun_i, int interval_i, int fun_cnt)
{
    return (interval_i * QPC + point_i) * fun_cnt + fun_i;
}

/**
 * Helper function to acquire assembled matrix for provided merge number.
 */
template<class T>
__device__
inline T*
get_assembled_for_merge(int merge_num)
{
    return reinterpret_cast<T*>(d_assembled[merge_num]);
}

/**
 * Helper function to acquire assembled matrix for provided step
 * @param step step = {0, 1, 2, ...}
 */
template<merge_type mt, class T>
struct get_assembled {
    __device__
    static T*
    f(int step);
};

template<class T>
struct get_assembled<_H, T> {
    __device__
    static T*
    f(int step)
    {
        return get_assembled_for_merge<T>(2 * step + 1);
    }
};

template<class T>
struct get_assembled<_V, T> {
    __device__
    static T*
    f(int step)
    {
        return get_assembled_for_merge<T>(2 * step + 2);
    }
};

template<class T>
__device__
inline T*
get_left_h(int global_mblock_idx, int small_size)
{
    return reinterpret_cast<T*>(&d_Bv[2 * global_mblock_idx * small_size]);
}

template<class T>
__device__
inline T*
get_right_h(T* left, int small_size)
{
    return left + small_size;
}

template<class T>
__device__
inline T*
get_top_v(int global_mblock_idx, int small_size)
{
    return reinterpret_cast<T*>(&d_Bh[global_mblock_idx * small_size]);
}

template<class T>
__device__
inline T*
get_bottom_v(T* top, int small_size)
{
    return top + gridDim.x * small_size;
}

template<merge_type mt, class T>
struct get_mergeM {
    __device__ static T* f();
};

template<class T>
struct get_mergeM<_H, T> {
    __device__ static T* f()
    {
        return reinterpret_cast<T*>(d_Bh);
    }
};

template<class T>
struct get_mergeM<_V, T> {
    __device__ static T* f()
    {
        return reinterpret_cast<T*>(d_Bv);
    }
};

template<int degree, class T>
__device__
inline T
get_N(int point_i, int fun_i, int interval_i, int _n)
{
    return d_Nvals[degree & 1][point_mem_idx(point_i, fun_i, interval_i, _n)];
}

template<int degree, class T>
__device__
inline T
get_dN(int point_i, int fun_i, int interval_i, int _n)
{
    return d_dNvals[degree & 1][point_mem_idx(point_i, fun_i, interval_i, _n)];
}

template<int degree, class T>
__device__
inline T
fun_B(int_2 point_idx,
      int_2 fun1_xy,
      int_2 fun2_xy,
      int_2 domain1_xy,
      int_2 domain2_xy,
      int _n)
{
    return    get_N<degree, T> (point_idx.y, fun1_xy.y, domain1_xy.y, _n)
            * get_dN<degree, T>(point_idx.x, fun1_xy.x, domain1_xy.x, _n)
            * get_N<degree, T> (point_idx.y, fun2_xy.y, domain2_xy.y, _n)
            * get_dN<degree, T>(point_idx.x, fun2_xy.x, domain2_xy.x, _n)

            + get_N<degree, T> (point_idx.x, fun1_xy.x, domain1_xy.x, _n)
            * get_dN<degree, T>(point_idx.y, fun1_xy.y, domain1_xy.y, _n)
            * get_N<degree, T> (point_idx.x, fun2_xy.x, domain2_xy.x, _n)
            * get_dN<degree, T>(point_idx.y, fun2_xy.y, domain2_xy.y, _n);
}

/**
 * This function evaluates one item of RHS vector for weak formulation
 * @param x point on the plane
 * @param point indexes of quadrature points in 2D
 * @param fun_xy (x,y) index of basis function
 * @param domain_xy (x,y) index of part of the domain where function
 *        is being integrated
 * @param _n total number of basis functions in each dimension
 * @return value of function being integrated
 */
template<int degree, class T>
__device__
inline T
fun_L(p_2D<T> x, int_2 point, int_2 fun_xy, int_2 domain_xy, int _n)
{
    return    get_N<degree,T>(point.x, fun_xy.x, domain_xy.x, _n)
            * (_f(x.x, x.y)
            * get_N<degree,T>(point.y, fun_xy.y, domain_xy.y, _n)
            - get_dN<degree,T>(point.y, fun_xy.y, domain_xy.y, _n));
}

template<int degree, class T>
__device__
inline T
eval_B(int_2 fun1_xy, int_2 fun2_xy, int_2 element_xy, int _n)
{
    T sum(0);
    p_2D<T> aa(interval<T>(element_xy.x+degree), interval<T>(element_xy.y+degree));

    aa/=2.0;

    for (int x = 0; x < QPC; ++x)
        for (int y = 0; y < QPC; ++y)
        {
            sum += weights[x] * weights[y]
                    * fun_B<degree,T>(int_2(x, y),
                                      fun1_xy,
                                      fun2_xy,
                                      domain_part_xy<degree>(element_xy, fun1_xy),
                                      domain_part_xy<degree>(element_xy, fun2_xy),
                                      _n);
        }

    return aa.x * aa.y * sum;
}

/**
 * This function evaluates one cell of RHS vector. It uses Gaussian quadratures
 * to integrate the function which is product of basis functions in two
 * dimensions on one element.
 * @param fun_xy 2D index of 2D basis function
 * @param element_xy 2D index of element
 * @param _n total number of functions in each dimension
 * @return
 */
template<int degree, class T>
__device__
T
eval_L(int_2 fun_xy, int_2 element_xy, int _n)
{
    T sum(0);
    p_2D<T> aa, bb;
    T a = d_knot_vector[element_xy.x + degree];
    T b = d_knot_vector[element_xy.x + degree + 1];
    aa.x = (b - a) / 2.0;
    bb.x = (b + a) / 2.0;

    a = d_knot_vector[element_xy.y + degree];
    b = d_knot_vector[element_xy.y + degree + 1];
    aa.y = (b - a) / 2.0;
    bb.y = (b + a) / 2.0;

    #pragma unroll
    for (int x = 0; x < QPC; ++x)
        #pragma unroll
        for (int y = 0; y < QPC; ++y)
        {
            sum += weights[x] * weights[y]
                    * fun_L<degree>(aa * p_2D<T>(points[x],points[y]) + bb,
                                    int_2(x, y),
                                    fun_xy,
                                    domain_part_xy<degree>(element_xy, fun_xy),
                                    _n);
        }
    return aa.x*aa.y*sum;
}

/**
 * This function evaluates one cell of local RHS vector
 * @param _n - number of elements in one dimension
 */
template<int degree, class T>
__global__
void
init_local_L(int _n)
{
    int_2 element_xy(blockIdx.x, blockIdx.y);
    int_2 function_xy = idx_to_xy<degree+1>(threadIdx.x);

    if(element_xy >= _n || threadIdx.x >= (degree+1)*(degree+1)) return;

    int n = L_mem_idx<degree>(element_xy, threadIdx.x, _n);
    if(element_xy.y == 0 && function_xy.y == 0)
    {
        d_Lh[n] = 0;
    }
    else if(element_xy.y == _n-1 && function_xy.y == degree)
    {
        d_Lh[n] = 0;
    }
    else
    {
        function_xy += element_xy;
        d_Lh[n] = eval_L<degree, T>(function_xy, element_xy, _n + degree);
    }
}

/**
 * This function evaluates one cell of local matrix
 * @param _n - number of elements in one dimension
 */
template<int degree, class T>
__global__
void
init_local_B(int _n)
{
    int_2 element_xy(blockIdx.x, blockIdx.y);
    int_2 product_ij(threadIdx.x, threadIdx.y);

    if(element_xy >= _n || product_ij >= (degree+1)*(degree+1)) return;

    int_2 fun_i_xy = idx_to_xy<degree+1>(product_ij.x);
    int_2 fun_j_xy = idx_to_xy<degree+1>(product_ij.y);

    int n = B_mem_idx<degree>(element_xy, product_ij, _n);
    if(element_xy.y == 0 && fun_j_xy.y == 0)
    {
        d_Bh[n] = (fun_j_xy == fun_i_xy);
    }
    else if(element_xy.y == _n - 1 && fun_j_xy.y == degree)
    {
        d_Bh[n] = (fun_j_xy == fun_i_xy);
    }
    else
    {
        fun_j_xy += element_xy;
        fun_i_xy += element_xy;
        d_Bh[n] = eval_B<degree, T>(fun_j_xy, fun_i_xy, element_xy, _n + degree);
    }
}

/**
 *
 * @param _n - number of functions
 */
template<int degree>
__global__
void
init_base_functions_and_derivatives(int _n)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= _n) return;

    #pragma unroll
    for (int i = 0; i < (degree + 1) * QPC; ++i)
    {
        d_Nvals[0][i * _n + n] = 0;
        d_Nvals[1][i * _n + n] = 0;
        d_dNvals[0][i * _n + n] = 0;
        d_dNvals[1][i * _n + n] = 0;
    }

    #pragma unroll
    for (int i = 0; i < QPC; ++i)
    {
        d_Nvals[0][i * _n + n] = 1;
    }
}

/**
 * This function is used for incremental evaluation of base functions and their
 * derivatives.
 * @param _n - total number of functions in one dimension
 * @param _d - degree of a function to be evaluated
 */
template<class T>
__global__
void
update_base(int _n, int _d)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x; // function number
    if (n >= _n) return;

    T h1 = interval<T > (n, n + _d);
    T h2 = interval<T > (n + 1, n + _d + 1);

    if (is_zero(h1))
    {
        #pragma unroll
        for (int i = 0; i < QPC; ++i)
        {
            // function
            d_Nvals[_d & 1][point_mem_idx(i, n, 0, _n)] = 0;
            //derivative
            d_dNvals[_d & 1][point_mem_idx(i, n, 0, _n)] = 0;
        }
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < QPC; ++i)
        {
            T x = (points[i] / 2.0 + 0.5) * interval<T > (n);
            // function
            d_Nvals[_d & 1][point_mem_idx(i, n, 0, _n)] =
                    x * d_Nvals[(~_d)&1][point_mem_idx(i, n, 0, _n)] / h1;
            //derivative
            d_dNvals[_d & 1][point_mem_idx(i, n, 0, _n)] =
                    (x * d_dNvals[(~_d)&1][point_mem_idx(i, n, 0, _n)]
                    + d_Nvals[(~_d)&1][point_mem_idx(i, n, 0, _n)]) / h1;
        }
    }

    for (int j = 1; j < _d; ++j)
    {
        #pragma unroll
        for (int i = 0; i < QPC; ++i)
        {
            T sum_fun, sum_der;
            T x = (points[i] / 2.0 + 0.5) * interval<T > (n + j)
                    + d_knot_vector[n + j];
            if (is_zero(h1))
            {
                sum_fun = 0;
                sum_der = 0;
            }
            else
            {
                sum_fun = (x - d_knot_vector[n]) *
                        d_Nvals[(~_d)&1][point_mem_idx(i, n, j, _n)] / h1;
                sum_der = ((x - d_knot_vector[n])
                        * d_dNvals[(~_d)&1][point_mem_idx(i, n, j, _n)]
                        + d_Nvals[(~_d)&1][point_mem_idx(i, n, j, _n)]) / h1;
            }

            if (is_zero(h2))
            {
                sum_fun += 0;
                sum_der += 0;
            }
            else
            {
                sum_fun += (d_knot_vector[n + _d + 1] - x)
                        * d_Nvals[(~_d)&1][point_mem_idx(i, n + 1, j - 1, _n)]
                        / h2;
                sum_der += ((d_knot_vector[n + _d + 1] - x)
                        * d_dNvals[(~_d)&1][point_mem_idx(i, n + 1, j - 1, _n)]
                        - d_Nvals[(~_d)&1][point_mem_idx(i, n + 1, j - 1, _n)])
                        / h2;
            }
            d_Nvals[_d & 1][point_mem_idx(i, n, j, _n)] = sum_fun;
            d_dNvals[_d & 1][point_mem_idx(i, n, j, _n)] = sum_der;
        }
    }

    if (is_zero(h2))
    {
        #pragma unroll
        for (int i = 0; i < QPC; ++i)
        {
            d_Nvals[_d & 1][point_mem_idx(i, n, _d, _n)] = 0;
            d_dNvals[_d & 1][point_mem_idx(i, n, _d, _n)] = 0;
        }
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < QPC; ++i)
        {
            T x = (points[i] / 2.0 + 0.5) * interval<T > (n + _d)
                    + d_knot_vector[n + _d];
            d_Nvals[_d & 1][point_mem_idx(i, n, _d, _n)] =
                    (d_knot_vector[n + _d + 1] - x)
                    * d_Nvals[(~_d)&1][point_mem_idx(i, n + 1, _d - 1, _n)] / h2;
            d_dNvals[_d & 1][point_mem_idx(i, n, _d, _n)] =
                    ((d_knot_vector[n + _d + 1] - x)
                    * d_dNvals[(~_d)&1][point_mem_idx(i, n + 1, _d - 1, _n)]
                    - d_Nvals[(~_d)&1][point_mem_idx(i, n + 1, _d - 1, _n)]) / h2;
        }
    }

}

/**
 * This function performs first merging step for local matrices.
 * Every block of threads merges (d+1)**2 matrices and eliminates one row.
 * Number of threads in block should be (2d+1)**2-1
 * @param _n total number of elements in one dimension
 */
template<int degree, class T>
__global__
void
initial_merge(int _n)
{
    const int sizeP = (degree+1)*(degree+1);      // dimension of small matrix
    const int sizeM = (2*degree+1)*(2*degree+1);  // dimension of merged matrix
    const int merged_cnt = (degree+1)*(degree+1);  // number of merged elements
    __shared__ volatile T s_B[sizeM][sizeM];
    __shared__ volatile T s_L[sizeM];
    clear_matrix_and_rhs(s_B, s_L);
    __syncthreads();

    // Load data to shared memory
    if(threadIdx.x < merged_cnt)
    {
        int_2 local_element_xy = idx_to_xy(threadIdx.x, degree+1);
        int_2 element_xy(blockIdx.x * (degree+1) + local_element_xy.x,
                         blockIdx.y * (degree+1) + local_element_xy.y);

        #pragma unroll
        for (int j = 0; j < sizeP ; ++j)
        {
            #pragma unroll
            for (int i = 0; i < sizeP ; ++i)
            {
                s_B[mapping<degree>(threadIdx.x, j)][mapping<degree>(threadIdx.x, i)]
                        += d_Bh[B_mem_idx<degree>(element_xy,int_2(i, j), _n)];
//                 TODO this might be necessary:
//                __syncthreads();
            }
            s_L[mapping<degree>(threadIdx.x, j)]
                    += d_Lh[L_mem_idx<degree>(element_xy, j, _n)];
        }
    }
    __syncthreads();

    calculate_first_row(s_B, s_L);
    __syncthreads();

    eliminate_first_row(s_B, s_L);
    __syncthreads();

    // Put data back into global memory
    int global_idx = xy_to_idx(blockIdx.x, blockIdx.y, _n / (degree+1));
    array_2d<T> assembled(get_assembled_for_merge<T>(0), sizeM + 1); // remember (+1) for RHS
    if(threadIdx.x == 0)
    {
        #pragma unroll
        for (int j = 0; j < sizeM; ++j)
            assembled[global_idx][j] = s_B[0][j];
        assembled[global_idx][sizeM] = s_L[0];
    }

    array_2d<T> dest(get_mergeM<_V, T>::f() + global_idx * (sizeM + 1)*(sizeM - 1),
                     sizeM + 1); // remember (+1) for RHS

    #pragma unroll
    for (int i = 0; i < sizeM - 1; ++i)
        dest[i][threadIdx.x+1] = s_B[i+1][threadIdx.x+1];

    dest[threadIdx.x][sizeM] = s_L[threadIdx.x+1];
}

template<merge_type mt, class T>
__global__
void
clean_AB(int step, int u_bound)
{
    fast_clean_vector(get_assembled<mt, T>::f(step), u_bound);
}

template<merge_type mt, class T>
__global__
void
clean_CD(int u_bound)
{
    fast_clean_vector(get_mergeM<mt, T>::f(), u_bound);
}

template<merge_type mt, int degree, class T>
struct ABCD_loader {
    static void f(int , int , int , int_2 , int_2 , int , int);
};

template<int degree, class T>
struct ABCD_loader<_H, degree, T> {

    __device__
    static void
    f(int step, int prl, int pmrl, int_2 segment_l, int_2 segment_r, int e_cnt, int loops)
    {
        int global_mblock_idx = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);
        int small_size = prl * (pmrl + 1);  // (+1) for RHS

        T* left_part = get_left_h<T>(global_mblock_idx, small_size);
        T* right_part = get_right_h(left_part, small_size);

        array_2d<T> left(left_part + pmrl - prl, pmrl + 1);  // (+1) for RHS
        array_2d<T> right(right_part + pmrl - prl, pmrl + 1);  // (+1) for RHS

        int rl = row_length<_H, degree>::f(step);
        T* tmp = get_assembled<_H, T>::f(step) + (global_mblock_idx * (rl + 1) * e_cnt);
        array_2d<T> AB(tmp, rl + 1);
        tmp = get_mergeM<_H, T>::f() + (global_mblock_idx * (rl + 1) * (rl - e_cnt));
        array_2d<T> CD(tmp - e_cnt * (rl + 1), rl + 1); // shift back as if eliminated rows were stored in this place in memory

        int idx = threadIdx.x;
        for (int loop = 0; loop < loops && idx < prl + 1; ++loop, idx += blockDim.x)
        {
            for (int i = 0; i < segment_l.x; ++i)
                CD[s2b_hl<degree>(i, prl)][s2b_hl<degree>(idx, prl)] = left[i][idx];
            for (int i = segment_l.x; i < segment_l.y; ++i)
                AB[s2b_hl<degree>(i, prl)][s2b_hl<degree>(idx, prl)] = left[i][idx];
            for (int i = segment_l.y; i < prl; ++i)
                CD[s2b_hl<degree>(i, prl)][s2b_hl<degree>(idx, prl)] = left[i][idx];
        }
        __syncthreads();

        idx = threadIdx.x;
        for (int loop = 0; loop < loops && idx < prl + 1; ++loop, idx += blockDim.x)
        {
            for (int i = 0; i < segment_r.x; ++i)
                CD[s2b_hr<degree>(i, prl)][s2b_hr<degree>(idx, prl)] += right[i][idx];
            for (int i = segment_r.x; i < segment_r.y; ++i)
                AB[s2b_hr<degree>(i, prl)][s2b_hr<degree>(idx, prl)] += right[i][idx];
            for (int i = segment_r.y; i < prl; ++i)
                CD[s2b_hr<degree>(i, prl)][s2b_hr<degree>(idx, prl)] += right[i][idx];
        }
    }
};

template<int degree, class T>
struct ABCD_loader<_V, degree, T> {

    __device__
    static void
    f(int step, int prl, int pmrl, int_2 segment_t, int_2 segment_b, int e_cnt, int loops)
    {
        int global_mblock_idx = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);
        int small_size = prl * (pmrl + 1);

        T* top_part = get_top_v<T>(xy_to_idx(blockIdx.x, 2 * blockIdx.y, gridDim.x), small_size);
        T* bottom_part = get_bottom_v(top_part, small_size);

        array_2d<T> top(top_part + pmrl - prl, pmrl + 1);
        array_2d<T> bottom(bottom_part + pmrl - prl, pmrl + 1);

        int rl = row_length<_V, degree>::f(step);
        T* tmp = get_assembled<_V, T>::f(step) + (global_mblock_idx * (rl + 1) * e_cnt);
        array_2d<T> AB(tmp, rl + 1);
        tmp = get_mergeM<_V, T>::f() + (global_mblock_idx * (rl + 1) * (rl - e_cnt));
        array_2d<T> CD(tmp - e_cnt * (rl+1), rl + 1); // shift back as if eliminated rows were stored in this place in memory

        int idx = threadIdx.x;
        for (int loop = 0; loop < loops && idx < prl + 1; ++loop, idx += blockDim.x)
        {
            for (int i = 0; i < segment_t.x; ++i)
                CD[s2b_vt<degree>(i, prl)][s2b_vt<degree>(idx, prl)] = top[i][idx];
            for (int i = segment_t.x; i < segment_t.y; ++i)
                AB[s2b_vt<degree>(i, prl)][s2b_vt<degree>(idx, prl)] = top[i][idx];
            for (int i = segment_t.y; i < prl; ++i)
                CD[s2b_vt<degree>(i, prl)][s2b_vt<degree>(idx, prl)] = top[i][idx];
        }
        __syncthreads();

        idx = threadIdx.x;
        for (int loop = 0; loop < loops && idx < prl + 1; ++loop, idx += blockDim.x)
        {
            for (int i = 0; i < segment_b.x; ++i)
                CD[s2b_vb<degree>(i, prl)][s2b_vb<degree>(idx, prl)] += bottom[i][idx];
            for (int i = segment_b.x; i < segment_b.y; ++i)
                AB[s2b_vb<degree>(i, prl)][s2b_vb<degree>(idx, prl)] += bottom[i][idx];
            for (int i = segment_b.y; i < prl; ++i)
                CD[s2b_vb<degree>(i, prl)][s2b_vb<degree>(idx, prl)] += bottom[i][idx];
        }
    }
};

template<merge_type mt, int degree, class T>
__global__
void
load_ABCD(int step,
          int prl,
          int pmrl,
          int_2 segment_1,
          int_2 segment_2,
          int e_cnt,
          int loops)
{
    ABCD_loader<mt, degree, T>::f(step, prl, pmrl, segment_1, segment_2, e_cnt, loops);
}

template<int degree, int GBLOCK_LEN, class T>
__device__
inline void
calculate_Astar(array_2d<T> &A, int idx, int cnt, int loops)
{
    __shared__ T s_row[GBLOCK_LEN];
    __shared__ T s_lead;

    for (int eliminated = 0; eliminated < cnt; ++eliminated)
    {
        int offset = 0;
        int lps = loops;
        while (lps--)
        {
            if (idx == 0) s_lead = A[eliminated][eliminated];
            __syncthreads();

            if (offset + idx < cnt && offset + idx > eliminated)
            {
                s_row[idx] = A[eliminated][offset + idx];
                s_row[idx] /= s_lead;
                A[eliminated][offset + idx] = s_row[idx];
            }
            __syncthreads();

            if (offset + idx < cnt && offset + idx > eliminated)
            {
                for (int minuend = eliminated + 1; minuend < cnt; ++minuend)
                    A[minuend][offset + idx] -= A[minuend][eliminated] * s_row[idx];
            }
            __syncthreads();
            offset += GBLOCK_LEN;
        }
    }
}

template<merge_type mt, int degree, int GBLOCK_LEN, class T>
__global__
void
calculate_Astar(int step, int rl, int cnt, int loops)
{
    int global_mblock_idx = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);
    array_2d<T> A(get_assembled<mt, T>::f(step)
                  + (global_mblock_idx * (rl + 1) * cnt), rl + 1);

    calculate_Astar<degree, GBLOCK_LEN>(A, threadIdx.x, cnt, loops);
}

template<int degree, int GBLOCK_LEN, class T>
__device__
inline void
calculate_BDstar(array_2d<T> &A, array_2d<T> &C, int e_cnt, int r_cnt, int inner_block)
{
    __shared__ T s_row[GBLOCK_LEN];
    __shared__ T s_lead;

    int idx = threadIdx.x;
    int offset = e_cnt + inner_block * GBLOCK_LEN;
    if (offset + idx > r_cnt) return;   // == r_cnt (thread for RHS)

    for(int i=0 ; i<e_cnt ; ++i)
    {
        if (idx == 0) s_lead = A[i][i];
        __syncthreads();

        s_row[idx] = A[i][offset + idx];
        s_row[idx] /= s_lead;
        A[i][offset + idx] = s_row[idx];

        for (int j = i + 1; j < e_cnt; ++j)
            if (offset + idx > i)
                A[j][offset + idx] -= A[j][i] * s_row[idx];
        for (int j = 0; j < r_cnt - e_cnt; ++j)
            if (offset + idx > i)
                C[j][offset + idx] -= C[j][i] * s_row[idx];
        __syncthreads();
    }
}

template<merge_type mt, int degree, int GBLOCK_LEN, class T>
__global__
void
calculate_BDstar(int step)
{
    int rl = row_length<mt, degree>::f(step);
    int cnt = eliminated<mt, degree>::f(step);
    int cols = (rl - cnt) + 1; // +1 for RHS

    int blocks_per_B = (cols / GBLOCK_LEN) + (cols % GBLOCK_LEN > 0);

    int global_mblock_idx = xy_to_idx(blockIdx.x / blocks_per_B,
                                      blockIdx.y,
                                      gridDim.x / blocks_per_B);

    array_2d<T> A(get_assembled<mt, T>::f(step) + (global_mblock_idx * (rl + 1) * cnt), rl + 1);
    array_2d<T> C(get_mergeM<mt, T>::f() + (global_mblock_idx * (rl + 1) * (rl - cnt)), rl + 1);

    calculate_BDstar<degree, GBLOCK_LEN>(A, C, cnt, rl, blockIdx.x % blocks_per_B);
}

template<int degree, int GBLOCK_LEN, class T>
__device__
inline void
calculate_Cstar(array_2d<T> &C, array_2d<T> &A,int idx, int row_cnt, int e_cnt)
{
    __shared__ T s_row[GBLOCK_LEN];

    for (int eliminated = 0; eliminated < e_cnt; ++eliminated)
    {
        for (int i = idx; i < e_cnt; i += blockDim.x)
            if(i > eliminated) s_row[i] = A[eliminated][i];
        __syncthreads();

        for (int i=idx ; i < row_cnt; i += blockDim.x)
        {
            T lead = C[i][eliminated];
            for (int c = eliminated + 1; c < e_cnt; ++c)
                C[i][c] -= lead * s_row[c];
        }
        __syncthreads();
    }
}

/**
 * max_row_len is required to know how much shared memory need to be allocated
 * for vertical and horizontal merge
 */
template<merge_type mt>
struct max_row_len {
    enum { val };
};

template<>
struct max_row_len<_H> {
    enum { val = 764 };
};

template<>
struct max_row_len<_V> {
    enum { val = 1532 };
};

template<merge_type mt, int degree, class T>
__global__
void
calculate_Cstar(int step)
{
    int rl = row_length<mt, degree>::f(step);
    int elmntd = eliminated<mt, degree>::f(step);
    int row_cnt = rl - elmntd;

    int global_mblock_idx = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);
    array_2d<T> A(get_assembled<mt, T>::f(step) + (global_mblock_idx * (rl + 1) * elmntd), rl + 1);
    array_2d<T> C(get_mergeM<mt, T>::f() + (global_mblock_idx * (rl + 1) * row_cnt), rl + 1);

    calculate_Cstar<degree, max_row_len<mt>::val>(C, A, threadIdx.x, row_cnt, elmntd);
}

template<class T>
__global__
void
rewrite_for_magma(int r_cnt, int pad)
{
    T* storage = get_mergeM<_V, T>::f();
    vertical_vec<T> RHS(storage + pad + r_cnt, r_cnt + pad + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    d_fun_vals[idx] = RHS[idx];
}

template<class T>
__global__
void
rewrite_after_magma(int r_cnt, int pad)
{
    T* storage = get_mergeM<_V, T>::f();
    vertical_vec<T> RHS(storage + pad + r_cnt, r_cnt + pad + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    RHS[idx] = d_fun_vals[idx];
}

template<class T>
__global__
void
rewrite_initial_result(int r_cnt, int pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    vertical_vec<T> RHS(get_mergeM<_V, T>::f() + pad + r_cnt, r_cnt + pad + 1);
    T* result = get_mergeM<_V, T>::f() + pad;
    result[idx] = RHS[idx];
}

template<int degree, int GBLOCK_LEN, class T>
__device__
inline void
backward_substitutionB(array_2d<T> &A, T* results, int e_cnt, int r_cnt, int inner_block)
{
    int offset = e_cnt + inner_block * GBLOCK_LEN;
    int idx = offset + threadIdx.x;
    if (idx >= r_cnt) return;

    for (int i = 0; i < e_cnt; ++i)
        A[i][idx] *= results[idx];
}

template<merge_type mt, int degree, int GBLOCK_LEN, class T>
__global__
void
backward_substitutionB(int step)
{
    int rl = row_length<mt, degree>::f(step);
    int cnt = eliminated<mt, degree>::f(step);
    int cols = (rl - cnt);

    int blocks_per_B = (cols / GBLOCK_LEN) + (cols % GBLOCK_LEN > 0);

    int global_mblock_idx = xy_to_idx(blockIdx.x / blocks_per_B,
                                      blockIdx.y,
                                      gridDim.x / blocks_per_B);

    array_2d<T> A(get_assembled<mt, T>::f(step) + (global_mblock_idx * (rl + 1) * cnt), rl + 1);
    T* results = get_mergeM<mt, T>::f() + global_mblock_idx * rl;

    backward_substitutionB<degree, GBLOCK_LEN>(A, results, cnt, rl, blockIdx.x % blocks_per_B);
}

template<int degree, class T>
__device__
inline void
backward_substitutionA(array_2d<T> &A, T* results, int e_cnt, int r_len)
{
    for (int i = e_cnt - 1; i >= 0; --i)
    {
        if(threadIdx.x == 0)
        {
            T sum = A[i][r_len];
            for (int j = i + 1; j < r_len; ++j)
                sum -= A[i][j];
            results[i] = sum;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < i; idx += blockDim.x)
            A[idx][i] *= results[i];

        __syncthreads();
    }
}

template<merge_type mt, int degree, class T>
__global__
void
backward_substitutionA(int step)
{
    int rl = row_length<mt, degree>::f(step);
    int cnt = eliminated<mt, degree>::f(step);

    int global_mblock_idx = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);

    array_2d<T> A(get_assembled<mt, T>::f(step) + (global_mblock_idx * (rl + 1) * cnt), rl + 1);
    T* results = get_mergeM<mt, T>::f() + global_mblock_idx * rl;

    backward_substitutionA<degree>(A, results, cnt, rl);
}

template<merge_type mt, int degree, class T>
struct solution_splitter {
    static void f(int);
};

template<int degree, class T>
struct solution_splitter<_H, degree, T> {
    __device__
    static void f(int step) {
        int rl = row_length<_H, degree>::f(step);
        int pmrl = previous_merge_row_length<_H, degree>::f(step);
        int prl = part_row_length<_H, degree>::f(step);
        int pelm = pmrl - prl;

        int global_mblock_idx = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);
        T* src = get_mergeM<_H, T>::f() + global_mblock_idx * rl;
        T* dst = get_mergeM<_V, T>::f() + 2 * global_mblock_idx * pmrl + pelm;

        for(int idx = threadIdx.x; idx < prl ; idx += blockDim.x)
            dst[idx] = src[s2b_hl<degree>(idx, prl)];

        dst += pmrl;
        for(int idx = threadIdx.x; idx < prl ; idx += blockDim.x)
            dst[idx] = src[s2b_hr<degree>(idx, prl)];
    }
};

template<int degree, class T>
struct solution_splitter<_V, degree, T> {
    __device__
    static void f(int step) {
        int rl = row_length<_V, degree>::f(step);
        int pmrl = previous_merge_row_length<_V, degree>::f(step);
        int prl = part_row_length<_V, degree>::f(step);
        int pelm = pmrl - prl;

        int global_mblock_idx = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);
        T* src = get_mergeM<_V, T>::f() + global_mblock_idx * rl;
        T* dst = get_mergeM<_H, T>::f() + xy_to_idx(blockIdx.x, 2 * blockIdx.y, gridDim.x) * pmrl + pelm;

        for(int idx = threadIdx.x; idx < prl ; idx += blockDim.x)
            dst[idx] = src[s2b_vt<degree>(idx, prl)];

        dst += pmrl * gridDim.x;
        for(int idx = threadIdx.x; idx < prl ; idx += blockDim.x)
            dst[idx] = src[s2b_vb<degree>(idx, prl)];
    }
};

template<merge_type mt, int degree, class T>
__global__
void
split_solution(int step)
{
    solution_splitter<mt, degree, T>::f(step);
}

template<int degree, class T>
__global__
void
final_backward_substitution()
{
    const int sizeM = (2*degree+1)*(2*degree+1);  // dimension of merged matrix

    int global_mblock_idx = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);
    T* results = get_mergeM<_V, T>::f() + global_mblock_idx * sizeM;
    T* A = get_assembled_for_merge<T>(0) + global_mblock_idx * (sizeM + 1);

    A[threadIdx.x + 1] *= results[threadIdx.x + 1];
    __syncthreads();

    if(threadIdx.x == 0)
    {
        T sum = A[sizeM];
        #pragma unroll
        for(int i = 1; i < sizeM; ++i) sum -= A[i];
        results[0] = sum;
    }
}

template<int degree, class T>
__global__
void
rewrite_result()
{
    const int sizeM = (2*degree+1)*(2*degree+1);  // dimension of merged matrix
    const int whole_matrix_size = gridDim.x * (degree + 1) + degree;// + degree;

    int src_idx_global = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);

    T* src = get_mergeM<_V, T>::f() + sizeM * src_idx_global;

    int dst_idx = xy_to_idx(blockIdx.x * (degree + 1),
                            blockIdx.y * (degree + 1),
                            whole_matrix_size);
    array_2d<T> dst(get_mergeM<_H, T>::f() + dst_idx, whole_matrix_size);

    if (blockIdx.x == 0 && threadIdx.x < degree)
        dst[threadIdx.y + degree][threadIdx.x] = src[mapping<degree>(xy_to_idx(0, degree, degree + 1), xy_to_idx(threadIdx.x, threadIdx.y, degree + 1))];

    if (blockIdx.y == 0 && threadIdx.y < degree)
        dst[threadIdx.y][threadIdx.x + degree] = src[mapping<degree>(xy_to_idx(degree, 0, degree + 1), xy_to_idx(threadIdx.x, threadIdx.y, degree + 1))];

    if (blockIdx.y == 0 && threadIdx.y < degree && blockIdx.x == 0 && threadIdx.x < degree)
        dst[threadIdx.y][threadIdx.x] = src[mapping<degree>(xy_to_idx(degree, 0, degree + 1), xy_to_idx(threadIdx.x, threadIdx.y, degree + 1))];

    dst[threadIdx.y + degree][threadIdx.x + degree] = src[mapping<degree>(xy_to_idx(degree, degree, degree + 1), xy_to_idx(threadIdx.x, threadIdx.y, degree + 1))];
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
        sum += d_Lh[i] * N<degree, T > (XX, i);
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
        sum += d_Lh[i] * N<degree > (x, i);
    return sum;
}

template<int degree, class T>
__global__
void
calculate_norm(T* result)
{
    int block_idx_global = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);
    array_2d<T> fun(reinterpret_cast<T *>(d_fun_vals) + block_idx_global * QPC * QPC, QPC);
    array_2d<T> grad(reinterpret_cast<T *>(d_der_vals) + block_idx_global * QPC * QPC, QPC);
    array_2d<p_2D<T> > coords(reinterpret_cast<p_2D<T> *>(d_coords) + block_idx_global * QPC * QPC, QPC);
    __shared__ T sum[QPC][QPC];

    T total = 0;
    sum[threadIdx.y][threadIdx.x] = weights[threadIdx.y] * weights[threadIdx.y]
            * pow(_u(coords[threadIdx.y][threadIdx.x].x, coords[threadIdx.y][threadIdx.x].y) - fun[threadIdx.y][threadIdx.x], 2);

    __syncthreads();
    if(threadIdx.x == 0 && threadIdx.y == 0)
        for(int i=0 ; i<QPC ; ++i)
            for(int j=0 ; j<QPC ; ++j)
                total += sum[i][j];
    __syncthreads();

    sum[threadIdx.y][threadIdx.x] = weights[threadIdx.y] * weights[threadIdx.y]
            * pow(_grad_u(coords[threadIdx.y][threadIdx.x].x, coords[threadIdx.y][threadIdx.x].y) - grad[threadIdx.y][threadIdx.x], 2);

    __syncthreads();
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        for(int i=0 ; i<QPC ; ++i)
            for(int j=0 ; j<QPC ; ++j)
                total += sum[i][j];

        result[block_idx_global] = total;
    }
}

template<int degree, class T>
__global__
void
calculate_function_in_quadrature_points()
{

    int block_idx_global = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);
    int elements_cnt = gridDim.x;
    array_2d<T> result(reinterpret_cast<T *>(d_fun_vals) + block_idx_global * QPC * QPC, QPC);
    array_2d<p_2D<T> > coords(reinterpret_cast<p_2D<T> *>(d_coords) + block_idx_global * QPC * QPC, QPC);
    array_2d<T> w(get_mergeM<_H, T>::f(), elements_cnt + degree);

    T sum(0);
    for (int x = 0; x <= degree; ++x)
        for (int y = 0; y <= degree; ++y)
        {
            sum += w[blockIdx.y + y][blockIdx.x + x]
                    * get_N<degree, T>(threadIdx.x, blockIdx.x + x, degree - x, base_funs_cnt<degree>(elements_cnt))
                    * get_N<degree, T>(threadIdx.y, blockIdx.y + y, degree - y, base_funs_cnt<degree>(elements_cnt));
        }
    result[threadIdx.y][threadIdx.x] = sum + _shift<T>(coords[threadIdx.y][threadIdx.x].x, coords[threadIdx.y][threadIdx.x].y);
}

template<int degree, class T>
__global__
void
calculate_gradient_in_quadrature_points()
{
    int block_idx_global = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);
    int elements_cnt = gridDim.x;
    array_2d<T> result(reinterpret_cast<T *>(d_der_vals) + block_idx_global * QPC * QPC, QPC);
    array_2d<p_2D<T> > coords(reinterpret_cast<p_2D<T> *>(d_coords) + block_idx_global * QPC * QPC, QPC);
    array_2d<T> w(get_mergeM<_H, T>::f(), elements_cnt + degree);

    T sum(0);
    for (int x = 0; x <= degree; ++x)
        for (int y = 0; y <= degree; ++y)
        {
            sum += w[blockIdx.y + y][blockIdx.x + x]
                    *(get_dN<degree, T>(threadIdx.x, blockIdx.x + x, degree - x, base_funs_cnt<degree>(elements_cnt))
                      * get_N<degree, T>(threadIdx.y, blockIdx.y + y, degree - y, base_funs_cnt<degree>(elements_cnt))
                    + get_dN<degree, T>(threadIdx.y, blockIdx.y + y, degree - y, base_funs_cnt<degree>(elements_cnt))
                      * get_N<degree, T>(threadIdx.x, blockIdx.x + x, degree - x, base_funs_cnt<degree>(elements_cnt)));
        }
    result[threadIdx.y][threadIdx.x] = sum + _grad_shift<T>(coords[threadIdx.y][threadIdx.x].x, coords[threadIdx.y][threadIdx.x].y);
}

template<int degree, class T>
__global__
void
calculate_coords()
{
    int global_idx = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x);

    array_2d<p_2D<T> > coords(reinterpret_cast<p_2D<T> *>(d_coords) + global_idx * QPC * QPC, QPC);

    T a = d_knot_vector[blockIdx.x + degree];
    T b = d_knot_vector[blockIdx.x + degree + 1];

    T aa = (b - a) / 2.0;
    T bb = (b + a) / 2.0;

    coords[threadIdx.y][threadIdx.x].x = aa * points[threadIdx.x] + bb;

    a = d_knot_vector[blockIdx.y + degree];
    b = d_knot_vector[blockIdx.y + degree + 1];

    aa = (b - a) / 2.0;
    bb = (b + a) / 2.0;

    coords[threadIdx.y][threadIdx.x].y = aa * points[threadIdx.y] + bb;
}

template<class T>
T
t_gen(int i, int N, int degree)
{
    if (i <= degree) return T(0);
    if (i < N + degree) return T(i - degree) / N;
    return T(1);
}

template<class T, int degree>
void
print_local_matrices_and_rhs_h(int n, int step)
{
    T *bb, *tmp;

    int m = mgrid_width<degree>(n);
    int max_wh = 7 * m / 4 * (degree * degree + degree) - degree * degree;
    int size = 2 * max_wh * (max_wh + 1);
    dim3 b = mblock_dim<_H, degree>::f(m,step);
    int blocks = b.x * b.y;

    bb = new T[size];
    T *bb_i = bb;

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Bh, sizeof (tmp)));
    gpuAssert(cudaMemcpy(bb, tmp, sizeof (T) * size, cudaMemcpyDeviceToHost));

    int cols = row_length<_H, degree>::f(step);
    int e_cnt = eliminated<_H, degree>::f(step);
    int r_cnt = cols - e_cnt;

    std::ostringstream fname;
    fname << "matrices/h_0_" << step << ".dat";
    std::clog << fname.str() << "\n";
    std::ofstream ofs(fname.str().c_str());
    for (int i = 0; i < blocks; ++i)
    {
        for (int r = 0 ; r < r_cnt ; ++r)
        {
            int skip = e_cnt;
//            int skip = 0;
            bb_i += skip;
            for(int j = skip ; j<=cols ; ++j)
                ofs << *(bb_i++) << " ";

            ofs << ";\n";
        }
        ofs << "\n";
    }

    delete [] bb;
}

template<class T, int degree>
void
print_local_matrices_and_rhs_v(int n, int step)
{
    T *bb, *tmp;

    int m = mgrid_width<degree>(n);
    int max_wh = 5 * m / 2 * (degree * degree + degree) - degree * degree;
    int size = max_wh * (max_wh + 1);
    dim3 b = mblock_dim<_V, degree>::f(m,step);
    int blocks = b.x * b.y;

    bb = new T[size];
    T *bb_i = bb;

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Bv, sizeof (tmp)));
    gpuAssert(cudaMemcpy(bb, tmp, sizeof (T) * size, cudaMemcpyDeviceToHost));

    int cols = row_length<_V, degree>::f(step);
    int e_cnt = eliminated<_V, degree>::f(step);
    int r_cnt = cols - e_cnt;

    std::ostringstream fname;
    fname << "matrices/v_0_" << step << ".dat";
    std::clog << fname.str() << "\n";
    std::ofstream ofs(fname.str().c_str());
    for (int i = 0; i < blocks; ++i)
    {
        for (int r = 0 ; r < r_cnt ; ++r)
        {
            int skip = e_cnt;
//            int skip = cols;
//            int skip = 0;
            bb_i += skip;
            for(int j = skip ; j<=cols ; ++j)
                ofs << *(bb_i++) << " ";

            ofs << ";\n";
        }
        ofs << "\n";
    }

    delete [] bb;
}

template<class T, int degree>
void
print_assembled_h(int n, int step)
{
    void* assembled[MAX_MERGES];
    gpuAssert(cudaMemcpyFromSymbol(assembled, d_assembled, sizeof (assembled)));


    int m = mgrid_width<degree>(n);
    int size = assembled_size<_H, degree>(m, step);
    T *tmp = new T[size];

    gpuAssert(cudaMemcpy(tmp, assembled[2 * step + 1],
                             sizeof (T) * size, cudaMemcpyDeviceToHost));

    int w = row_length<_H, degree>::f(step) + 1;
    int x = size / w;
    std::cout << step << "HORIZONTAL ASSEMBLED:\n";
    for (int i = 0; i < x; ++i)
    {
        for (int j = 0; j < w; ++j)
            std::cout << tmp[i * w + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n\n";

    delete [] tmp;
}

template<class T, int degree>
void
print_assembled_v(int n, int step)
{
    void* assembled[MAX_MERGES];
    gpuAssert(cudaMemcpyFromSymbol(assembled, d_assembled, sizeof (assembled)));

    T *tmp;

    int m = mgrid_width<degree>(n);
    int size = assembled_size<_V, degree>(m, step);

    tmp = new T[size];

    gpuAssert(cudaMemcpy(tmp, assembled[2 * step + 2],
                             sizeof (T) * size, cudaMemcpyDeviceToHost));

    int w = row_length<_V, degree>::f(step) + 1;
    int x = size / w;

    std::ostringstream fname;
    fname << "matrices/va_0_" << step << ".dat";
    std::clog << fname.str() << "\n";
    std::ofstream ofs(fname.str().c_str());
    for (int i = 0; i < x; ++i)
    {
        for (int j = 0; j < w; ++j)
            ofs << tmp[i * w + j] << " ";
        ofs << "\n";
    }
    ofs << "\n\n";

    delete [] tmp;
}

template<class T, int degree>
void
print_assembled_0(int n)
{
    void* assembled[MAX_MERGES];
    gpuAssert(cudaMemcpyFromSymbol(assembled, d_assembled, sizeof (assembled)));

    T *tmp;

    int size = ((2*degree+1)*(2*degree+1)+1)*n*n/((degree+1)*(degree+1));

    tmp = new T[size];

    gpuAssert(cudaMemcpy(tmp, assembled[0],
                             sizeof (T) * size, cudaMemcpyDeviceToHost));

    int w = (2*degree+1)*(2*degree+1) + 1;
    int x = size / w;
    std::cout << "INITIAL ASSEMBLED:\n";
    for (int i = 0; i < x; ++i)
    {
        for (int j = 0; j < w; ++j)
            std::cout << tmp[i * w + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n\n";

    delete [] tmp;
}

// <editor-fold defaultstate="collapsed" desc="Temporary section for seq check">

template<class T>
void
seq_print_and_copy(array_2d<T> &mat, array_2d<T> dst,int cnt, std::ostream &ostr = std::cout)
{
    for (int y = cnt; y < mat.row_length - 1; ++y)
    {
        for (int x = cnt; x < mat.row_length; ++x)
        {
            ostr << mat[y][x] << " ";
            dst[y-cnt][x-cnt] = mat[y][x];
        }
        ostr << ";\n";
    }
    ostr << "\n";
}

template<class T>
void
seq_print_assembled(array_2d<T> &mat,int cnt, std::ostream &ostr = std::cout)
{
    for (int y = 0; y < cnt; ++y)
    {
        for (int x = 0; x < mat.row_length; ++x)
        {
            ostr << mat[y][x] << " ";
        }
        ostr << "\n";
    }
}

template<class T>
void
seq_eliminate(array_2d<T> &mat, int cnt)
{
    for (int eliminated = 0; eliminated < cnt; ++eliminated)
    {
        for (int i=eliminated + 1; i < mat.row_length; ++i)
            mat[eliminated][i] /= mat[eliminated][eliminated];

        for (int mult = eliminated + 1; mult < mat.row_length - 1; ++mult)
            for (int i = eliminated + 1; i < mat.row_length; ++i)
                mat[mult][i] -= mat[eliminated][i] * mat[mult][eliminated];
    }
}

template<int degree, class T>
void
seq_merge_h(T* src, T* dst, int mgrid_width, int step)
{
    dim3 block_grid = mblock_dim<_H, degree>::f(mgrid_width, step);
    int merged = block_grid.x * block_grid.y * block_grid.z;

    int prl = part_row_length<_H, degree>::f(step);
    int smallSize = prl * (prl + 1);

    int rl = row_length<_H, degree>::f(step);
    array_2d<T> tmp(new T[rl * (rl + 1)], rl + 1);

    int e_cnt = eliminated<_H, degree>::f(step);

    std::ostringstream fname;
    fname << "matrices/h_1_" << step << ".dat";
    std::clog << fname.str() << "\n";
    std::ofstream ofs(fname.str().c_str());
    for (int block = 0; block < merged; ++block)
    {
        memset(tmp.p, 0, sizeof(T) * rl * (rl+1));
        array_2d<T> left(src + block * 2 * smallSize, prl + 1);
        array_2d<T> right(left.p + smallSize, prl + 1);

        for (int y = 0; y < prl; ++y)
        {
            for (int x = 0; x < prl + 1; ++x)
            {
                tmp[s2b_hl<degree>(y, prl)][s2b_hl<degree>(x, prl)] += left[y][x];
                tmp[s2b_hr<degree>(y, prl)][s2b_hr<degree>(x, prl)] += right[y][x];
            }
        }

//        seq_eliminate(tmp, e_cnt);
        seq_print_and_copy(tmp, array_2d<T>(dst + block * (rl - e_cnt) * (rl - e_cnt +1), rl - e_cnt+1), e_cnt, ofs);
    }

    delete [] tmp.p;
}

template<int degree, class T>
void
seq_merge_v(T* src, T* dst, int mgrid_width, int step)
{
    dim3 block_grid = mblock_dim<_V, degree>::f(mgrid_width, step);
    int merged = block_grid.x * block_grid.y * block_grid.z;

    int prl = part_row_length<_V, degree>::f(step);
    int smallSize = prl * (prl + 1);

    int rl = row_length<_V, degree>::f(step);
    array_2d<T> tmp(new T[rl * (rl + 1)], rl + 1);

    int e_cnt = eliminated<_V, degree>::f(step);

//    for (int i=0 ; i< merged * 2 * smallSize; ++i)
//        std::cout << src[i] << " ";
//    std::cout << "\n\n";

    std::ostringstream fname;
    fname << "matrices/v_1_" << step << ".dat";
    std::clog << fname.str() << "\n";
    std::ofstream ofs(fname.str().c_str());

    std::ostringstream fname_a;
    fname_a << "matrices/va_X_" << step << ".dat";
    std::clog << fname_a.str() << "\n";
    std::ofstream ofs_a(fname_a.str().c_str());

    for (int block = 0; block < merged; ++block)
    {
        memset(tmp.p, 0, sizeof(T) * rl * (rl+1));
        int_2 block_xy = idx_to_xy(block, block_grid.x);
        int bb = xy_to_idx(block_xy.x, 2*block_xy.y, block_grid.x);

        array_2d<T> top(src + bb * smallSize, prl + 1);
        array_2d<T> bottom(top.p + block_grid.x * smallSize, prl + 1);

//        std::cout << top << "\n\n";
//        std::cout << bottom << "\n\n";

        for (int y = 0; y < prl; ++y)
        {
            for (int x = 0; x < prl + 1; ++x)
            {
                tmp[s2b_vt<degree>(y, prl)][s2b_vt<degree>(x, prl)] += 1;//top[y][x];
                tmp[s2b_vb<degree>(y, prl)][s2b_vb<degree>(x, prl)] += 2;//bottom[y][x];
            }
        }

        seq_print_assembled(tmp, e_cnt, ofs_a);

//        seq_eliminate(tmp, e_cnt);
        seq_print_and_copy(tmp, array_2d<T>(dst + block * (rl - e_cnt) * (rl - e_cnt +1), rl - e_cnt+1), e_cnt, ofs);
    }

    ofs_a << "\n\n";

    delete [] tmp.p;
}

template<int degree, class T>
void
perform_sequential_check(T *array, int n, int steps)
{
    T* vert;
    T* horz;

    int m = mgrid_width<degree>(n);

    int max_wh = 5 * m / 2 * (degree * degree + degree) - degree * degree;
    int size = max_wh * (max_wh + 1);
    vert = new T[size];

    int mat_size = (2*degree + 1) * (2*degree + 1) - 1;
    int mat_cnt = n*n/((degree+1)*(degree+1));

    int idx=0;
    for (int i = 0; i < mat_cnt; ++i)
    {
        for (int x = 0; x < mat_size; ++x)
        {
            T* tmp = &array[i*(mat_size*(mat_size + 2))+x*(mat_size+2)+1];
            for (int j=0; j<mat_size + 1; ++j)
                vert[idx++] = tmp[j];
        }
    }

    max_wh = 7 * m / 4 * (degree * degree + degree) - degree * degree;
    int initial_size = n * n * (degree + 1) * (degree + 1) * (degree + 1) * (degree + 1);
    size = std::max(2 * max_wh * (max_wh + 1), initial_size);
    horz = new T[size];

//    for (int step = 0; step < steps; ++step)
//    {
//        seq_merge_h<degree>(vert, horz, m, step);
//        seq_merge_v<degree>(horz, vert, m, step);
//    }
      seq_merge_v<degree>(horz, vert, m, 7);

    delete [] horz;
    delete [] vert;
}

// </editor-fold>

template<class T, int degree>
void
print_local_matrices_and_rhs_after_first_merge(int n)
{
    T *bb, *tmp;

    int m = mgrid_width<degree>(n);
    // Allocate space for vertical merging
    // maximum width/heigh of matrix
    int max_wh = 5 * m / 2 * (degree * degree + degree) - degree * degree;

    int size_bb = max_wh * (max_wh + 1);

    bb = new T[size_bb];

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Bv, sizeof (tmp)));
    gpuAssert(cudaMemcpy(bb, tmp, sizeof (T) * size_bb, cudaMemcpyDeviceToHost));

//    const int sizeM = (2*degree + 1) * (2*degree + 1) - 1;

//    std::cout << "MATRICES AFTER FIRST MERGE:\n";
//    for (int i = 0; i < n*n/((degree+1)*(degree+1)); ++i)
//    {
//        for (int x = 0; x < sizeM; ++x)
//        {
//            print_vector_horizontal(&bb[i*(sizeM*(sizeM + 2))+x*(sizeM+2)+1], sizeM+1);
//            std::cout << ";\n";
//        }
//        std::cout << "\n";
//    }

    perform_sequential_check<degree>(bb, n, lg(m / 2));

    delete [] bb;
}

template<class T, int degree>
void
print_whole_matrix_and_rhs_assembled(int n)
{
    T *bb, *tmp, *ll;

    int size_bb = n * n * (degree + 1) * (degree + 1) * (degree + 1) * (degree + 1);
    int size_ll = n * n * (degree + 1) * (degree + 1);

    bb = new T[size_bb];
    ll = new T[size_ll];

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Bh, sizeof (tmp)));
    gpuAssert(cudaMemcpy(bb, tmp, sizeof (T) * size_bb, cudaMemcpyDeviceToHost));

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Lh, sizeof (tmp)));
    gpuAssert(cudaMemcpy(ll, tmp, sizeof (T) * size_ll, cudaMemcpyDeviceToHost));

    int B_size = (n + degree) * (n + degree);
    int L_size = (n + degree) * (n + degree);

    T** B = allocate_square_matrix<T>(B_size);
    T* L = new T[L_size];

    for(int i =0 ; i<L_size ; ++i) L[i] = 0;

    zeroize_square_matrix(B, B_size);

    std::cout << "WHOLE MATRIX\n";
    for (int y = 0; y < n; y++)
    {
        for (int x = 0; x < n; x++)
        {
            T* local = &(bb[((y*n)+x)*(degree+1)*(degree+1)*(degree+1)*(degree+1)]);
            T* rhs = &(ll[((y*n)+x)*(degree+1)*(degree+1)]);

            for(int j = 0 ; j < (degree+1)*(degree+1) ; ++j)
            {
                int_2 j_xy = idx_to_xy<(degree+1)>(j);
                int bj = xy_to_idx(x + j_xy.x, y + j_xy.y, n + degree);

                L[bj] += rhs[j];

                for (int i = 0; i < (degree+1)*(degree+1); i++)
                {
                    int_2 i_xy = idx_to_xy<(degree+1)>(i);
                    int bi = xy_to_idx(x + i_xy.x, y + i_xy.y, n + degree);

                    B[bj][bi] += local[(j*(degree+1)*(degree+1))+i];
                }
            }
        }
    }



    print_square_matrix(B, B_size);

    print_vector_vertical(L, L_size);
//    print_vector_vertical(ll, size_ll);

    free_square_matrix(B, B_size);
    delete [] L;
    delete [] bb;
    delete [] ll;
}

template<class T, int degree>
void
print_local_matrices(int n)
{
    T *bb, *tmp, *ll;

    int size_bb = n * n * (degree + 1) * (degree + 1) * (degree + 1) * (degree + 1);
    int size_ll = n * n * (degree + 1) * (degree + 1);

    bb = new T[size_bb];
    ll = new T[size_ll];

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Bh, sizeof (tmp)));
    gpuAssert(cudaMemcpy(bb, tmp, sizeof (T) * size_bb, cudaMemcpyDeviceToHost));

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Lh, sizeof (tmp)));
    gpuAssert(cudaMemcpy(ll, tmp, sizeof (T) * size_ll, cudaMemcpyDeviceToHost));

    int dim = (degree + 1) * (degree + 1);

    std::cout << "LOCAL MATRICES\n";
    for(int i=0 ; i<n*n ; ++i)
    {
        array_2d<T> M(bb + i * dim * dim, dim);
        T* rhs = ll + i * dim;
        int_2 xy = idx_to_xy(i, n);
        std::cout << "M{" << xy.y + 1  << "," << xy.x + 1 << "} = [\n";
        for(int y=0 ; y<dim ; ++y)
        {
            for(int x=0 ; x<dim ; ++x)
            {
                std::cout << M[y][x] << " ";
            }
            std::cout << "   " << rhs[y] << ";\n";
        }
        std::cout << "];\n";
    }
}

template<class T, int degree>
void
debug_device(int n)
{
//    int m = mgrid_width<degree>(n);
////    int max_wh = 5 * m / 2 * (degree * degree + degree) - degree * degree;
//    int max_wh = 7 * m / 4 * (degree * degree + degree) - degree * degree;
//    int size = max_wh * (max_wh + 1);
//
//    T *tmp, *bb = new T[size];
//
//    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Bh, sizeof (tmp)));
//    gpuAssert(cudaMemcpy(bb, tmp, sizeof (T) * size, cudaMemcpyDeviceToHost));
//
//    size = 64;
//    for(int i=0 ; i<size ; ++i)
//    {
//        if (i%8 == 0)
//            std::cout << "\n";
//        std::cout << bb[i] << " ";
//    }
//    std::cout << "\n";
//
//    delete [] bb;
}

template<class T, int degree>
void
calculate_base_funcs(int n)
{
    int tpb = 256;
    int block_count = ((n + degree) / tpb) + ((n + degree) % tpb ? 1 : 0);
    init_base_functions_and_derivatives<degree><<<block_count, tpb >>>(n + degree);
    check_error("init_base_functions_and_derivatives", cudaGetLastError());
    for (int i = 0; i < degree; ++i)
    {
        update_base<T><<<block_count, tpb>>>(n + degree, i + 1);
        check_error("update_base", cudaGetLastError());
    }
}

/**
 * Allocates appropriate number of arrays to store already assembled rows.
 * Arrays are indexed as follows. One array per step.
 * <ul>
 *  <li> 0    - initial merge
 *  <li> 2k+1 - k-th horizontal merge
 *  <li> 2k+2 - k-th vertical merge
 * </ul>
 * @param m width of multi-grid (obtained from size of full
 *          grid). It should always be a number which is a power of 2.
 */
template<class T, int degree>
void
allocate_memory_for_assembled_rows(int m)
{
    T * tmp_p[MAX_MERGES];
    for (int i = 0; i < MAX_MERGES; ++i) tmp_p[i] = 0;

    /*
     * Number of steps performed by solver.
     * Each step consists of horizontal and vertical merge (in this order)
     */
    int steps = steps_count(m);

    gpuAssert(cudaMalloc(&tmp_p[0], sizeof (T) * m * m / 4 * (pow<int>(2 * degree + 1, 2)+1)));
    for (int i = 0; i < steps; ++i)
    {
        gpuAssert(cudaMalloc(&tmp_p[2 * i + 1],
                                 sizeof (T) * assembled_size<_H, degree>(m, i)));
        gpuAssert(cudaMalloc(&tmp_p[2 * i + 2],
                                 sizeof (T) * assembled_size<_V, degree>(m, i)));
    }
    gpuAssert(cudaMemcpyToSymbol(d_assembled, tmp_p, sizeof (tmp_p)));
}

void
free_memory_for_assembled_rows()
{
    void* tmp_p[MAX_MERGES];
    gpuAssert(cudaMemcpyFromSymbol(tmp_p, d_assembled, sizeof (tmp_p)));
    for (int i = 0; i < MAX_MERGES; ++i)
        if (tmp_p[i]) gpuAssert(cudaFree(tmp_p[i]));
}

/**
 * Allocates necessary memory on device
 * @param n - number of elements in one dimension
 */
template<class T, int degree>
void
prepare_device(int n)
{
    T *tmp, *t;
    t = new T[n + 2 * degree + 1];
    for (int i = 0; i <= n + 2 * degree; ++i)
        t[i] = t_gen<T>(i, n, degree);

    // Allocate knot vector
    gpuAssert(cudaMalloc(&tmp, sizeof (T) * knots_cnt<degree>(n)));
    gpuAssert(cudaMemcpy(tmp, t, sizeof (T) * knots_cnt<degree>(n),
                             cudaMemcpyHostToDevice));
    gpuAssert(cudaMemcpyToSymbol(d_knot_vector, &tmp, sizeof (tmp)));
    delete[] t;

    int m = mgrid_width<degree>(n);
    // Allocate space for vertical merging
    // maximum width/heigh of matrix
    int max_wh = 5 * m / 2 * (degree * degree + degree) - degree * degree;

    int size = max_wh * (max_wh + 1);
    gpuAssert(cudaMalloc(&tmp, sizeof (T) * size));
    gpuAssert(cudaMemcpyToSymbol(d_Bv, &tmp, sizeof (tmp)));

    // Allocate space for horizontal merging
    max_wh = 7 * m / 4 * (degree * degree + degree) - degree * degree;

    int initial_size = n * n * (degree + 1) * (degree + 1) * (degree + 1) * (degree + 1);
    size = std::max(2 * max_wh * (max_wh + 1), initial_size);
    gpuAssert(cudaMalloc(&tmp, sizeof (T) * size));
    gpuAssert(cudaMemcpyToSymbol(d_Bh, &tmp, sizeof (tmp)));

    // Allocate fronts (L part)
    gpuAssert(cudaMalloc(&tmp, sizeof (T) * n * n * (degree + 1) * (degree + 1)));
    gpuAssert(cudaMemcpyToSymbol(d_Lh, &tmp, sizeof (tmp)));

    // Allocate fronts (L part) for horizontal merge
    gpuAssert(cudaMalloc(&tmp, sizeof (T) * n * n * (degree + 1) * (degree + 1)));
    gpuAssert(cudaMemcpyToSymbol(d_Lv, &tmp, sizeof (tmp)));

    allocate_memory_for_assembled_rows<T, degree>(m);

    // Allocate matrices for accumulative base function evaluation
    T * dev_ptrs[2];

    // functions
    gpuAssert(cudaMalloc(&dev_ptrs[0], sizeof (T) * base_funs_cnt<degree>(n) * (degree + 1) * QPC));
    gpuAssert(cudaMalloc(&dev_ptrs[1], sizeof (T) * base_funs_cnt<degree>(n) * (degree + 1) * QPC));
    gpuAssert(cudaMemcpyToSymbol(d_Nvals, dev_ptrs, sizeof (dev_ptrs)));

    // derivatives
    gpuAssert(cudaMalloc(&dev_ptrs[0], sizeof (T) * base_funs_cnt<degree>(n) * (degree + 1) * QPC));
    gpuAssert(cudaMalloc(&dev_ptrs[1], sizeof (T) * base_funs_cnt<degree>(n) * (degree + 1) * QPC));
    gpuAssert(cudaMemcpyToSymbol(d_dNvals, dev_ptrs, sizeof (dev_ptrs)));

    // Allocate space for result
    size = pow<int>(QPC, 2) * pow<int>(n, 2);
    gpuAssert(cudaMalloc(&tmp, sizeof (T) * size));
    gpuAssert(cudaMemset(tmp, 0, sizeof (T) * size));
    gpuAssert(cudaMemcpyToSymbol(d_fun_vals, &tmp, sizeof (tmp)));

    gpuAssert(cudaMalloc(&tmp, sizeof (T) * size));
    gpuAssert(cudaMemset(tmp, 0, sizeof (T) * size));
    gpuAssert(cudaMemcpyToSymbol(d_der_vals, &tmp, sizeof (tmp)));

    gpuAssert(cudaMalloc(&tmp, sizeof (p_2D<T>) * size));
    gpuAssert(cudaMemcpyToSymbol(d_coords, &tmp, sizeof (tmp)));

    cudaThreadSynchronize();
}

/**
 *
 * @param n number of elements
 */
template<int degree, class T>
void
prepare_result(int n)
{
    dim3 block_grid(n,n);
    dim3 thread_grid(QPC, QPC);

    calculate_coords<degree, T><<<block_grid, thread_grid>>>();
    check_error("calculate_coords", cudaGetLastError());
    calculate_gradient_in_quadrature_points<degree, T><<<block_grid, thread_grid>>>();
    check_error("calculate_gradient_in_quadrature_points", cudaGetLastError());
    calculate_function_in_quadrature_points<degree, T><<<block_grid, thread_grid>>>();
    check_error("calculate_function_in_quadrature_points", cudaGetLastError());
}

void
cleanup_device()
{
    void *tmp;

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Lh, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));
    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Lv, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Bh, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));
    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_Bv, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));

    free_memory_for_assembled_rows();

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_knot_vector, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));

    // Free matrices for accumulative base function evaluation
    void *dev_ptrs[2];
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
    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_coords, sizeof (tmp)));
    gpuAssert(cudaFree(tmp));
}

/**
 * This function initializes local stiffness matrices for equation
 * @param n number of elements in one dimension
 */
template<class T, int degree>
void
init_fronts(int n)
{
    calculate_base_funcs<T, degree>(n);     // TODO maybe move this to constructor?

    {
        dim3 block_grid(n, n, 1);
        dim3 thread_grid((degree + 1) * (degree + 1), 1, 1);

        init_local_L<degree, T><<<block_grid, thread_grid>>>(n);
        check_error("eval_L_for_element", cudaGetLastError());
    }

    {
        dim3 block_grid(n, n, 1);
        dim3 thread_grid((degree + 1) * (degree + 1), (degree + 1) * (degree + 1), 1);

        init_local_B<degree, T><<<block_grid, thread_grid>>>(n);
        check_error("eval_B_for_element", cudaGetLastError());
    }
    cudaThreadSynchronize();
}

template<class T, int degree>
inline void
launch_initial_merge(int n)
{
    dim3 block_grid(n/(degree+1), n/(degree+1));
    dim3 thread_grid((2 * degree + 1) * (2 * degree + 1) - 1);

    initial_merge<degree, T><<<block_grid, thread_grid>>>(n);
    check_error("initial_merge", cudaGetLastError());
}

template<merge_type mt, class T, int degree>
inline void
launch_clean_AB(int mgrid_width, int step)
{
    int cells = assembled_size<mt, degree>(mgrid_width, step);
    dim3 block_grid, thread_grid;
    create_configuration_for_num_threads(cells, block_grid, thread_grid);

    clean_AB<mt, T><<<block_grid, thread_grid>>>(step, cells);
    check_error("clean_AB - " + mt, cudaGetLastError());
}

template<merge_type mt, class T, int degree>
inline void
launch_clean_CD(int mgrid_width, int step)
{
    int cells = for_merging_size<mt, degree>(mgrid_width, step);
    dim3 block_grid, thread_grid;
    create_configuration_for_num_threads(cells, block_grid, thread_grid);

    clean_CD<mt, T><<<block_grid, thread_grid>>>(cells);
    check_error("clean_CD - " + mt, cudaGetLastError());
}

template<merge_type mt, class T, int degree>
inline void
launch_load_ABCD(int mgrid_width, int step)
{
    dim3 block_grid = mblock_dim<mt, degree>::f(mgrid_width, step);
    const int required_threads = part_row_length<mt, degree>::f(step) + 1;// (+1) RHS
    dim3 thread_grid = prepare_thread_grid(required_threads, MAX_SHARED_ROW);
    int loops = required_threads / thread_grid.x + (required_threads % thread_grid.x > 0);

    int prl = part_row_length<mt, degree>::f(step);
    int pmrl = previous_merge_row_length<mt, degree>::f(step);

    int_2 segment_0 = merged_segment<mt, 0, degree>::f(prl);
    int_2 segment_1 = merged_segment<mt, 1, degree>::f(prl);

    int e_cnt = segment_0.length();
    load_ABCD<mt, degree, T><<<block_grid, thread_grid>>>(step, prl, pmrl,
                                                          segment_0, segment_1,
                                                          e_cnt, loops);
    check_error("load_ABCD - " + mt, cudaGetLastError());
}

template<merge_type mt, class T, int degree>
inline void
launch_calculate_Astar(int mgrid_width, int step)
{
    dim3 block_grid = mblock_dim<mt, degree>::f(mgrid_width, step);
    const int required_threads = eliminated<mt, degree>::f(step);
    dim3 thread_grid = prepare_thread_grid(required_threads, MAX_SHARED_ROW);
    int loops = required_threads / thread_grid.x + (required_threads % thread_grid.x > 0);

    int rl = row_length<mt, degree>::f(step);
    int cnt = eliminated<mt, degree>::f(step);

    calculate_Astar<mt, degree, MAX_SHARED_ROW, T><<<block_grid, thread_grid>>>(step, rl, cnt, loops);
    check_error("calculate_Astar - " + mt, cudaGetLastError());
}

template<merge_type mt, class T, int degree>
inline void
launch_calculate_BDstar(int mgrid_width, int step)
{
    const int TPB = 128;
    int required_threads = row_length<mt, degree>::f(step) - eliminated<mt, degree>::f(step) + 1;
    const int gblocks_per_mblock = (required_threads / TPB) + (required_threads % TPB > 0);
    dim3 block_grid = mblock_dim<mt, degree>::f(mgrid_width, step);
    block_grid.x *= gblocks_per_mblock;
    dim3 thread_grid = prepare_thread_grid(required_threads, TPB);

    calculate_BDstar<mt, degree, TPB, T><<<block_grid, thread_grid>>>(step);
    check_error("calculate_BDstar - " + mt, cudaGetLastError());
}

template<merge_type mt, class T, int degree>
inline void
launch_calculate_Cstar(int mgrid_width, int step)
{
    dim3 block_grid = mblock_dim<mt, degree>::f(mgrid_width, step);
    const int required_threads = eliminated<mt, degree>::f(step);
    dim3 thread_grid = prepare_thread_grid(required_threads, MAX_SHARED_ROW);

    calculate_Cstar<mt, degree, T><<<block_grid, thread_grid>>>(step);
    check_error("calculate_Cstar - " + mt, cudaGetLastError());
}

template<class T>
void
solve_magma(int r_cnt, int pad);

template<>
void
solve_magma<double>(int r_cnt, int pad)
{
    typedef double T;
    int info, ldda, lddb, *ipiv;
    T *mat, *rhs;
    gpuAssert(cudaMemcpyFromSymbol(&mat, d_Bv, sizeof (T*)));
    gpuAssert(cudaMemcpyFromSymbol(&rhs, d_fun_vals, sizeof (T*)));
    ldda = r_cnt + pad + 1;
    lddb = r_cnt;
    mat += pad;

    ipiv = new int[r_cnt];
    for(int i=0 ; i<r_cnt ; ++i) ipiv[i] = i+1;

    int res;
    res = magma_dgetrf_nopiv_gpu(r_cnt, r_cnt, mat, ldda, /*ipiv,*/ &info);
    if (res != MAGMA_SUCCESS)
    {
        std::clog << "Factorization ERROR " << info << "\n";
    }
    res = magma_dgetrs_gpu(MagmaTrans, r_cnt, 1, mat, ldda, ipiv, rhs, lddb, &info);
    if (res != MAGMA_SUCCESS)
    {
        std::clog << "Solver ERROR " << info << "\n";
    }

    delete[] ipiv;
}

template<>
void
solve_magma<float>(int r_cnt, int pad)
{
    typedef float T;
    int info, ldda, lddb, *ipiv;
    T *mat, *rhs;
    gpuAssert(cudaMemcpyFromSymbol(&mat, d_Bv, sizeof (T*)));
    ldda = r_cnt + pad + 1;
    lddb = r_cnt;
    rhs = mat + r_cnt * ldda;
    mat += pad;

    ipiv = new int[r_cnt];
    for(int i=0 ; i<r_cnt ; ++i) ipiv[i] = i+1;

    magma_sgetrs_gpu(MagmaNoTrans,r_cnt, 1, mat, ldda, ipiv, rhs, lddb, &info);

    delete[] ipiv;
}

template<class T, int degree>
inline void
launch_magma_solver(int steps)
{
    const int TPB = 64;
    const int elmnt = eliminated<_V, degree>::f(steps-1);
    const int pad = elmnt;
    const int r_cnt = row_length<_V, degree>::f(steps-1) - elmnt;
    const int required_threads = r_cnt;
    dim3 thread_grid, block_grid;
    create_configuration_for_num_threads(required_threads, block_grid, thread_grid, TPB);

    rewrite_for_magma<T><<<block_grid, thread_grid>>>(r_cnt, pad);
    check_error("rewrite_for_magma", cudaGetLastError());

    solve_magma<T>(r_cnt, pad);

    rewrite_after_magma<T><<<block_grid, thread_grid>>>(r_cnt, pad);
    check_error("rewrite_after_magma", cudaGetLastError());
}

template<class T, int degree>
inline void
launch_rewrite_initial_result(int steps)
{
    const int TPB = 128;
    const int elmndt = eliminated<_V, degree>::f(steps-1);
    const int pad = elmndt;
    const int r_cnt = row_length<_V, degree>::f(steps-1) - elmndt;
    const int required_threads = r_cnt;
    const int gblocks_per_mblock = (required_threads / TPB) + (required_threads % TPB > 0);
    dim3 block_grid = gblocks_per_mblock;
    dim3 thread_grid = prepare_thread_grid(required_threads, TPB);

    rewrite_initial_result<T><<<block_grid, thread_grid>>>(r_cnt, pad);
    check_error("rewrite_initial_result", cudaGetLastError());
}

template<merge_type mt, class T, int degree>
inline void
launch_backward_substitutionB(int mgrid_width, int step)
{
    const int TPB = 128;
    int required_threads = row_length<mt, degree>::f(step) - eliminated<mt, degree>::f(step);
    const int gblocks_per_mblock = (required_threads / TPB) + (required_threads % TPB > 0);
    dim3 block_grid = mblock_dim<mt, degree>::f(mgrid_width, step);
    block_grid.x *= gblocks_per_mblock;
    dim3 thread_grid = prepare_thread_grid(required_threads, TPB);

    backward_substitutionB<mt, degree, TPB, T><<<block_grid, thread_grid>>>(step);
    check_error("backward_substitutionB - " + mt, cudaGetLastError());
}

template<merge_type mt, class T, int degree>
inline void
launch_backward_substitutionA(int mgrid_width, int step)
{
    dim3 block_grid = mblock_dim<mt, degree>::f(mgrid_width, step);
    const int required_threads = eliminated<mt, degree>::f(step);
    dim3 thread_grid = prepare_thread_grid(required_threads, MAX_SHARED_ROW);

    backward_substitutionA<mt, degree, T><<<block_grid, thread_grid>>>(step);
    check_error("backward_substitutionA - " + mt, cudaGetLastError());
}

template<merge_type mt, class T, int degree>
inline void
launch_split_solution(int mgrid_width, int step)
{
    dim3 block_grid = mblock_dim<mt, degree>::f(mgrid_width, step);
    const int required_threads = part_row_length<mt, degree>::f(step);
    dim3 thread_grid = prepare_thread_grid(required_threads, MAX_SHARED_ROW);

    split_solution<mt, degree, T><<<block_grid, thread_grid>>>(step);
    check_error("split_solution - " + mt, cudaGetLastError());
}

template<class T, int degree>
inline void
launch_final_backward_substitution(int m)
{
    dim3 block_grid(m / 2, m / 2);
    dim3 thread_grid((2 * degree + 1) * (2 * degree + 1) - 1);

    final_backward_substitution<degree, T><<<block_grid, thread_grid>>>();
    check_error("final_backward_substitution", cudaGetLastError());
}

template<class T, int degree>
inline void
launch_rewrite_result(int m)
{
    dim3 block_grid(m / 2, m / 2);
    dim3 thread_grid(degree + 1, degree + 1);

    rewrite_result<degree, T><<<block_grid, thread_grid>>>();
    check_error("rewrite_result", cudaGetLastError());
}

template<class T, int degree>
void
solve_last_equation(int n)
{
    int m = mgrid_width<degree>(n);
    int steps = lg(m / 2);
    //    print_local_matrices_and_rhs_v<T, degree>(n, steps - 1);
//    launch_final_elimination<T, degree>(steps);
//    launch_initial_substitution<T, degree>(steps);
    launch_magma_solver<T, degree>(steps);
//    print_local_matrices_and_rhs_v<T, degree>(n, steps - 1);
    launch_rewrite_initial_result<T, degree>(steps);
//    print_local_matrices_and_rhs_v<T, degree>(n, steps - 1);
    cudaThreadSynchronize();
}

template<class T, int degree>
void
backward_substitution(int n)
{
    int m = mgrid_width<degree>(n);
    int steps = lg(m / 2);

    for (int step = steps - 1; step >= 0; --step)
    {
        launch_clean_CD<_H, T, degree>(m, step);
//        print_local_matrices_and_rhs_v<T, degree>(n, step);
        launch_backward_substitutionB<_V, T, degree>(m, step);
        launch_backward_substitutionA<_V, T, degree>(m, step);
        launch_split_solution<_V, T, degree>(m, step);
//        print_assembled_v<T, degree>(n, step);
//        print_local_matrices_and_rhs_v<T, degree>(n, step);

        launch_clean_CD<_V, T, degree>(m, step);
//        print_local_matrices_and_rhs_h<T, degree>(n, step);
        launch_backward_substitutionB<_H, T, degree>(m, step);
        launch_backward_substitutionA<_H, T, degree>(m, step);
        launch_split_solution<_H, T, degree>(m, step);
//        print_assembled_h<T, degree>(n, step);
//        print_local_matrices_and_rhs_h<T, degree>(n, step);
    }

//    print_local_matrices_and_rhs_v<T, degree>(n, steps-1);
    launch_final_backward_substitution<T, degree>(m);
//    print_local_matrices_and_rhs_v<T, degree>(n, steps-1);
    launch_rewrite_result<T, degree>(m);

//    print_local_matrices_and_rhs_h<T, degree>(n, steps-1);

    prepare_result<degree, T>(n);

    cudaThreadSynchronize();
}

template<class T, int degree>
void
eliminate(int n)
{

//    print_local_matrices<T, degree>(n);
    launch_initial_merge<T, degree>(n);
//    print_local_matrices_and_rhs_after_first_merge<T, degree>(n);
//    print_assembled_0<T, degree>(n);

    int m = mgrid_width<degree>(n);
    int steps = lg(m / 2);

//    steps = 1;
    for (int step = 0; step < steps; ++step)
    {
//        std::cerr << "step " << step << "\n";

        launch_clean_AB<_H, T, degree>(m, step);
        launch_clean_CD<_H, T, degree>(m, step);
        launch_load_ABCD<_H, T, degree>(m, step);
        launch_calculate_Astar<_H, T, degree>(m, step);
        launch_calculate_Cstar<_H, T, degree>(m, step);
        launch_calculate_BDstar<_H, T, degree>(m, step);
//        print_assembled_h<T, degree>(n, step);
//        print_local_matrices_and_rhs_h<T, degree>(n, step);

        launch_clean_AB<_V, T, degree>(m, step);
        launch_clean_CD<_V, T, degree>(m, step);
        launch_load_ABCD<_V, T, degree>(m, step);
//        print_assembled_v<T, degree>(n, step);
//        print_local_matrices_and_rhs_v<T, degree>(n, step);
        launch_calculate_Astar<_V, T, degree>(m, step);
        launch_calculate_Cstar<_V, T, degree>(m, step);
        launch_calculate_BDstar<_V, T, degree>(m, step);
    }

    cudaThreadSynchronize();
}

template<class T, int degree>
void
print_result(int n, std::ostream &ostr)
{
    int size = n * n * QPC * QPC;

    p_2D<T> *tmp;
    p_2D<T> *p = new p_2D<T>[size];
    T *tmp_v;
    T *val = new T[size];

    gpuAssert(cudaMemcpyFromSymbol(&tmp, d_coords, sizeof (tmp)));
    gpuAssert(cudaMemcpy(p, tmp, sizeof(p_2D<T>) * size, cudaMemcpyDeviceToHost));

    gpuAssert(cudaMemcpyFromSymbol(&tmp_v, d_fun_vals, sizeof (tmp_v)));
    gpuAssert(cudaMemcpy(val, tmp_v, sizeof(T) * size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; ++i)
        ostr << p[i] << " " << val[i] << "\n";

    delete [] p;
    delete [] val;
}

template<class T, int degree>
T
calculate_error(int n)
{
    int N2 = n * n;
    T *tmp;
    T *result = new T[N2];

    gpuAssert(cudaMalloc(&tmp, sizeof (T) * N2));

    dim3 block_grid(n,n);
    dim3 thread_grid(QPC, QPC);

    calculate_norm<degree, T> <<<block_grid, thread_grid>>>(tmp);
    check_error("calculate_norm", cudaGetLastError());

    gpuAssert(cudaMemcpy(result, tmp, sizeof (T) * N2, cudaMemcpyDeviceToHost));

    T sum = 0;
    for (int i = 0; i < N2; ++i)
    {
        sum += result[i];
    }

    delete []result;

    gpuAssert(cudaFree(tmp));
    return sqrt(sum);
}

typedef void (*device_fun)(int);
// <editor-fold defaultstate="collapsed" desc="interface functions (float)">

template<>
void
CUDA_prepare_device<float>(int degree, int n)
{
    static device_fun prepares[] = {
        prepare_device<float, 1 >,
        prepare_device<float, 2 >,
        prepare_device<float, 3 >,
    };

    prepares[degree - 1](n);
}

template<>
void
CUDA_init_fronts<float>(int degree, int n)
{
    static device_fun initializers[] = {
        init_fronts<float, 1 >,
        init_fronts<float, 2 >,
        init_fronts<float, 3 >,
    };

    initializers[degree - 1](n);
}

template<>
void
CUDA_eliminate<float>(int degree, int n)
{
    static device_fun solvers[] = {
        eliminate<float, 1 >,
        eliminate<float, 2 >,
        eliminate<float, 3 >,
    };

    solvers[degree - 1](n);
}

template<>
void
CUDA_solve_last_equation<float>(int degree, int n)
{
    static device_fun solvers[] = {
        solve_last_equation<float, 1>,
        solve_last_equation<float, 2>,
        solve_last_equation<float, 3>,
    };

    solvers[degree - 1](n);
}

template<>
void
CUDA_backward_substitution<float>(int degree, int n)
{
    static device_fun backward_subs[] = {
        backward_substitution<float, 1>,
        backward_substitution<float, 2>,
        backward_substitution<float, 3>,
    };

    backward_subs[degree - 1](n);
}

template<>
float
CUDA_error<float>(int degree, int n)
{
    typedef float (*error_fun)(int);
    static error_fun calculators[] = {
        calculate_error<float, 1 >,
        calculate_error<float, 2 >,
        calculate_error<float, 3 >,
    };

    return calculators[degree - 1](n);
}

template<>
void
CUDA_print_result<float>(int degree, int n, std::ostream &ostr)
{
    typedef void (*print_fun)(int, std::ostream &);
    static print_fun printers[] = {
        print_result<float, 1 >,
        print_result<float, 2 >,
        print_result<float, 3 >,
    };

    printers[degree - 1](n, ostr);
}

template<>
void
CUDA_debug<float>(int degree, int n)
{
    typedef void (*print_fun)(int);
    static print_fun debuggers[] = {
        debug_device<float, 1 >,
        debug_device<float, 2 >,
        debug_device<float, 3 >,
    };

    debuggers[degree - 1](n);
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="interface functions (double)">

template<>
void
CUDA_prepare_device<double>(int degree, int n)
{
    static device_fun prepares[] = {
        prepare_device<double, 1 >,
        prepare_device<double, 2 >,
        prepare_device<double, 3 >,
    };

    prepares[degree - 1](n);
}

template<>
void
CUDA_init_fronts<double>(int degree, int n)
{
    static device_fun initializers[] = {
        init_fronts<double, 1 >,
        init_fronts<double, 2 >,
        init_fronts<double, 3 >,
    };

    initializers[degree - 1](n);
}

template<>
void
CUDA_eliminate<double>(int degree, int n)
{
    static device_fun elims[] = {
        eliminate<double, 1 >,
        eliminate<double, 2 >,
        eliminate<double, 3 >,
    };

    elims[degree - 1](n);
}

template<>
void
CUDA_solve_last_equation<double>(int degree, int n)
{
    static device_fun solvers[] = {
        solve_last_equation<double, 1>,
        solve_last_equation<double, 2>,
        solve_last_equation<double, 3>,
    };

    solvers[degree - 1](n);
}

template<>
void
CUDA_backward_substitution<double>(int degree, int n)
{
    static device_fun backward_subs[] = {
        backward_substitution<double, 1>,
        backward_substitution<double, 2>,
        backward_substitution<double, 3>,
    };

    backward_subs[degree - 1](n);
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
    };

    debuggers[degree - 1](n);
}
// </editor-fold>

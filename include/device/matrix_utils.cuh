/*
 * File:   matrix_utils.cuh
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 */

#ifndef MATRIX_UTILS_CUH
#define	MATRIX_UTILS_CUH

#include "utils.cuh"

/**
 * Converts x,y index (coordinates) to one dimensional index
 * @param x
 * @param y
 * @param n - maximum index in dimension x
 */
__device__ __host__
inline int
xy_to_idx(int x, int y, int n)
{
    return y * n + x;
}

/**
 * Converts x,y index (coordinates) to one dimensional index
 * @param p
 * @param n - maximum index in dimension x
 */
__device__ __host__
inline int
xy_to_idx(int_2 p, int n)
{
    return xy_to_idx(p.x, p.y, n);
}

/**
 * Converts one dimensional index to two dimensional index (x,y)
 * @param idx
 * @param n - maximum index in dimension x
 */
__device__ __host__
inline int_2
idx_to_xy(int idx, int n)
{
    return int_2(idx % n, idx / n);
}

/**
 * Converts one dimensional index to two dimensional index (x,y)
 * @param idx
 * @param n - maximum index in dimension x
 */
template<int n>
__device__ __host__
inline int_2
idx_to_xy(int idx)
{
    return int_2(idx % n, idx / n);
}

/**
 * Calculates place in d_B array where value of b(N_i,N_j) for provided element
 * is stored
 * @param element_xy 2D index of element
 * @param product_ij 2D index of product of basis functions i and j
 * @param _n total number of elements in one dimension
 */
template<int degree>
__device__ __host__
inline int
B_mem_idx(int_2 element_xy, int_2 product_ij, int _n)
{
    return ((element_xy.y * _n) + element_xy.x)
            *(degree+1)*(degree+1)*(degree+1)*(degree+1)
            + product_ij.y*(degree+1)*(degree+1)
            + product_ij.x;
}

/**
 * Calculates place in d_L array where value of l(v) for supplied basis function
 * and element is stored
 * @param element_xy 2D index of element
 * @param basis_idx index of basis function
 * @param _n total number of elements in one dimension
 */
template<int degree>
__device__ __host__
inline int
L_mem_idx(int_2 element_xy, int basis_idx, int _n)
{
    return ((element_xy.y * _n) + element_xy.x) * (degree+1)*(degree+1)
            + basis_idx;
}


/**
 * Clears matrix M(X,Y) assuming that thread block dimension is [X,1]
 */
template<class T, int X, int Y>
__device__
inline void
clear_matrix_and_rhs(T (&M)[X][Y], T (&RHS)[X])
{
    #pragma unroll
    for (int i = 0; i < Y; ++i) M[threadIdx.x+1][i] = 0;
    RHS[threadIdx.x+1] = 0;
    M[0][threadIdx.x+1] = 0;
    if(threadIdx.x == 0)
    {
        M[0][0] = 0;
        RHS[0] = 0;
    }
}


template<int degree, class T>
__device__
inline void
pivot_rows_cyclic(T B[][2 * degree + 1][2 * degree + 1],
                  T L[][2 * degree + 1])
{
    T tmp_B[2 * degree + 1];
    T tmp_L;

    for (int i = 0; i < 2 * degree + 1; ++i)
        tmp_B[i] = B[threadIdx.x][degree][i];
    for (int i = degree; i > 0; --i)
        for (int x = 0; x < 2 * degree + 1; ++x)
            B[threadIdx.x][i][x] = B[threadIdx.x][i - 1][x];
    for (int i = 0; i < 2 * degree + 1; ++i)
        B[threadIdx.x][0][i] = tmp_B[i];

    tmp_L = L[threadIdx.x][degree];
    for (int i = degree; i > 0; --i)
        L[threadIdx.x][i] = L[threadIdx.x][i - 1];
    L[threadIdx.x][0] = tmp_L;
}

template<int degree, class T>
__device__
inline void
pivot_columns_cyclic(T B[][2 * degree + 1][2 * degree + 1])
{
    T tmp_B[2 * degree + 1];
    for (int i = 0; i < 2 * degree + 1; ++i)
        tmp_B[i] = B[threadIdx.x][i][degree];
    for (int i = degree; i > 0; --i)
        for (int x = 0; x < 2 * degree + 1; ++x)
            B[threadIdx.x][x][i] = B[threadIdx.x][x][i - 1];
    for (int i = 0; i < 2 * degree + 1; ++i)
        B[threadIdx.x][i][0] = tmp_B[i];
}

/**
 * Calculates first row of the matrix and RHS for Gaussian elimination. It
 * divides each element in the row by the first element. This method assumes
 * that width of the matrix Y is equal to number of threads in block plus 1
 * (Y = blockDim.x + 1)
 * @param M
 * @param RHS
 */
template<class T, int X, int Y>
__device__
inline void
calculate_first_row(T (&M)[X][Y], T (&RHS)[X])
{
    T lead = M[0][0];
    M[0][threadIdx.x+1] /= lead;

    __syncthreads();

    if(threadIdx.x == 0) {
        M[0][0] /= lead;
        RHS[0] /= lead;
    }
}

template<class T, int X, int Y>
__device__
inline void
eliminate_first_row(T (&M)[X][Y], T (&RHS)[X])
{
    T lead = M[threadIdx.x+1][0];
    #pragma unroll
    for (int i = 1; i < Y; ++i)
    {
        M[threadIdx.x+1][i] -= lead * M[0][i];
    }
    RHS[threadIdx.x+1] -= lead * RHS[0];
}

template<class T>
__device__
inline void
fast_clean_vector(T* vector, int u_bound)
{
    int idx = xy_to_idx(blockIdx.x, blockIdx.y, gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < u_bound) vector[idx] = 0;
}

#endif	/* MATRIX_UTILS_CUH */


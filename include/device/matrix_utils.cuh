/* 
 * File:   matrix_utils.cuh
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 */

#ifndef MATRIX_UTILS_CUH
#define	MATRIX_UTILS_CUH

template<int degree, class T>
__device__
inline void
pivot_rows_cyclic(T B[][2 * degree + 1][2 * degree + 1])
{
    T tmp_B[2 * degree + 1];

    for (int i = 0; i < 2 * degree + 1; ++i)
        tmp_B[i] = B[threadIdx.x][degree][i];
    for (int i = degree; i > 0; --i)
        for (int x = 0; x < 2 * degree + 1; ++x)
            B[threadIdx.x][i][x] = B[threadIdx.x][i - 1][x];
    for (int i = 0; i < 2 * degree + 1; ++i)
        B[threadIdx.x][0][i] = tmp_B[i];
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

#endif	/* MATRIX_UTILS_CUH */


/*
 * File:   merge_utils.cuh
 * Author: Krzysztof Kuznik
 *
 * Created on July 20, 2011, 1:27 PM
 */

#ifndef MERGE_UTILS_CUH
#define	MERGE_UTILS_CUH

#include "utils.cuh"
#include <iostream>
#include <string>

__constant__ int mapping_matrix_1[4][4] = {
    {1,2,8,0},
    {2,3,0,4},
    {8,0,7,6},
    {0,4,6,5}
};

__constant__ int mapping_matrix_2[9][9] = {
    { 1,  2,  5,  3,  4,  6, 24, 23,  0},
    { 2,  5,  7,  4,  6,  9, 23,  0, 11},
    { 5,  7,  8,  6,  9, 10,  0, 11, 12},
    { 3,  4,  6, 24, 23,  0, 19, 20, 18},
    { 4,  6,  9, 23,  0, 11, 20, 18, 13},
    { 6,  9, 10,  0, 11, 12, 18, 13, 14},
    {24, 23,  0, 19, 20, 18, 21, 22, 17},
    {23,  0, 11, 20, 18, 13, 22, 17, 15},
    { 0, 11, 12, 18, 13, 14, 17, 15, 16}
};

__constant__ int mapping_matrix_3[16][16] = {
    { 1,  2,  3, 10,  4,  5,  6, 11,  7,  8,  9, 12, 48, 47, 46,  0},
    { 2,  3, 10, 13,  5,  6, 11, 16,  8,  9, 12, 19, 47, 46,  0, 22},
    { 3, 10, 13, 14,  6, 11, 16, 17,  9, 12, 19, 20, 46,  0, 22, 23},
    {10, 13, 14, 15, 11, 16, 17, 18, 12, 19, 20, 21,  0, 22, 23, 24},
    { 4,  5,  6, 11,  7,  8,  9, 12, 48, 47, 46,  0, 37, 38, 39, 36},
    { 5,  6, 11, 16,  8,  9, 12, 19, 47, 46,  0, 22, 38, 39, 36, 25},
    { 6, 11, 16, 17,  9, 12, 19, 20, 46,  0, 22, 23, 39, 36, 25, 26},
    {11, 16, 17, 18, 12, 19, 20, 21,  0, 22, 23, 24, 36, 25, 26, 27},
    { 7,  8,  9, 12, 48, 47, 46,  0, 37, 38, 39, 36, 40, 41, 42, 35},
    { 8,  9, 12, 19, 47, 46,  0, 22, 38, 39, 36, 25, 41, 42, 35, 28},
    { 9, 12, 19, 20, 46,  0, 22, 23, 39, 36, 25, 26, 42, 35, 28, 29},
    {12, 19, 20, 21,  0, 22, 23, 24, 36, 25, 26, 27, 35, 28, 29, 30},
    {48, 47, 46,  0, 37, 38, 39, 36, 40, 41, 42, 35, 43, 44, 45, 34},
    {47, 46,  0, 22, 38, 39, 36, 25, 41, 42, 35, 28, 44, 45, 34, 31},
    {46,  0, 22, 23, 39, 36, 25, 26, 42, 35, 28, 29, 45, 34, 31, 32},
    { 0, 22, 23, 24, 36, 25, 26, 27, 35, 28, 29, 30, 34, 31, 32, 33}
};

enum merge_type {
    _H,
    _V
};

inline
std::string
operator+(const char *str, merge_type mt)
{
    std::string result(str);
    if (mt == _H) {
        result += "Horizontal";
    } else if (mt == _V) {
        result += "Vertical";
    } else {
        result += "Unknown";
    }
    return result;
}

template<int degree>
inline int
mapping(int i, int j);

template<>
__device__
inline int
mapping<1>(int i, int j)
{
    return mapping_matrix_1[i][j];
}

template<>
__device__
inline int
mapping<2>(int i, int j)
{
    return mapping_matrix_2[i][j];
}

template<>
__device__
inline int
mapping<3>(int i, int j)
{
    return mapping_matrix_3[i][j];
}

/**
 * Calculates number of steps that must be performed to solve equation.
 * Each step consists of horizontal and vertical merge.
 * @param mgrid_width - width of multi-grid
 */
__device__ __host__
inline int
steps_count(int mgrid_width)
{
    return lg(mgrid_width / 2);
}

template<int degree>
__device__ __host__
inline int
mgrid_width(int grid_width)
{
    return 2 * grid_width / (degree + 1);
}

/**
 * Calculates row length for matrix corresponding to merge of multi-blocks
 * in provided step.
 */
template <merge_type mt, int degree>
struct row_length {
    static int f(int);
};

template <int degree>
struct row_length<_H, degree> {
    __device__ __host__
    static int
    f(int step) {
        // TODO this can be probably optimized (10*n/2-1) for p=1
        return (7 * (1 << (step + 2)) >> 2) * (degree * degree + degree) - degree * degree;
    }
};

template <int degree>
struct row_length<_V, degree> {
    __device__ __host__
    static int
    f(int step) {
        int off = 0;
        if (step == -1) off = 1 - degree;
        return off + (5 * (1 << (step + 2)) >> 1) * (degree * degree + degree) - degree * degree;
    }
};

/*
 * Number of rows eliminated in each multi-block
 * (for horizontal merge there are 2 sub-blocks in multi-block)
 * Example: multi-block for horizontal merge with 2 sub-blocks (top and
 * bottom). In this case merge_size is 4.
 *
 *          o-|-o-|-o            o-|-o-|-o
 *          |   |   |            |       |
 *          - 1 - 1 -            -   1   -
 *          |   |   |            |       |
 *          o-|-o-|-o    --->    o-|-o-|-o
 *          |   |   |            |       |
 *          - 2 - 2 -            -   2   -
 *          |   |   |            |       |
 *          o-|-o-|-o            o-|-o-|-o
*/

/**
 * Calculates number of rows eliminated in each multi-block in provided step
 */
template <merge_type mt, int degree>
struct eliminated {
    static int f(int);
};

template <int degree>
struct eliminated<_H, degree> {
    __device__ __host__
    static int
    f(int step) {
         return (((1 << (step + 2)) >> 1) * (degree * degree + degree) - 2 * degree * degree) >> 1;
    }
};

template <int degree>
struct eliminated<_V, degree> {
    __device__ __host__
    static int
    f(int step) {
        if(step == -1) return 1;
        return ((1 << (step + 2)) >> 1) * (degree * degree + degree) - degree * degree;
    }
};

/**
 * Calculates row length of each matrix that takes part in merging
 * of multi-blocks in provided step.
 */
template <merge_type mt, int degree>
struct part_row_length {
    static int f(int);
};

template <int degree>
struct part_row_length<_H, degree> {
    __device__ __host__
    static int
    f(int step) {
        return row_length<_V, degree>::f(step - 1)
                - eliminated<_V, degree>::f(step - 1);
    }
};

template <int degree>
struct part_row_length<_V, degree> {
    __device__ __host__
    static int
    f(int step) {
        return row_length<_H, degree>::f(step) - eliminated<_H, degree>::f(step);
    }
};

/**
 * Calculates row length of matrix that was result of previous merging.
 */
template <merge_type mt, int degree>
struct previous_merge_row_length {
    static int f(int);
};

template <int degree>
struct previous_merge_row_length<_H, degree> {
    __device__ __host__
    static int
    f(int step) {
        return row_length<_V, degree>::f(step - 1);
    }
};

template <int degree>
struct previous_merge_row_length<_V, degree> {
    __device__ __host__
    static int
    f(int step) {
        return row_length<_H, degree>::f(step);
    }
};

template<merge_type mt, int degree>
struct mblock_dim {
    static int_2 f(int, int);
};

template<int degree>
struct mblock_dim<_H, degree> {
    __device__ __host__
    static int_2
    f(int mgrid_width, int step)
    {
        const int x = mgrid_width / (1 << (step + 2));
        return int_2(x, 2 * x);
    }
};

template<int degree>
struct mblock_dim<_V, degree> {
    __device__ __host__
    static int_2
    f(int mgrid_width, int step)
    {
        const int x = mgrid_width / (1 << (step + 2));
        return int_2(x, x);
    }
};

/**
 * Calculates number of square multi-blocks in multi-grid in provided step.
 */
template<int degree>
__device__ __host__
inline int
square_mblock_count(int mgrid_width, int step)
{
    const int x = mgrid_width / (1 << (step + 2));
    return x * x;
}

template<merge_type mt>
struct merges_in_square {
    enum { val };
};

template<>
struct merges_in_square<_H> {
    enum { val=2 };
};

template<>
struct merges_in_square<_V> {
    enum { val=1 };
};

/**
 * Calculates length of array used for storage of assembled rows in provided
 * step for <mt> merge for multi-grid with edge of size mgrid_width.
 * This size includes RHS vector.
 */
template<merge_type mt, int degree>
__device__ __host__
inline int
assembled_size(int mgrid_width, int step)
{
    return square_mblock_count<degree>(mgrid_width, step)
            * eliminated<mt, degree>::f(step) * merges_in_square<mt>::val
            * (row_length<mt, degree>::f(step) + 1);
}

/**
* Calculates length of array used for <mt> merge for rows that are not
* eliminated in this step for multi-grid with edge of size mgrid_width.
* This size includes RHS vector.
*/
template<merge_type mt, int degree>
__device__ __host__
static int
for_merging_size(int mgrid_width, int step)
{
    const int rl = row_length<mt, degree>::f(step);
    const int el = eliminated<mt, degree>::f(step);
    return square_mblock_count<degree>(mgrid_width, step)
            * merges_in_square<mt>::val * (rl - el) * (rl + 1);
}

/**
 * This function calculates indexes of rows in left small matrix that are to be
 * merged.
 * @param sl width of small matrix
 * @return pair [min, max)
 */
template<int degree>
__device__ __host__
inline int_2
merged_segment_hl(int sl)
{
    return int_2(sl / 4 + degree * degree, sl / 2);
}

/**
 * This function calculates indexes of rows in right small matrix that are to be
 * merged.
 * @param sl width of small matrix
 * @return pair [min, max)
 */
template<int degree>
__device__ __host__
inline int_2
merged_segment_hr(int sl)
{
    return int_2(3 * sl / 4 + degree * degree, sl);
}

/**
 * This function calculates indexes of rows in top small matrix that are to be
 * merged.
 * @param sl width of small matrix
 * @return pair [min, max)
 */
template<int degree>
__device__ __host__
inline int_2
merged_segment_vt(int sl)
{
    return int_2(sl / 2 + degree * degree, 5 * sl / 6);
}

/**
 * This function calculates indexes of rows in bottom small matrix that are to be
 * merged.
 * @param sl width of small matrix
 * @return pair [min, max)
 */
template<int degree>
__device__ __host__
inline int_2
merged_segment_vb(int sl)
{
    return int_2(degree * degree, sl / 3);
}

template<merge_type mt, int segment_num, int degree>
struct merged_segment {
    static int_2 f(int);
};

template<int degree>
struct merged_segment<_H, 0, degree> {
    __device__ __host__
    static int_2
    f(int sl) {
        return merged_segment_hl<degree>(sl);
    }
};

template<int degree>
struct merged_segment<_H, 1, degree> {
    __device__ __host__
    static int_2
    f(int sl) {
        return merged_segment_hr<degree>(sl);
    }
};

template<int degree>
struct merged_segment<_V, 0, degree> {
    __device__ __host__
    static int_2
    f(int sl) {
        return merged_segment_vt<degree>(sl);
    }
};

template<int degree>
struct merged_segment<_V, 1, degree> {
    __device__ __host__
    static int_2
    f(int sl) {
        return merged_segment_vb<degree>(sl);
    }
};

/**
 * Small to big mapping of index for horizontal merge (multi-block left)
 * @param idx - index in small matrix
 * @param sl - row length in small matrix
 * @return index in big matrix
 */
template<int degree>
__device__ __host__
inline int
s2b_hl(int idx, int sl)
{
    const int p2 = degree * degree;
    const int t1 = (sl >> 2) + p2;
    const int t2 = sl >> 1;
    const int t3 = t2 + p2;
    const int x_len = t2 - t1;
    if (idx < t1) return idx + x_len;
    if (idx < t2) return idx - t1;
    if (idx < t3) return sl + x_len + (t3 - idx - 1);
    return sl + x_len + p2 + idx - t3; // RHS case included here
}

/**
 * Small to big mapping of index for horizontal merge (multi-block right)
 * @param idx - index in small matrix
 * @param sl - row length in small matrix
 * @return index in big matrix
 */
template<int degree>
__device__ __host__
inline int
s2b_hr(int idx, int sl)
{
    const int p2 = degree * degree;
    const int t1 = (sl >> 1) + (sl >> 2);
    const int t2 = t1 + p2;
    const int x_len = (sl >> 2) - p2;
    if (idx == sl) return 7 * x_len + 6 * p2;   // RHS case
    if (idx < t1) return x_len + (x_len + p2) + idx;
    if (idx < t2) return sl + x_len + (t2 - idx - 1);
    return sl - idx - 1;
}

/**
 * Small to big mapping of index for vertical merge (multi-block top)
 * @param idx - index in small matrix
 * @param sl - row length in small matrix
 * @return index in big matrix
 */
template<int degree>
__device__ __host__
inline int
s2b_vt(int idx, int sl)
{
    const int p2 = degree * degree;
    const int t1 = (sl>>1) + p2;
    const int t2 = 5 * sl / 6;
    const int t3 = t2 + p2;
    const int x_len = t2 - t1;
    if (idx < t1) return idx + x_len;
    if (idx < t2) return t2 - idx - 1;
    if (idx < t3) return x_len + sl + sl + p2 - idx - 1;
    return x_len + (sl << 1) - (t3 << 1) + (p2 << 1) + idx; // RHS case here
}

/**
 * Small to big mapping of index for vertical merge (multi-block bottom)
 * @param idx - index in small matrix
 * @param sl - row length in small matrix
 * @return index in big matrix
 */
template<int degree>
__device__ __host__
inline int
s2b_vb(int idx, int sl)
{
    const int p2 = degree * degree;
    const int t1 = p2;
    const int t2 = sl / 3;
    const int x_len = t2 - t1;
    if (idx == sl) return 5 * x_len + 4 * p2;   // RHS case
    if (idx < t1) return x_len + sl + (t2>>1) + (t1 - idx - 1);
    if (idx < t2) return idx - p2;
    return x_len + (sl>>1) + (idx - t2);
}

#endif	/* MERGE_UTILS_CUH */


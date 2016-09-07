/*
 * File:   utils.cuh
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 */

#ifndef UTILS_CUH
#define	UTILS_CUH

#include <cmath>

template<class T>
__device__
inline T
is_zero(const T &x);

template<>
__device__
inline float
is_zero<float>(const float &x)
{
    return fabsf(x) < FLT_EPSILON;
}

template<>
__device__
inline double
is_zero<double>(const double &x)
{
    return fabs(x) < DBL_EPSILON;
}

template <int degree>
__device__ __host__
inline int
basis_funs_cnt(int elements_cnt)
{
    return elements_cnt + degree;
}

template <int degree>
__device__ __host__
inline int
elements_cnt(int bf_cnt)
{
    return bf_cnt - degree;
}

template <int degree>
__device__ __host__
inline int
knots_cnt(int elements_cnt)
{
    return elements_cnt + 2 * degree + 1;
}

__device__ __host__
inline int
lg(int x)
{
    int i=0;
    while((x >>= 1) > 0) ++i;
    return i;
}

__device__ __host__
inline int
div_ceil(int a, int b)
{
    return (a / b) + (a % b > 0);
}

/**
 * @param tn - threads needed
 * @param tpb - threads per block
 * @return number of blocks on GPU
 */
inline int
calculate_blocks(int tn, int tpb)
{
    return div_ceil(tn, tpb);
}

inline dim3
calculate_blocks(dim3 threads_needed, dim3 max_threads_per_block)
{
    return dim3(calculate_blocks(threads_needed.x, max_threads_per_block.x),
                calculate_blocks(threads_needed.y, max_threads_per_block.y),
                calculate_blocks(threads_needed.z, max_threads_per_block.z));
}

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

template<int degree>
inline int
merges_cnt_for_step(int ne, int step)
{
    return ne / ((degree + 1) * (1 << step));
}

/**
 * @param tn - threads needed
 * @param tpb - threads per block
 * @return number of blocks on GPU
 */
inline int
prepare_block_grid(int tn, int tpb)
{
    return (tn / tpb) + (tn % tpb > 0);
}

template<int degree>
inline int
steps_cnt(int ne)
{
    return lg(ne / (degree + 1));
}

template<class T>
struct p_2D {
    T x, y;

    __device__ __host__
    p_2D(T x, T y) : x(x), y(y) { }

    __device__ __host__
    p_2D(T x) : x(x), y(x) { }

    __device__ __host__
    p_2D() {}

    __device__ __host__
    p_2D<T>
    operator+(T num) const
    {
        return p_2D<T>(x + num, y + num);
    }

    __device__ __host__
    p_2D<T>
    operator+(p_2D<T> p) const
    {
        return p_2D<T>(x + p.x, y + p.y);
    }

    __device__ __host__
    p_2D<T>
    operator-(p_2D<T> p) const
    {
        return p_2D<T>(x - p.x, y - p.y);
    }

    __device__ __host__
    p_2D<T>
    operator-(T num) const
    {
        return p_2D<T>(x - num, y - num);
    }

    __device__ __host__
    p_2D<T>
    operator*(T num) const
    {
        return p_2D<T>(x * num, y * num);
    }

    __device__ __host__
    p_2D<T>
    operator*(p_2D<T> p) const
    {
        return p_2D<T>(x * p.x, y * p.y);
    }

    __device__ __host__
    p_2D<T> &
    operator+=(p_2D<T>& a)
    {
        x += a.x;
        y += a.y;
        return *this;
    }

    __device__ __host__
    p_2D<T> &
    operator/=(T num)
    {
        x /= num;
        y /= num;
        return *this;
    }

    __device__ __host__
    friend
    p_2D<T>
    operator*(T num, const p_2D<T>& p)
    {
        return p*num;
    }

    __device__ __host__
    bool
    operator>=(T num) const
    {
        return x >= num || y >= num;
    }

    __device__ __host__
    bool
    operator==(const p_2D<T>& p) const
    {
        return x == p.x && y == p.y;
    }

    __device__ __host__
    operator dim3() const
    {
        return dim3(x, y, 1);
    }

    __device__ __host__
    T
    length() const
    {
        return y - x;
    }

        __device__ __host__
    T
    area() const
    {
        return x * y;
    }
};

typedef p_2D<int> int_2;

template<class T>
struct vertical_vec {
    T* p;
    int padding;

    __device__ __host__
    vertical_vec(T* p, int padding) : p(p), padding(padding) { }

    __device__ __host__
    T&
    operator[](int n)
    {
        return *(p + n * padding);
    }
};

template<class T>
struct array_2d {
    T* p;
    int row_length;

    __device__ __host__
    array_2d(T* p, int row_length) : p(p), row_length(row_length) { }

    __device__ __host__
    T*
    operator[](int n) const
    {
        return p + n * row_length;
    }
};

inline std::ostream&
operator<<(std::ostream &ostr, const dim3 &d)
{
    return ostr << d.x << " " << d.y << " " << d.z;
}

template<class T>
inline std::ostream&
operator<<(std::ostream &ostr, const p_2D<T> &d)
{
    return ostr << d.x << " " << d.y;
}

inline
std::ostream&
operator<<(std::ostream &ostr, const int_2 &d)
{
    return ostr << d.x << " " << d.y;
}

#endif	/* UTILS_CUH */


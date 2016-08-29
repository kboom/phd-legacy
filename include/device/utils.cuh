/* 
 * File:   utils.cuh
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 */

#ifndef UTILS_CUH
#define	UTILS_CUH

#include <cmath>
#include <cutil_inline.h>
#include <iostream>
#include <string>

// maximum number of threads in a block
#define MAX_TPB 1024

// maximum number of blocks in x dimension
#define MAX_GRID_X 65535

/**
 * Function for calculating logarithm with base 2 (only integer value).
 */
__device__ __host__
inline int
lg(int x)
{
    int i=0;
    while((x >>= 1) > 0) ++i;
    return i;
}

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
base_funs_cnt(int elements_cnt)
{
    return elements_cnt + degree;
}

template <int degree>
__device__ __host__
inline int
knots_cnt(int elements_cnt)
{
    return elements_cnt + 2 * degree + 1;
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
};

typedef p_2D<int> int_2;

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

/**
 * Given element coordinates and function coordinates this function evaluates
 * which part of basis function domain is located in provided element.
 * @param element_xy index of an element
 * @param fun_xy index of a basis function
 * @return 
 */
template <int degree, class T>
__device__ __host__
inline p_2D<T>
domain_part_xy(const p_2D<T> &element_xy, const p_2D<T> &fun_xy)
{
    return element_xy - fun_xy + degree;
}

void
check_error(const std::string &str, cudaError_t err_code)
{
    if (err_code != ::cudaSuccess)
        std::cerr << str << " -- " << cudaGetErrorString(err_code) << "\n";
}

std::ostream&
operator<<(std::ostream &ostr, const dim3 &d)
{
    return ostr << d.x << " " << d.y << " " << d.z;
}

template<class T>
std::ostream&
operator<<(std::ostream &ostr, const p_2D<T> &d)
{
    return ostr << d.x << " " << d.y;
}

std::ostream&
operator<<(std::ostream &ostr, const int_2 &d)
{
    return ostr << d.x << " " << d.y;
}

template<class T>
std::ostream&
operator<<(std::ostream &ostr, const array_2d<T> &array)
{
    for (int y = 0; y < array.row_length - 1; ++y)
    {
        for (int x = 0; x < array.row_length; ++x)
            ostr << array[y][x] << " ";
        ostr << ";\n";
    }
    return ostr;
}

inline dim3
prepare_thread_grid(int required_threads, int max_threads_x)
{
    return (required_threads < max_threads_x)
            ? dim3(required_threads)
            : dim3(max_threads_x);
}

inline void
create_configuration_for_num_threads(int num_threads,
                                     dim3 & block_grid,
                                     dim3 & thread_grid,
                                     int threads_per_block = MAX_TPB)
{
    thread_grid = prepare_thread_grid(num_threads, threads_per_block);
    int required_blocks = num_threads / thread_grid.x
                          + (num_threads % thread_grid.x > 0);
    block_grid.x = std::min(required_blocks, MAX_GRID_X);
    block_grid.y = (required_blocks / block_grid.x)
            + (required_blocks % block_grid.x > 0);
    block_grid.z = 1;
    
}
#endif	/* UTILS_CUH */


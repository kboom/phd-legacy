#ifndef BASE_FUNCTIONS_CUH
#define	BASE_FUNCTIONS_CUH

#include <config.h>

__device__ FEM_PRECISION *d_knot_vector;

/**
 * This function calculates length of knot vector interval starting in point
 * indexed <b>a</b> and ending in point indexed <b>b</b>
 */
template<class T>
__device__
inline T
interval(int a, int b)
{
    return d_knot_vector[b] - d_knot_vector[a];
}

/**
 * This function calculates length of knot vector interval starting in point
 * indexed <b>a</b> and ending in point indexed <b>a+1</b>
 */
template<class T>
__device__
inline T
interval(int a)
{
    return interval<T > (a, a + 1);
}

/**
 * Recursive calculation of value of B-spline <i>N_i</i> in point <b>x<b/>
 */
template<int degree, class T>
__device__
inline T
N(T x, int i)
{
    if (x <= d_knot_vector[i] || x > d_knot_vector[i + degree + 1]) return T(0);
    T h1 = d_knot_vector[i + degree] - d_knot_vector[i];
    T h2 = d_knot_vector[i + degree + 1] - d_knot_vector[i + 1];
    return (is_zero(h1) ? T(0) : (x - d_knot_vector[i]) * N < degree - 1 > (x, i) / h1)
            + (is_zero(h2) ? T(0) :
            (d_knot_vector[i + degree + 1] - x) * N < degree - 1 > (x, i + 1) / h2);
}

/**
 * Specialized template for double and degree = 0
 */
template<>
__device__
inline double
N < 0, double> (double x, int i)
{
    return (x <= d_knot_vector[i] || x > d_knot_vector[i + 1]) ? 0.0 : 1.0;
}

/**
 * Specialized template for float and degree = 0
 */
template<>
__device__
inline float
N < 0, float> (float x, int i)
{
    return (x <= d_knot_vector[i] || x > d_knot_vector[i + 1]) ? 0.0f : 1.0f;
}

/**
 * Recursive calculation of value of B-spline <i>N_i</i> derivative
 * in point <b>x<b/>
 */
template<int degree, class T>
__device__
inline T
dN(T x, int i)
{
    if (x <= d_knot_vector[i] || x > d_knot_vector[i + degree + 1]) return T(0);
    T h1 = d_knot_vector[i + degree] - d_knot_vector[i];
    T h2 = d_knot_vector[i + degree + 1] - d_knot_vector[i + 1];
    return (is_zero(h1) ?
            0
            :
            ((x - d_knot_vector[i]) * dN < degree - 1 > (x, i)
            + N < degree - 1 > (x, i)) / h1)

            + (is_zero(h2) ?
            0
            :
            ((d_knot_vector[i + degree + 1] - x)
            * dN < degree - 1 > (x, i + 1)
            - N < degree - 1 > (x, i + 1)) / h2);
}

/**
 * Specialized template for double and degree = 0
 */
template<>
__device__
inline double
dN < 0, double> (double, int)
{
    return 0.0;
}

/**
 * Specialized template for float and degree = 0
 */
template<>
__device__
inline float
dN < 0, float> (float, int)
{
    return 0.0f;
}


#endif	/* BASE_FUNCTIONS_CUH */


/* 
 * File:   base_functions.cuh
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 */

#ifndef BASE_FUNCTIONS_CUH
#define	BASE_FUNCTIONS_CUH

#ifndef FEM_PRECISION
#define FEM_PRECISION double
#endif

__device__ FEM_PRECISION *d_knot_vector;

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

template<>
__device__
inline double
N < 0, double> (double x, int i)
{
    return (x <= d_knot_vector[i] || x > d_knot_vector[i + 1]) ? 0.0 : 1.0;
}

template<>
__device__
inline float
N < 0, float> (float x, int i)
{
    return (x <= d_knot_vector[i] || x > d_knot_vector[i + 1]) ? 0.0f : 1.0f;
}

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

template<>
__device__
inline double
dN < 0, double> (double, int)
{
    return 0.0;
}

template<>
__device__
inline float
dN < 0, float> (float, int)
{
    return 0.0f;
}


#endif	/* BASE_FUNCTIONS_CUH */


/* 
 * File:   base_functions.h
 * Author: Krzysztof Kuźnik
 *
 * Created on February 13, 2011, 6:07 PM
 */

#ifndef BASE_FUNCTIONS_H
#define	BASE_FUNCTIONS_H

template<class T>
inline T
is_zero(const T &x)
{
    return std::abs(x) < std::numeric_limits<T>::epsilon();
}

template<class T, int degree>
inline T
N(T x, int i, T *t)
{
    if (x <= t[i] || x > t[i + degree + 1]) return T(0);
    T h1 = t[i + degree] - t[i];
    T h2 = t[i + degree + 1] - t[i + 1];
    return (is_zero(h1) ? 0 : (x - t[i]) * N<T, degree - 1 > (x, i, t) / h1)
            + (is_zero(h2) ? 0 :
            (t[i + degree + 1] - x) * N<T, degree - 1 > (x, i + 1, t) / h2);
}

template<>
inline double N<double, 0 > (double x, int i, double *t)
{
    if (x <= t[i] || x > t[i + 1]) return 0.0;
    return 1.0;
}

template<>
inline float N<float, 0 > (float x, int i, float *t)
{
    if (x <= t[i] || x > t[i + 1]) return 0.0f;
    return 1.0f;
}

template<class T, int degree>
inline T
dN(T x, int i, T *t)
{
    if (x <= t[i] || x > t[i + degree + 1]) return T(0);
    T h1 = t[i + degree] - t[i];
    T h2 = t[i + degree + 1] - t[i + 1];
    return (is_zero(h1) ?
            0
            :
            ((x - t[i]) * dN<T, degree - 1 > (x, i, t) / h1)
            + (N<T, degree - 1 > (x, i, t) / h1))

            + (is_zero(h2) ?
            0
            :
            ((t[i + degree + 1] - x)
            * dN<T, degree - 1 > (x, i + 1, t) / h2)
            + (N<T, degree - 1 > (x, i + 1, t) / -h2));
}

template<>
inline double dN<double, 0 > (double, int, double*)
{
    return 0.0;
}

template<>
inline float dN<float, 0 > (float, int, float*)
{
    return 0.0f;
}


#endif	/* BASE_FUNCTIONS_H */


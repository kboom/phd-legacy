/*
 * File:   quadratures.h
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 *
 * Created on February 13, 2011, 6:22 PM
 */

#ifndef QUADRATURES_H
#define	QUADRATURES_H

template <class T, int N>
T *
get_lobatto_points();

template <class T, int N>
T *
get_lobatto_weights();

template <class T, int N>
T *
get_gauss_legendre_points();

template <class T, int N>
T *
get_gauss_legendre_weights();

template<class T, class F>
T
integrate(F fun, T a, T b)
{
    if (a >= b) return 0;
    // now only Gauss-Legendre 4-point quadrature supported
    // TODO choose quadrature accordingly to spline degree
        static T points[4] = {-0.861136311594052,
                              -0.339981043584856,
                              0.339981043584856,
                              0.861136311594052};
        static T weights[4] = {0.347854845137454,
                               0.652145154862546,
                               0.652145154862546,
                               0.347854845137454};
//    static T points[7] = {0,
//                          0.4058451513773971669066064,
//                          -0.4058451513773971669066064,
//                          0.7415311855993944398638648,
//                          -0.7415311855993944398638648,
//                          0.9491079123427585245261897,
//                          -0.9491079123427585245261897};
//    static T weights[7] = {0.4179591836734693877551020,
//                           0.3818300505051189449503698,
//                           0.3818300505051189449503698,
//                           0.2797053914892766679014678,
//                           0.2797053914892766679014678,
//                           0.1294849661688696932706114,
//                           0.1294849661688696932706114};

    //     static T points[8] = {-0.1834346424956498049394761,
    //                          0.1834346424956498049394761,
    //                          -0.5255324099163289858177390,
    //                          0.5255324099163289858177390,
    //                          -0.7966664774136267395915539,
    //                          0.7966664774136267395915539,
    //                          -0.9602898564975362316835609,
    //                          0.9602898564975362316835609};
    //    static T weights[8] = {0.3626837833783619829651504,
    //                           0.3626837833783619829651504,
    //                           0.3137066458778872873379622,
    //                           0.3137066458778872873379622,
    //                           0.2223810344533744705443560,
    //                           0.2223810344533744705443560,
    //                           0.1012285362903762591525314,
    //                           0.1012285362903762591525314};

    T sum(0);
    T aa, bb;
    aa = (b - a) / 2.0;
    bb = (b + a) / 2.0;
    for (int i = 0; i < 4; ++i)
    {
        sum += weights[i] * fun(aa * points[i] + bb);
    }
    return aa*sum;
}


#endif	/* QUADRATURES_H */


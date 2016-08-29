/* 
 * File:   quadratures.h
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 *
 * Created on February 13, 2011, 6:22 PM
 */

#ifndef QUADRATURES_H
#define	QUADRATURES_H

template<class T, class F>
T
integrate(F fun, T a, T b)
{
    if (a >= b) return 0;
    static T points[4] = {-0.861136311594052,
                          -0.339981043584856,
                          0.339981043584856,
                          0.861136311594052};
    static T weights[4] = {0.347854845137454,
                           0.652145154862546,
                           0.652145154862546,
                           0.347854845137454};

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

template<class T, class F>
T
integrate_2d(F fun, T a1, T b1, T a2, T b2)
{
    if (a1 >= b1) return 0;
    if (a2 >= b2) return 0;
    static T points[4] = {-0.861136311594052,
                          -0.339981043584856,
                          0.339981043584856,
                          0.861136311594052};
    static T weights[4] = {0.347854845137454,
                           0.652145154862546,
                           0.652145154862546,
                           0.347854845137454};

    T sum(0);
    T aa1, bb1;
    T aa2, bb2;
    aa1 = (b1 - a1) / 2.0;
    bb1 = (b1 + a1) / 2.0;
    aa2 = (b2 - a2) / 2.0;
    bb2 = (b2 + a2) / 2.0;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            sum += weights[i] * weights[j]
                    * fun(aa1 * points[i] + bb1, aa2 * points[j] + bb2);
        }
    }
    return aa1 * aa2 * sum;
}

#endif	/* QUADRATURES_H */


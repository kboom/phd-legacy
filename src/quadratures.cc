#include <quadratures.h>

// <editor-fold defaultstate="collapsed" desc="Gauss-Legendre">

// <editor-fold defaultstate="collapsed" desc="2">

template<>
double *
get_gauss_legendre_points<double, 2 > ()
{
    static double points[] = {-0.5773502691896258, 0.5773502691896258};
    return points;
}

template<>
double *
get_gauss_legendre_weights<double, 2 > ()
{
    static double weights[] = {1, 1};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="3">
template<>
double *
get_gauss_legendre_points<double, 3 > ()
{
    static double points[] = {-0.774596669241483,
                              0,
                              0.774596669241483};
    return points;
}

template<>
double *
get_gauss_legendre_weights<double, 3 > ()
{
    static double weights[] = {0.555555555555556,
                               0.888888888888889,
                               0.555555555555556};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="4">

template<>
double *
get_gauss_legendre_points<double, 4 > ()
{
    static double points[] = {-0.861136311594052,
                              -0.339981043584856,
                              0.339981043584856,
                              0.861136311594052};
    return points;
}

template<>
double *
get_gauss_legendre_weights<double, 4 > ()
{
    static double weights[] = {0.347854845137454,
                               0.652145154862546,
                               0.652145154862546,
                               0.347854845137454};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="5">

template<>
double *
get_gauss_legendre_points<double, 5 > ()
{
    static double points[] = {-0.9061798459386640,
                              -0.5384693101056831,
                              0.0000000000000000,
                              0.5384693101056831,
                              0.9061798459386640};
    return points;
}

template<>
double *
get_gauss_legendre_weights<double, 5 > ()
{
    static double weights[] = {0.23692688505618909,
                               0.47862867049936647,
                               0.56888888888888889,
                               0.47862867049936647,
                               0.23692688505618909};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="6">

template<>
double *
get_gauss_legendre_points<double, 6> ()
{
    static double points[] = {-0.932469514203152,
                              -0.661209386466265,
                              -0.238619186083197,
                              0.238619186083197,
                              0.661209386466265,
                              0.932469514203152};
    return points;
}

template<>
double *
get_gauss_legendre_weights<double, 6> ()
{
    static double weights[] = {0.171324492379170,
                               0.360761573048139,
                               0.467913934572691,
                               0.467913934572691,
                               0.360761573048139,
                               0.171324492379170};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="7">

template<>
double *
get_gauss_legendre_points<double, 7 > ()
{
    static double points[] = {-0.949107912342759,
                              - 0.741531185599394,
                              - 0.405845151377397,
                              0,
                              0.405845151377397,
                              0.741531185599394,
                              0.949107912342759};
    return points;
}

template<>
double *
get_gauss_legendre_weights<double, 7 > ()
{
    static double weights[] = {0.129484966168870,
                               0.279705391489277,
                               0.381830050505119,
                               0.417959183673469,
                               0.381830050505119,
                               0.279705391489277,
                               0.129484966168870};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="8">
template<>
double *
get_gauss_legendre_points<double, 8 > ()
{
    static double points[] = {-0.960289856497536,
                              -0.796666477413627,
                              -0.525532409916329,
                              -0.183434642495650,
                              0.183434642495650,
                              0.525532409916329,
                              0.796666477413627,
                              0.960289856497536};
    return points;
}

template<>
double *
get_gauss_legendre_weights<double, 8 > ()
{
    static double weights[] = {0.101228536290376,
                               0.222381034453374,
                               0.313706645877887,
                               0.362683783378362,
                               0.362683783378362,
                               0.313706645877887,
                               0.222381034453374,
                               0.101228536290376};
    return weights;
}
// </editor-fold>

// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="Lobatto">

// <editor-fold defaultstate="collapsed" desc="2">

template<>
double *
get_lobatto_points<double, 2 > ()
{
    static double points[] = {-1.0000,
                              1.0000};
    return points;
}

template<>
double *
get_lobatto_weights<double, 2 > ()
{
    static double weights[] = {1.000,
                               1.000};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="3">

template<>
double *
get_lobatto_points<double, 3 > ()
{
    static double points[] = {-1.0000,
                              0,
                              1.0000};
    return points;
}

template<>
double *
get_lobatto_weights<double, 3 > ()
{
    static double weights[] = {0.333333333333333,
                               1.333333333333333,
                               0.333333333333333};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="4">

template<>
double *
get_lobatto_points<double, 4 > ()
{
    static double points[] = {-1.0000,
                              -0.447213595499958,
                              0.447213595499958,
                              1.0000};
    return points;
}

template<>
double *
get_lobatto_weights<double, 4 > ()
{
    static double weights[] = {0.166666666666667,
                               0.833333333333333,
                               0.833333333333333,
                               0.166666666666667};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="5">

template<>
double *
get_lobatto_points<double, 5 > ()
{
    static double points[] = {-1.0000,
                              -0.654653670707977,
                              0.0,
                              0.654653670707977,
                              1.0000};
    return points;
}

template<>
double *
get_lobatto_weights<double, 5 > ()
{
    static double weights[] = {0.1000,
                               0.544444444444444,
                               0.711111111111111,
                               0.544444444444444,
                               0.1000};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="6">

template<>
double *
get_lobatto_points<double, 6 > ()
{
    static double points[] = {-1.0000,
                              -0.765055323929465,
                              -0.285231516480645,
                              0.285231516480645,
                              0.765055323929465,
                              1.0000};
    return points;
}

template<>
double *
get_lobatto_weights<double, 6 > ()
{
    static double weights[] = {0.066666666666667,
                               0.378474956297847,
                               0.554858377035486,
                               0.554858377035486,
                               0.378474956297847,
                               0.066666666666667};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="7">

template<>
double *
get_lobatto_points<double, 7 > ()
{
    static double points[] = {-1.0000,
                              -0.830223896278567,
                              -0.468848793470714,
                              0.0,
                              0.468848793470714,
                              0.830223896278567,
                              1.0000};
    return points;
}

template<>
double *
get_lobatto_weights<double, 7 > ()
{
    static double weights[] = {0.047619047619048,
                               0.276826047361566,
                               0.431745381209863,
                               0.487619047619048,
                               0.431745381209863,
                               0.276826047361566,
                               0.047619047619048};
    return weights;
}
// </editor-fold>

// <editor-fold defaultstate="collapsed" desc="8">

template<>
double *
get_lobatto_points<double, 8 > ()
{
    static double points[] = {-1.0000,
                              -0.871740148509607,
                              -0.591700181433142,
                              -0.209299217902479,
                              0.209299217902479,
                              0.591700181433142,
                              0.871740148509607,
                              1.0000};
    return points;
}

template<>
double *
get_lobatto_weights<double, 8 > ()
{
    static double weights[] = {0.035714285714286,
                               0.210704227143506,
                               0.341122692483504,
                               0.412458794658704,
                               0.412458794658704,
                               0.341122692483504,
                               0.210704227143506,
                               0.035714285714286};
    return weights;
}
// </editor-fold>

// </editor-fold>

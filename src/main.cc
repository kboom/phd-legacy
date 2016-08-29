#include <config.h>
#include <benchmark.h>
#include <base_functions.h>

template<int degree, class T>
void
test_runner(int test_count = 20)
{
    for (int i = 0; i < test_count; ++i)
    {
        test_method_exponent<T, degree > (2, (degree + 1) * 2, 10000);
    }
}

template<int degree, class T>
void
test_once()
{
    test_runner<degree, T > (1);
}

int
main()
{
    typedef FEM_PRECISION precision;

    //        int test_count = 1;
    //
    //    test_runner < 1, precision > (test_count);
    //    test_runner < 2, precision > (test_count);
    //        test_runner < 3, precision > (test_count);
    //    test_runner < 4, precision > (test_count);
    //    test_runner < 5, precision > (test_count);
        
//    test_method_exponent<float, 3> (2, 4 * 2, 4096);
    test_method_exponent<precision, 1>(4, 512);
//    std::cout << "\n";
    test_method_exponent<precision, 2>(6, 2* 192);
//    std::cout << "\n";
    test_method_exponent<precision, 3>(8, 2 * 128);

//    run_method<precision, 1>(1, 8, true);
//    run_method<precision, 2>(2, 12, true);
//    run_method<precision, 3>(2, 128, true);
    //    run_method<double, 2 > (2, 3 * 4, false);


    return 0;
}
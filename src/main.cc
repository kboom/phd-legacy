#include <benchmark.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;

void print_result(const SingleRunTimeResults &srtr, ostream &ostr) {
    ostr << boost::format("%15s\t") % srtr.initialization_us;
    ostr << boost::format("%15s\t") % srtr.factorization_us;
    ostr << boost::format("%15s\t") % srtr.solution_us;
    ostr << boost::format("%15s\t%15s\t%15e\n")
            % srtr.error_calculation_us
            % srtr.total_us
            % srtr.error;
}

void print_result_for_Ne(int Ne, const vector<SingleRunTimeResults> &result, ostream &ostr) {
    typedef vector<SingleRunTimeResults>::const_iterator TRit;

    ostr << boost::format("%8d\t") % Ne;
    for (TRit it = result.begin() ; it != result.end() ; ++it) {
        print_result(*it, ostr);
    }
}

void find_min_for_Ne(int Ne, const vector<SingleRunTimeResults> &result, ostream &ostr) {
    typedef vector<SingleRunTimeResults>::const_iterator TRit;

    ostr << boost::format("%8d\t") % Ne;
    SingleRunTimeResults min = *result.begin();
    for (TRit it = result.begin() ; it != result.end() ; ++it) {
        const SingleRunTimeResults &tmp = *it;
        min.initialization_us = ::std::min(tmp.initialization_us, min.initialization_us);
        min.factorization_us = ::std::min(tmp.factorization_us, min.factorization_us);
        min.solution_us = ::std::min(tmp.solution_us, min.solution_us);
        min.error_calculation_us = ::std::min(tmp.error_calculation_us, min.error_calculation_us);
    }
    print_result(min, ostr);
}

void print_results(const TestResult &results, ostream &ostr = cout) {
    typedef TestResult::const_iterator TRit;

    for (TRit it = results.begin() ; it != results.end() ; ++it) {
        find_min_for_Ne(it->first, it->second, ostr);
//        ostr << "-----\n";
    }
}

template<int degree, class T>
void
test_runner(int test_count = 20)
{
    Tester<T, degree> tester("CUDA");
    tester.set_benchmark(1);
    tester.set_rhs_cnt(1);
    tester.set_reps(test_count);

    print_results(tester.test_method_exponent((degree + 1) * 2, 20000));
}

template<int degree, class T>
void
single_run(int intervals, int rhs_cnt = 1)
{
    Tester<T, degree> tester("CUDA");
    tester.set_benchmark(1);
    tester.set_rhs_cnt(rhs_cnt);

    tester.run_method(intervals, false);
}

int
main()
{
    typedef double precision;

    // int test_count = 1;

    // test_runner < 1, precision > (test_count);
    // test_runner < 2, precision > (test_count);
    // test_runner < 3, precision > (test_count);
    // test_runner < 4, precision > (test_count);
    // test_runner < 5, precision > (test_count);

    // Maciek 10.08
    single_run<2, precision> (128, 128);
    single_run<2, precision> (256, 256);
    single_run<2, precision> (512, 512);
    single_run<2, precision> (1024, 1024);

    //    test_method_exponent<float, 3> (2, 4 * 2, 4096);
    //            test_method_exponent<float, 1 > (2, 4, 4096);

//     single_run<4, precision>(5 * 16);
//      run_method<precision, 1 > (1, 2 * 4, false);
//    run_method<precision, 3 > (2,  4 * 2048, false);
//    run_method<precision, 3 > (2, 1024, false);
//        run_method<double, 2 > (2, 3 * 4, false);


    return 0;
}
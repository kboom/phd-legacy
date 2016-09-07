
#include <benchmark.h>

using namespace benchmark_functions;
using namespace std;

template<>
benchmark<double>&
get_benchmark<double>(int i) {
    static benchmark<double> benchmarks[] = {
        benchmark<double>(benchmark_0::u,
                benchmark_0::a,
                benchmark_0::b,
                benchmark_0::c,
                benchmark_0::f,
                benchmark_0::beta,
                benchmark_0::gamma),
        benchmark<double>(benchmark_1::u,
                benchmark_1::a,
                benchmark_1::b,
                benchmark_1::c,
                benchmark_1::f,
                benchmark_1::beta,
                benchmark_1::gamma),
        benchmark<double>(benchmark_2::u,
                benchmark_2::a,
                benchmark_2::b,
                benchmark_2::c,
                benchmark_2::f,
                benchmark_2::beta,
                benchmark_2::gamma),
        benchmark<double>(benchmark_3::u,
                benchmark_3::a,
                benchmark_3::b,
                benchmark_3::c,
                benchmark_3::f,
                benchmark_3::beta,
                benchmark_3::gamma),
        benchmark<double>(benchmark_4::u,
                benchmark_4::a,
                benchmark_4::b,
                benchmark_4::c,
                benchmark_4::f,
                benchmark_4::beta,
                benchmark_4::gamma)
    };
    return benchmarks[i];
 }

template<>
benchmark<float>&
get_benchmark<float>(int i) {
    static benchmark<float> benchmarks[] = {
        benchmark<float>(benchmark_0::u,
                benchmark_0::a,
                benchmark_0::b,
                benchmark_0::c,
                benchmark_0::f,
                benchmark_0::beta,
                benchmark_0::gamma),
        benchmark<float>(benchmark_1::u,
                benchmark_1::a,
                benchmark_1::b,
                benchmark_1::c,
                benchmark_1::f,
                benchmark_1::beta,
                benchmark_1::gamma),
        benchmark<float>(benchmark_2::u,
                benchmark_2::a,
                benchmark_2::b,
                benchmark_2::c,
                benchmark_2::f,
                benchmark_2::beta,
                benchmark_2::gamma),
        benchmark<float>(benchmark_3::u,
                benchmark_3::a,
                benchmark_3::b,
                benchmark_3::c,
                benchmark_3::f,
                benchmark_3::beta,
                benchmark_3::gamma),
        benchmark<float>(benchmark_4::u,
                benchmark_4::a,
                benchmark_4::b,
                benchmark_4::c,
                benchmark_4::f,
                benchmark_4::beta,
                benchmark_4::gamma)
    };
    return benchmarks[i];
 }

string
duration2string(const boost::posix_time::time_duration &t)
{
    std::ostringstream os;
    int seconds = t.minutes() * 60 + t.seconds();
    os << boost::format("%d.%06d") % seconds % t.fractional_seconds();
    return os.str();
}
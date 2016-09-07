/*
 * File:   benchmark.h
 * Author: Krzysztof Kuźnik <kmkuznik@gmail.com>
 */

#ifndef BENCHMARK_H
#define	BENCHMARK_H

#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <solver_factory.h>

namespace benchmark_functions {

    template<class T, class DADU, class B, class DU, class C, class U>
    inline T
    f(T x, DADU dadu, B b, DU du, C c, U u)
    {
        return -(dadu(x)) + b(x) * du(x) + c(x) * u(x);
    }

    namespace benchmark_0 {

        template <class T>
        T
        u(T x)
        {
            return x;
        }

        template <class T>
        T
        a(T)
        {
            return 1;
        }

        template <class T>
        T
        b(T)
        {
            return 0;
        }

        template <class T>
        T
        c(T)
        {
            return 0;
        }

        template <class T>
        T
        du(T)
        {
            return 1;
        }

        template <class T>
        T
        adu(T x)
        {
            return a(x) * du(x);
        }

        template <class T>
        T
        dadu(T)
        {
            return 0;
        }

        template <class T>
        T
        beta(T)
        {
            return 0;
        }

        template <class T>
        T
        gamma(T x)
        {
            return adu(x) + beta(x) * u(x);
        }

        template <class T>
        T
        f(T x)
        {
            return benchmark_functions::f(x, dadu<T>, b<T>, du<T>, c<T>, u<T>);
        }

    }

    namespace benchmark_1 {

        template <class T>
        T
        u(T x)
        {
            return sin(15 * x) * cos(24 * x) * x;
        }

        template <class T>
        T
        a(T x)
        {
            return sin(x);
        }

        template <class T>
        T
        b(T x)
        {
            return x;
        }

        template <class T>
        T
        c(T x)
        {
            return -x;
        }

        template <class T>
        T
        du(T x)
        {
            return (-sin(9 * x)
                    + sin(39 * x)
                    - 9 * x * cos(9 * x)
                    + 39 * x * cos(39 * x)) / 2.0;
        }

        template <class T>
        T
        adu(T x)
        {
            return a(x) * du(x);
        }

        template <class T>
        T
        dadu(T x)
        {
            return (cos(x)*(-sin(9 * x)
                    + sin(39 * x)
                    - 9 * x * cos(9 * x)
                    + 39 * x * cos(39 * x))
                    + 3 * sin(x)*(27 * x * sin(9 * x)
                    - 507 * x * sin(39 * x)
                    - 6 * cos(9 * x)
                    + 26 * cos(39 * x))) / 2.0;
        }

        template <class T>
        T
        beta(T)
        {
            return 1;
        }

        template <class T>
        T
        gamma(T x)
        {
            return adu(x) + beta(x) * u(x);
        }

        template <class T>
        T
        f(T x)
        {
            return benchmark_functions::f(x, dadu<T>, b<T>, du<T>, c<T>, u<T>);
        }

    }

    namespace benchmark_2 {

        template <class T>
        T
        u(T x)
        {
            return 4 * x * x;
        }

        template <class T>
        T
        a(T)
        {
            return 1;
        }

        template <class T>
        T
        b(T)
        {
            return 1;
        }

        template <class T>
        T
        c(T)
        {
            return 1;
        }

        template <class T>
        T
        du(T x)
        {
            return 8 * x;
        }

        template <class T>
        T
        adu(T x)
        {
            return a(x) * du(x);
        }

        template <class T>
        T
        dadu(T)
        {
            return 8;
        }

        template <class T>
        T
        beta(T)
        {
            return 1;
        }

        template <class T>
        T
        gamma(T x)
        {
            return adu(x) + beta(x) * u(x);
        }

        template <class T>
        T
        f(T x)
        {
            return benchmark_functions::f(x, dadu<T>, b<T>, du<T>, c<T>, u<T>);
        }

    }

    namespace benchmark_3 {

        template <class T>
        T
        u(T x)
        {
            return (x * x * x) / 6.0 + x / 2.0;
        }

        template <class T>
        T
        a(T)
        {
            return 1;
        }

        template <class T>
        T
        b(T)
        {
            return 0;
        }

        template <class T>
        T
        c(T)
        {
            return 0;
        }

        template <class T>
        T
        du(T x)
        {
            return (x * x) / 2.0 + 0.5;
        }

        template <class T>
        T
        adu(T x)
        {
            return a(x) * du(x);
        }

        template <class T>
        T
        dadu(T x)
        {
            return x;
        }

        template <class T>
        T
        beta(T)
        {
            return 0;
        }

        template <class T>
        T
        gamma(T x)
        {
            return adu(x) + beta(x) * u(x);
        }

        template <class T>
        T
        f(T x)
        {
            return benchmark_functions::f(x, dadu<T>, b<T>, du<T>, c<T>, u<T>);
        }

    }

    namespace benchmark_4 {

        template <class T>
        T
        u(T)
        {
            return 0;
        }

        template <class T>
        T
        a(T)
        {
            return -1.0;
        }

        template <class T>
        T
        b(T)
        {
            return 0.5;
        }

        template <class T>
        T
        c(T)
        {
            return 0;
        }

        template <class T>
        T
        du(T)
        {
            return 0;
        }

        template <class T>
        T
        adu(T x)
        {
            return a(x) * du(x);
        }

        template <class T>
        T
        dadu(T)
        {
            return 0;
        }

        template <class T>
        T
        beta(T)
        {
            return 0;
        }

        template <class T>
        T
        gamma(T x)
        {
            return adu(x) + beta(x) * u(x);
        }

        template <class T>
        T
        f(T x)
        {
            return benchmark_functions::f(x, dadu<T>, b<T>, du<T>, c<T>, u<T>);
        }

    }

}

std::string
duration2string(const boost::posix_time::time_duration &t);

template<class T>
struct benchmark {
    T(*u)(T);
    T(*a)(T);
    T(*b)(T);
    T(*c)(T);
    T(*f)(T);
    T(*beta)(T);
    T(*gamma)(T);

    benchmark(T(*_u)(T), T(*_a)(T), T(*_b)(T), T(*_c)(T), T(*_f)(T),
              T(*_beta)(T), T(*_gamma)(T))
    : u(_u),
    a(_a),
    b(_b),
    c(_c),
    f(_f),
    beta(_beta),
    gamma(_gamma) { }
};

template<class T>
benchmark<T>& get_benchmark(int i);

struct SingleRunTimeResults {
    double initialization_us;
    double factorization_us;
    double solution_us;
    double error_calculation_us;
    double total_us;
    double error;
};

typedef std::map<int, std::vector<SingleRunTimeResults> > TestResult;

template <class T, int D>
class Tester {
    std::string solver_name_;
    benchmark<T> benchmark_;
    unsigned rhs_cnt_;
    unsigned reps_;

    SingleRunTimeResults
    single_run(int elements);

public:
    Tester(const std::string solver_name = "MUMPS") :
        solver_name_(solver_name),
        benchmark_(get_benchmark<T>(0)),
        rhs_cnt_(1),
        reps_(1) { }

    std::vector<SingleRunTimeResults>
    test_method(int elements);

    TestResult
    test_method_exponent(int start, int stop);

    void run_method(int intervals,
                    bool print_error = false,
                    std::ostream &ostr = std::cout);

    void set_benchmark(const benchmark<T> &new_benchmark) {
        benchmark_ = new_benchmark;
    }

    void set_benchmark(int i) {
        set_benchmark(get_benchmark<T>(i));
    }

    void set_solver(const std::string &name) {
        solver_name_ = name;
    }

    void set_rhs_cnt(unsigned rhs_cnt) {
        rhs_cnt_ = rhs_cnt;
    }

    void set_reps(unsigned reps) {
        reps_ = reps;
    }
};

template<class T, int D>
SingleRunTimeResults
Tester<T, D>::
single_run(int elements)
{
    bspline_fem_solver<T>* solver;
    boost::posix_time::ptime t1, t2;
    boost::posix_time::ptime total1, total2;
    SingleRunTimeResults result;

    solver = create_solver<T, D > (solver_name_, elements, rhs_cnt_);

    total1 = boost::posix_time::microsec_clock::local_time();

    t1 = boost::posix_time::microsec_clock::local_time();
    solver->init(benchmark_.a, benchmark_.b, benchmark_.c, benchmark_.f,
                 benchmark_.beta(1.0), benchmark_.gamma(1.0));
    t2 = boost::posix_time::microsec_clock::local_time();

    result.initialization_us = (t2 - t1).total_microseconds();

    t1 = boost::posix_time::microsec_clock::local_time();
    solver->factorize_matrix();
    t2 = boost::posix_time::microsec_clock::local_time();
    result.factorization_us = (t2 - t1).total_microseconds();

    t1 = boost::posix_time::microsec_clock::local_time();
    solver->solve();
    t2 = boost::posix_time::microsec_clock::local_time();

    result.solution_us = (t2 - t1).total_microseconds();

    t1 = boost::posix_time::microsec_clock::local_time();
    T err = solver->error(benchmark_.u);
    t2 = boost::posix_time::microsec_clock::local_time();

    total2 = boost::posix_time::microsec_clock::local_time();

    result.error_calculation_us = (t2 - t1).total_microseconds();
    result.total_us = (total2 - total1).total_microseconds();
    result.error = err;

    delete solver;

    return result;
}

template<class T, int D>
std::vector<SingleRunTimeResults>
Tester<T, D>::
test_method(int elements)
{
    std::vector<SingleRunTimeResults> results;
    for (unsigned i = 0 ; i < reps_ ; ++i)
        results.push_back(single_run(elements));

    return results;
}

template<class T, int D>
TestResult
Tester<T, D>::
test_method_exponent(int start, int stop)
{
    std::vector<int> intervals;
    TestResult result;

    for (int i = start; i <= stop; i *= 2)
        intervals.push_back(i);
    for (unsigned i=0 ; i < intervals.size() ; ++i) {
        std::vector<SingleRunTimeResults> tmp =
                test_method(intervals[i]);
        result[intervals[i]] = tmp;
    }

    return result;
}

template<class T, int D>
void
Tester<T, D>::
run_method(int intervals,
           bool print_error,
           std::ostream &ostr)
{
    bspline_fem_solver<T> *solver =
            create_solver<T, D>(solver_name_, intervals, 1);

    solver->init(benchmark_.a, benchmark_.b, benchmark_.c, benchmark_.f,
                 benchmark_.beta(1.0), benchmark_.gamma(1.0));
    solver->factorize_matrix();
    solver->solve();
    solver->print_result(ostr);

    if (print_error)
    {
        T err = solver->error(benchmark_.u);
        std::cerr << boost::format("%15e\n") % err;
    }

    delete solver;
}

#endif	/* BENCHMARK_H */


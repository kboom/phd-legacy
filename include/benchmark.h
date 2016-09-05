#ifndef BENCHMARK_H
#define	BENCHMARK_H

#include <cmath>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <solver_factory.h>

inline
std::string
duration2string(const boost::posix_time::time_duration &t)
{
    std::ostringstream os;
    int seconds = t.minutes() * 60 + t.seconds();
    os << boost::format("%d.%06d") % seconds % t.fractional_seconds();
    return os.str();
}

template<class T, int D>
void
test_method(int start, int stop, int step = 1,
            std::ostream &ostr = std::cout)
{
    bspline_fem_solver<T>* solver;
    boost::posix_time::ptime t1, t2;
    boost::posix_time::ptime total1, total2;

    for (int intervals = start; intervals <= stop; intervals += step)
    {
        solver = create_solver< T, D > ("CUDA", intervals);

        ostr << boost::format("%8d\t") % intervals;

        total1 = boost::posix_time::microsec_clock::local_time();

        t1 = boost::posix_time::microsec_clock::local_time();
        solver->init();
        t2 = boost::posix_time::microsec_clock::local_time();

        ostr << boost::format("%15s\t") % duration2string(t2 - t1);

        t1 = boost::posix_time::microsec_clock::local_time();
        solver->eliminate();
        t2 = boost::posix_time::microsec_clock::local_time();
        ostr << boost::format("%15s\t") % duration2string(t2 - t1);

        t1 = boost::posix_time::microsec_clock::local_time();
        solver->solve_last_equation();
        t2 = boost::posix_time::microsec_clock::local_time();
        ostr << boost::format("%15s\t") % duration2string(t2 - t1);

        t1 = boost::posix_time::microsec_clock::local_time();
        solver->backward_substitution();
        t2 = boost::posix_time::microsec_clock::local_time();
        ostr << boost::format("%15s\t") % duration2string(t2 - t1);

        t1 = boost::posix_time::microsec_clock::local_time();
        T err = solver->error();

        t2 = boost::posix_time::microsec_clock::local_time();

        total2 = boost::posix_time::microsec_clock::local_time();

        ostr << boost::format("%15s\t%15s\t%15e\n")
                % duration2string(t2 - t1)
                % duration2string(total2 - total1)
                % err;

        delete solver;
    }
}

template<class T, int D>
void
test_method(int benchmark_num,
            int start,
            int stop,
            int step = 1,
            std::ostream &ostr = std::cout)
{
    test_method < T, D > (start, stop, step, ostr);
}

template<class T, int D>
void
test_method_exponent(int start,
                     int stop,
                     std::ostream &ostr = std::cout)
{
    std::vector<int> intervals;
    for (int i = start; i <= stop; i *= 2)
        intervals.push_back(i);

    for (std::vector<int>::iterator it = intervals.begin();
         it != intervals.end(); ++it)
        test_method<T, D > (*it, *it, 1, ostr);
}

template<class T, int D>
void
run_method(int intervals,
           bool print_error = false,
           std::ostream &ostr = std::cout)
{
    bspline_fem_solver<T>* solver;

    solver = create_solver< T, D > ("CUDA", intervals);

    solver->init();

    solver->eliminate();
    solver->solve_last_equation();
    solver->backward_substitution();

    solver->print_result(ostr);

    if (print_error)
    {
        T err = solver->error();
        std::cerr << boost::format("%15e\n") % err;
    }

    delete solver;
}

#endif	/* BENCHMARK_H */


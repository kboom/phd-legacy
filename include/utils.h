/* 
 * File:   print_helpers.h
 * Author: Krzysztof Kuznik <kmkuznik at gmail.com>
 *
 * Created on July 17, 2011, 10:53 AM
 */

#ifndef UTILS_H
#define	UTILS_H

#include <iostream>

template<class T>
T**
allocate_square_matrix(int n) {
    T **B = new T*[n];
    for(int i = 0 ; i < n ; ++i)
    {
        B[i] = new T[n];
    }
    
    return B;
}

template<class T>
void
free_square_matrix(T** B, int n) {
    for(int i = 0 ; i < n ; ++i)
        delete[] B[i];
    delete[] B;
}

template<class T>
void
zeroize_square_matrix(T** B, int n) {
    for(int i = 0 ; i < n ; ++i)
        for(int j = 0 ; j < n ; ++j)
            B[i][j] = 0;
}


template<class T>
void
print_vector_vertical(T* vec, int size, std::ostream &ostr = std::cout)
{
    for(int i= 0 ; i < size ; ++i)
        ostr << vec[i] << " ;\n";
}

template<class T>
void
print_vector_horizontal(T* vec, int size, std::ostream &ostr = std::cout)
{
    for(int i= 0 ; i < size ; ++i)
        ostr << vec[i] << " ";
}

template<class T>
void
print_square_matrix(T** B, int n, std::ostream &ostr = std::cout)
{
    for(int i = 0 ; i < n ; ++i)
    {
        print_vector_horizontal(B[i], n, ostr);
        ostr << ";\n";
    }
}

#endif	/* UTILS_H */


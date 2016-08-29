/* 
 * File:   CUDA_interface.h
 * Author: Krzysztof Kuźnik <kmkuznik at gmail.com>
 */

#ifndef CUDA_INTERFACE_H
#define	CUDA_INTERFACE_H

#include <iostream>

template<class T>
void CUDA_prepare_device(int degree, int n);

template<class T>
void CUDA_init_fronts(int degree, int n);

template<class T>
void CUDA_eliminate(int degree, int n);

template<class T>
void CUDA_solve_last_equation(int degree, int n);

template<class T>
void CUDA_backward_substitution(int degree, int n);

template<class T>
T CUDA_error(int error, int n);

template<class T>
void CUDA_print_result(int error, int n, std::ostream &ostr);

template<class T>
void CUDA_debug(int degree, int n);

void cleanup_device();
#endif	/* CUDA_INTERFACE_H */


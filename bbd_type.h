/**
author: dztu
date: Sep 30, 2022
version: v1.0
**/

#ifndef _BBD_TYPE_H_
#define _BBD_TYPE_H_

#define int_bbd int
#define float_bbd double


#define SOLVER_TYPE_NONE    0
#define SOLVER_TYPE_CHOLMOD 1
#define SOLVER_TYPE_KLU     2
#define SOLVER_TYPE_NICSLU  3
#define SOLVER_TYPE_PARDISO 4
#define SOLVER_TYPE_BBD_NICSLU 5
#define SOLVER_TYPE_BBD_KLU    6
#define SOLVER_TYPE_CHOLMOD_LITE 7   // without reordering
#define SOLVER_TYPE_PARALLEL_BBD_KLU 8


#define CSPARSE_BLAS 0
#define GRAPH_BLAS 1
#define COMB_BLAS 2
#define PARDISO_BLAS 3
#define MKL_BLAS 4

#endif
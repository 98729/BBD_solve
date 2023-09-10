#ifndef _MPI_FUNC_H_
#define _MPI_FUNC_H_


#include "sparse_matrix.h"
#include "bbd_type.h"
#include "bbd_solver.h"
#include "/opt/intel/oneapi/mpi/2021.7.0/include/mpi.h"

void gather_schur_rhs(
    float_bbd *schur_rhs_sum,
    float_bbd *schur_rhs_arr_single,
    int_bbd schur_size);
void gather_schur_mtx(
    std::vector<SparseMatrix *> &schur_mtx_arr,
    SparseMatrix *schur_mtx_arr_single,
    int comm_rank, int comm_size);
void gather_xi(
    float_bbd *xi_arr,
    float_bbd *xi_arr_single, int *rhs_recvCount, int *displs,
    int sendnum);




#endif
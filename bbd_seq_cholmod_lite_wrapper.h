
#ifndef _BBD_SEQ_CHOLMOD_SOLVER_H_
#define _BBD_SEQ_CHOLMOD_SOLVER_H_

#include "cholmod_lite.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"
#include "./utils/Timer.h"
#include "bbd_type.h"
#include "bbd_schur.h"
#include "./utils/metis_api.h"
#include "./utils/cmd_line_parser.h"
#include "./utils/read_matrix.h"
#include "bbd_solver.h"
#include "./utils/bbd_blas.h"
#include "./utils/result_saver.h"
#include "./extlibs/gmres/gmres.h"


void bbd_seq_cholmod_lite_solve(SparseMatrix *A_org, SparseMatrix *Ch_matrix, SparseMatrix *BU_matrix, 
                    std::vector<int_bbd> &node_index, std::vector<float_bbd> &time_step, Params *params) {

    printf("==============Using Sequential BBD Solver with CHOLMOD_LITE==============\n");

    printf("==============Metis Partition==============\n");
    Timer2 tmp_timer;
    tmp_timer.start();

    int_bbd n_parts;  // number of partition, if not set, we will use the best partiton according to n_proc, matrix size and sparsity
    n_parts = (params->n_parts != -1) ? params->n_parts : 32;             // TODO: decide n_parts according to the # of processes 

    Graph *A_graph = convert_graph(A_org);
    idx_t *part = metis_partition(A_graph, n_parts, BBD_METIS_TYPE_KWAY);

    int *start_point = new int[n_parts + 2];
    int *Q = new int[A_org->get_nrow()];

    // TODO: update permute_matrix API, merge
    SparseMatrix *Pinv_matrix = permute_matrix(A_org, part, n_parts, Q, start_point);
    cs *cs_pinv_mtx = Pinv_matrix->to_csparse();
    cs *cs_p_mtx = cs_transpose(cs_pinv_mtx, 1);
    SparseMatrix *P_matrix = new SparseMatrix(cs_p_mtx);

    cs *cs_A_org = A_org->to_csparse();

    cs *cs_PA = cs_multiply(cs_p_mtx, cs_A_org);
    cs *cs_PAPinv = cs_multiply(cs_PA, cs_pinv_mtx);
    delete cs_PA;

    SparseMatrix *A_metis = new SparseMatrix(cs_PAPinv);
    A_metis->save_file_matlab("A_metis.m");

    tmp_timer.elapsedWallclockTime(params->metis_time);
    printf("==============Metis Partition Done, Runtime: %g ms==============\n", params->metis_time);


    printf ("==============Sequential BBD Start==============\n");
    std::vector<SparseMatrix *> matrix_arr(n_parts, NULL);   // used to store each partition

    std::vector<float_bbd *> xi_arr(n_parts, NULL);          // used to store each rhs, xi will be written to this arrary if it's sequential BBD
    std::vector<BBDSchur *> solver_arr(n_parts, NULL);       // solver array for each partition

    std::vector<SparseMatrix *> schur_mtx_arr(n_parts, NULL); // CA-1B
    std::vector<float_bbd *> schur_rhs_arr(n_parts, NULL);    // CA-1bi   size: nrow - schur_size

    float_bbd *tmp_rhs, *xc, *x_sol;
    int_bbd nrow, schur_size;
    SparseMatrix *D_matrix; 

    nrow = Ch_matrix->get_nrow();
    schur_size = start_point[n_parts+1] - start_point[n_parts];

    tmp_rhs = new float_bbd[nrow]();
    // tmp_rhs = update_rhs(0, BU_matrix, Ch_matrix, tmp_rhs);             // val = C/h x0 + BU[0] 
    for(int i=0; i<nrow; i++) {
      tmp_rhs[i] = 1.0;
    }

    printf ("=============KLU Test Start==============\n");
    klu_symbolic *Symbolic; klu_numeric *Numeric; klu_common Common; klu_defaults (&Common);
    Symbolic = klu_analyze (A_metis->get_nrow(), A_metis->get_colj(), A_metis->get_rowi(), &Common);
    Numeric = klu_factor (A_metis->get_colj(), A_metis->get_rowi(), A_metis->get_value(), Symbolic, &Common) ;
    klu_solve(Symbolic, Numeric, A_metis->get_nrow(), 1, tmp_rhs, &Common);
    for(int i=0; i<A_metis->get_nrow(); i++) printf("x[%d] = %f \n", i, tmp_rhs[i]);
    for(int i=0; i<nrow; i++) tmp_rhs[i] = 1.0;
    printf ("==============KLU Test Done==============\n");


    A_metis->to_lower();      // we have to remove upper triangular if using chomod_lite
    xc = get_xc_rhs(tmp_rhs, nrow, schur_size); // bottom part of rhs

    // TODO dense permutation
    // tmp_rhs_matrix = bbd_mm_mult_add(P_matrix, tmp_rhs_matrix, 1.0, CSPARSE_BLAS);    // Pb <= PAQQTx = Pb       //TODO memo leakage here

/////////////////////////////////////////////////////////////////////
    // unroll this for loop using OpenMP
    for(int i=0; i<n_parts; i++) {
      matrix_arr[i] = extract_one_partition(A_metis, start_point, n_parts, i);  // check done
      xi_arr[i] = extract_one_rhs(tmp_rhs, start_point, i);                     // check done  xi
      // string filename = "matrix_part_" + std::to_string(i) + ".m"; 
      // matrix_arr[i]->save_file_matlab(filename.c_str());
    }

    D_matrix = get_Dc_matrix(matrix_arr[0]);    // matrix in right-bottom corner; check done
    D_matrix->to_full();

    // unroll this for loop using MPI
    for(int i=0; i<n_parts; i++) {
      solver_arr[i] = new BBDSchur(SOLVER_TYPE_CHOLMOD_LITE, matrix_arr[i]->get_schur_size(), matrix_arr[i]);
      solver_arr[i]->fact();
      schur_mtx_arr[i] = solver_arr[i]->get_Schur();                 // CA-1B
      schur_rhs_arr[i] = solver_arr[i]->get_Schur_RHS(xi_arr[i]);    // CA-1bi

      SparseMatrix *tmp_x = new SparseMatrix(schur_size, 1, schur_rhs_arr[i], false);
      // filename = "CAb_part_" + std::to_string(i) + ".m"; 
      // tmp_x->save_file_matlab(filename.c_str());
    }

    for(int i=0; i<n_parts; i++) {
      D_matrix = bbd_mm_add(D_matrix, schur_mtx_arr[i], 1.0, -1.0, CSPARSE_BLAS, true);  // the last param used to free memory of old Dc_mtx
      bbd_dense_add(schur_size, schur_rhs_arr[i], xc, -1.0,  MKL_BLAS);                  // in-place operation  y = ax + y
    }

    D_matrix->to_dense();
    DenseMatrix *dense_Dc_mtx = new DenseMatrix(schur_size, 
                                              schur_size, D_matrix->get_value());
    // GMRES(dense_Dc_mtx, xc);


    //  Ai * xc = bi - Bi * xc
    for(int i=0; i<n_parts; i++) {                                  // MPI_Bcast, broadcast xc all processes
      solver_arr[i]->update_rhs(xi_arr[i], xc);     // in-place operation, will be written to             Ai xi  =  bi - Bi * xc
      solver_arr[i]->solve(xi_arr[i], xi_arr[i]);   // in-place operation, Ly = b, Ux = y
    }

    // merge xi, and xc, eg: [x0, x1, x2, ... xc]        
    x_sol = merge_solution(xi_arr, xc, start_point, n_parts);       // MPI_GatherV

    // TODO: tmp_rhs_matrix = cs_permute();    // recover solutions
    for(int i=0; i<A_metis->get_nrow(); i++) {
      printf("x[%d] = %f \n", i, x_sol[i]);
    }

    // Move to next iteration
    // tmp_rhs_matrix = update_rhs(1, BU_matrix, Ch_matrix, tmp_rhs_matrix);
    printf ("==============Sequential BBD Done==============\n");


}

#endif
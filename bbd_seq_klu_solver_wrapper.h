#ifndef _BBD_SEQ_KLU_SOLVER_H_
#define _BBD_SEQ_KLU_SOLVER_H_

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
#include "./extlibs/SuiteSparse/include/klu.h"
#include "./utils/precision_check.h"
#include "stdlib.h"



void bbd_seq_klu_solve(SparseMatrix *A_org, SparseMatrix *Ch_matrix, SparseMatrix *BU_matrix, 
                    std::vector<int_bbd> &node_index, std::vector<float_bbd> &time_step, Params *params) {


    Timer2 total_timer, tmp_timer;
    
    printf ("=============KLU Test Start==============\n");
    tmp_timer.start();
    float_bbd *klu_rhs = new float_bbd[A_org->get_nrow()]();
    for(int i=0; i<A_org->get_nrow(); i++) klu_rhs[i] = 1.0;
    klu_symbolic *Symbolic; klu_numeric *Numeric; klu_common Common; klu_defaults (&Common);
    // Common.btf = 0; Common.ordering=2; Common.scale=0;
    Symbolic = klu_analyze (A_org->get_nrow(), A_org->get_colj(), A_org->get_rowi(), &Common);
    Numeric = klu_factor (A_org->get_colj(), A_org->get_rowi(), A_org->get_value(), Symbolic, &Common) ;
    klu_solve(Symbolic, Numeric, A_org->get_nrow(), 1, klu_rhs, &Common);
    for(int i=0; i<50; i++) printf("x[%d] = %f \n", i, klu_rhs[i]);
    double klu_time; tmp_timer.elapsedWallclockTime(klu_time);
    printf("==============Direct KLU Runtime: %g ms==============\n", klu_time);
    printf ("==============KLU Test Done ==============\n");


    printf("==============Using Sequential BBD Solver with KLU==============\n");

    // printf("==============MC64 Ordering==============\n"); 
    // int *mc64_col_perm;
    // float_bbd *mc64_col_scale;
    // mc64_col_perm = new int[A_org->get_nrow() * 2]();
    // mc64_col_scale = new float_bbd[A_org->get_nrow() * 2]();
    // A_org->save_file_coo("A_org.mtx");
    // printf("Matrix size: %d, Diag count: %d before MC64\n", A_org->get_ncol(), A_org->get_diag_count());
    // SparseMatrix *A_mc64 = hsl_mc64_ordering(A_org, mc64_col_perm, mc64_col_scale);
    // printf("Matrix size: %d, Diag count: %d after MC64\n",A_mc64->get_ncol(), A_mc64->get_diag_count());
    // A_mc64->save_file_coo("A_mc64.mtx");

    SparseMatrix *A_mc64 = A_org;
    printf("==============Metis Ordering==============\n");

    /****** METIS Fill-in Reducing Ordering ******/
    tmp_timer.start(); total_timer.start();

    Graph *A_graph1 = convert_graph(A_mc64);

    int *metis_perm, *metis_iperm;
    metis_perm = new int[A_mc64->get_nrow()]();
    metis_iperm = new int[A_mc64->get_nrow()]();

    SparseMatrix *A_order_metis = metis_ordering(A_graph1, A_mc64, metis_perm, metis_iperm);
    A_order_metis->save_file_coo("A_order_metis1.mtx");
    printf("==============Partition Ordering==============\n");
    int_bbd n_parts;  // number of partition, if not set, we will use the best partiton according to n_proc, matrix size and sparsity
    n_parts = (params->n_parts != -1) ? params->n_parts : 32;             // TODO: decide n_parts according to the # of processes 

    Graph *A_graph2 = convert_graph(A_order_metis);
    idx_t *part = metis_partition(A_graph2, n_parts, BBD_METIS_TYPE_KWAY);

    int *start_point = new int[n_parts + 2];
    int *Q = new int[A_order_metis->get_nrow()]();
    int *P = new int[A_order_metis->get_nrow()]();

    SparseMatrix *Pinv_matrix = permute_matrix(A_order_metis, part, n_parts, Q, start_point);
    for(int i=0; i<A_order_metis->get_nrow(); i++) P[Q[i]] = i;

    cs *cs_A_metis = cs_permute(A_order_metis->to_csparse(), P, Q, 1);
    SparseMatrix *A_metis = new SparseMatrix(cs_A_metis);
    A_metis->save_file_coo("A_order_metis2.mtx");

    tmp_timer.elapsedWallclockTime(params->metis_time);
    printf("==============Metis Partition Done, Runtime: %g ms==============\n", params->metis_time);



    printf ("==============Sequential BBD with KLU Start==============\n");
    std::vector<SparseMatrix *> matrix_arr(n_parts, NULL);   // used to store each partition
    std::vector<float_bbd *> xi_arr(n_parts, NULL);          // used to store each rhs, xi will be written to this arrary if it's sequential BBD
    std::vector<BBDSchur *> solver_arr(n_parts, NULL);       // solver array for each partition
    std::vector<SparseMatrix *> schur_mtx_arr(n_parts, NULL); // CA-1B
    std::vector<float_bbd *> schur_rhs_arr(n_parts, NULL);    // CA-1bi   size: nrow - schur_size

    float_bbd *tmp_rhs_before, *tmp_rhs_middle, *tmp_rhs, *xc, *x_sol, *final_sol;
    int_bbd nrow, schur_size;
    SparseMatrix *D_matrix; 

    nrow = A_metis->get_nrow();
    schur_size = start_point[n_parts+1] - start_point[n_parts];
    final_sol = new float_bbd[nrow]();

    tmp_rhs_before = new float_bbd[nrow]();
    tmp_rhs_middle = new float_bbd[nrow]();
    tmp_rhs = new float_bbd[nrow]();
    for(int i=0; i<nrow; i++) tmp_rhs_before[i] = 1.0;
    // tmp_rhs = update_rhs(0, BU_matrix, Ch_matrix, tmp_rhs);             // val = C/h x0 + BU[0] 

    // for(int i=0; i<nrow; i++) tmp_rhs_middle[i] = tmp_rhs_before[[i]];
    // for(int i=0; i<nrow; i++) tmp_rhs_middle[i] *= row_scale[i];
    // for(int i=0; i<nrow; i++) tmp_rhs[i] = tmp_rhs_middle[metis_perm[Q[i]]] ;

    for(int i=0; i<nrow; i++) tmp_rhs[i] = tmp_rhs_before[metis_perm[Q[i]]];


    xc = get_xc_rhs(tmp_rhs, nrow, schur_size);         // bottom part of rhs
    // tmp_rhs_matrix = bbd_mm_mult_add(P_matrix, tmp_rhs_matrix, 1.0, CSPARSE_BLAS);    // Pb <= PAQQTx = Pb       //TODO memo leakage here

    // unroll this for loop using OpenMP
    for(int i=0; i<n_parts; i++) {
      matrix_arr[i] = extract_one_partition(A_metis, start_point, n_parts, i);  // check done
      xi_arr[i] = extract_one_rhs(tmp_rhs, start_point, i);                     // check done  xi
      string filename = "matrix_part_" + std::to_string(i) + ".m"; 
      matrix_arr[i]->save_file_matlab(filename.c_str());
      string filename2 = "matrix_part_" + std::to_string(i) + ".mtx";
      matrix_arr[i]->save_file_coo(filename2.c_str());
    }

    D_matrix = get_Dc_matrix(matrix_arr[0]);    // matrix in right-bottom corner; check done

    // unroll this for loop using MPI
    for(int i=0; i<n_parts; i++) {
      printf("factorize partition %d,  size: %d, schur_size: %d, nnz, %d, diag_count: %d\n", i, 
             matrix_arr[i]->get_nrow(), matrix_arr[i]->get_schur_size(), matrix_arr[i]->get_nnz(), matrix_arr[i]->get_diag_count() );
      solver_arr[i] = new BBDSchur(SOLVER_TYPE_BBD_KLU, matrix_arr[i]->get_schur_size(), matrix_arr[i]);
      solver_arr[i]->fact();
      schur_mtx_arr[i] = solver_arr[i]->get_Schur();                 // CA-1B
      schur_rhs_arr[i] = solver_arr[i]->get_Schur_RHS(xi_arr[i]);    // CA-1bi
    }

    for(int i=0; i<n_parts; i++) {
      printf("add partition %d \n", i);
      D_matrix = bbd_mm_add(D_matrix, schur_mtx_arr[i], 1.0, -1.0, CSPARSE_BLAS, true);  // the last param used to free memory of old Dc_mtx
      bbd_dense_add(schur_size, schur_rhs_arr[i], xc, -1.0,  MKL_BLAS);                  // in-place operation  y = ax + y

    }
    printf("GMRES Solve \n");
    printf("D_matrix size: %d, nnz, %d, density: %lf\n", D_matrix->get_nrow(), D_matrix->get_nnz(),  D_matrix->get_density());
    
    // D_matrix->to_dense();
    // DenseMatrix *dense_Dc_mtx = new DenseMatrix(schur_size, schur_size, D_matrix->get_value());
    // GMRES(dense_Dc_mtx, xc);

    klu_symbolic *Symbolic2; klu_numeric *Numeric2; klu_common Common2; klu_defaults (&Common2);
    Symbolic2 = klu_analyze (D_matrix->get_nrow(), D_matrix->get_colj(), D_matrix->get_rowi(), &Common2);
    Numeric2= klu_factor (D_matrix->get_colj(), D_matrix->get_rowi(), D_matrix->get_value(), Symbolic2, &Common2) ;
    klu_solve(Symbolic2, Numeric2, D_matrix->get_nrow(), 1, xc, &Common2);
    for(int i=0; i<schur_size; i++) printf("xc[%d] = %f \n", i, xc[i]); 

    //  Ai * xc = bi - Bi * xc
    for(int i=0; i<n_parts; i++) {                            // MPI_Bcast, broadcast xc all processes
      printf("update and solve rhs partition %d \n", i);
      solver_arr[i]->update_rhs(xi_arr[i], xc);               // in-place operation, will be written to             Ai xi  =  bi - Bi * xc
      solver_arr[i]->solve(xi_arr[i], xi_arr[i]);             // in-place operation, Ly = b, Ux = y
    }
     
    printf("Merge solution \n");
    x_sol = merge_solution(xi_arr, xc, start_point, n_parts);     // merge xi, and xc, eg: [x0, x1, x2, ... xc]    
    // int *mc64_col_perm_inv = new int[nrow];
    // for(int i=0; i<nrow; i++) mc64_col_perm_inv[mc64_col_perm[i]] = i;
    for(int i=0; i<nrow; i++) final_sol[i] = x_sol[P[metis_iperm[i]]];
    // for(int i=0; i<nrow; i++) final_sol[i] *= mc64_col_scale[i];
    for(int i=0; i<50; i++) printf("x[%d] = %f \n", i, final_sol[i]); 

    rmse_check(klu_rhs, final_sol, nrow);

    // Move to next iteration
    // tmp_rhs_matrix = update_rhs(1, BU_matrix, Ch_matrix, tmp_rhs_matrix);

    double bbd_klu;
    total_timer.elapsedWallclockTime(bbd_klu);
    printf("==============BBD KLU Runtime: %g ms==============\n", bbd_klu);
    printf ("==============Sequential BBD with KLU Done==============\n");


}

#endif

#ifndef _KLU_SOLVER_H_
#define _KLU_SOLVER_H_

#include "cholmod_lite.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"
#include "./utils/Timer.h"
#include "bbd_type.h"
#include "bbd_schur.h"
#include "./extlibs/SuiteSparse/include/klu.h"
#include "./utils/metis_api.h"
#include "./utils/cmd_line_parser.h"
#include "./utils/read_matrix.h"
#include "bbd_solver.h"
#include "./utils/bbd_blas.h"
#include "./utils/result_saver.h"
#include "./extlibs/gmres/gmres.h"


void bbd_klu_solve(SparseMatrix *A_org, SparseMatrix *Ch_matrix, SparseMatrix *BU_matrix, 
                    std::vector<int_bbd> &node_index, std::vector<float_bbd> &time_step, Params *params){

    Timer2 klu_timer;

    printf ("==============Use KLU Solver Start==============\n");
    klu_symbolic *Symbolic;
    klu_numeric *Numeric;
    klu_common Common;
    klu_defaults (&Common);
    // Common.btf = 0;

    klu_timer.start();
    Symbolic = klu_analyze (A_org->get_nrow(), A_org->get_colj(), A_org->get_rowi(), &Common);
    klu_timer.elapsedWallclockTime(params->sym_time);
    printf("==============Symbolic Analysis Done, Runtime: %g ms==============\n", params->sym_time);


    klu_timer.start();
    Numeric = klu_factor (A_org->get_colj(), A_org->get_rowi(), A_org->get_value(), Symbolic, &Common) ;
    klu_timer.elapsedWallclockTime(params->num_time);
    printf("==============Numerical Analysis Done, Runtime: %g ms==============\n", params->num_time);


    // transient iteration
    klu_timer.start();
    SparseMatrix *tmp_rhs_matrix;

    // added by cc
    ResultSaver res_handle = ResultSaver("./saved_results.csv", node_index, time_step);

    float_bbd *tmp_rhs = new float_bbd[Ch_matrix->get_nrow()]();

    res_handle.print_results(0, tmp_rhs); 


    size_t num_ts = time_step.size();
    for(int ts=1; ts<num_ts; ts++) {
        tmp_rhs = update_rhs(ts, BU_matrix, Ch_matrix, tmp_rhs, params->tran_method);   //TODO: SpMV val = C/h * x0 + BU[0] 

        // tmp_rhs

        klu_solve(Symbolic, Numeric, A_org->get_nrow(), 1, tmp_rhs, &Common);

        
        res_handle.print_results(ts, tmp_rhs);
        // printf ("No. %d ts: ", ts);
        // for (int_bbd i=0 ; i<3; i++) printf ("x[%d]=%.5g  ", i, rhs_tmp[i]);  // atten: use %.5g
        // printf ("...... \n");
    }
    for(int i=0;i<20;i++)
    {
        printf("%d  ",tmp_rhs[i]);
    }
    res_handle.close();
    klu_free_symbolic (&Symbolic, &Common);
    klu_free_numeric (&Numeric, &Common);

    klu_timer.elapsedWallclockTime(params->solve_time);
    printf("==============Solve Done, %d Time Steps, Runtime: %g ms==============\n", num_ts, params->solve_time);

}




#endif
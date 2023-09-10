
#ifndef _NICSLU_SOLVER_H_
#define _NICSLU_SOLVER_H_

#include "dense_matrix.h"
#include "sparse_matrix.h"
#include "./utils/Timer.h"
#include "bbd_type.h"
// #include "./extlibs/SuiteSparse/include/klu.h"
#include "./utils/cmd_line_parser.h"
#include "bbd_solver.h"
#include "./utils/bbd_blas.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "./extlibs/nicslu/nicslu202110/include/nicslu.h"
#include "./extlibs/nicslu/nicslu202110/include/nicslu_cpp.inl"
#include "./extlibs/nicslu/nicslu202110/include/nics_common.h"


const char *const ORDERING_METHODS[] = { "", "", "", "", "AMD", "AMM", "AMO1","AMO2","AMO3","AMDF" };
// CSR format
void bbd_nicslu_solve(SparseMatrix *A_org, SparseMatrix *Ch_matrix, SparseMatrix *BU_matrix, 
                    std::vector<int_bbd> &node_index, std::vector<float_bbd> &time_step, Params *params) {

    A_org->to_csr();

    //double cfg[32]={0,0.001,-1,6,10.0,0,4,1.5,80,4,8,0.95,0,1,3,10e-10,10e12,5.0,0,2,0,0,1,1};
    double stat[32];

    CNicsLU solver;
    
    float_bbd *ax;
    unsigned int *ai,*ap;
    _uint_t  n, row, col, nz, nnz, i, j;
    float_bbd res[4];

    row=A_org->get_nrow();
    n=row;
    nnz=A_org->get_nnz(); 
    ai=(unsigned int*)A_org->get_colj();
    ap=(unsigned int*)A_org->get_rowi();
    ax=A_org->get_value();

    float_bbd *tmp_rhs = new float_bbd[Ch_matrix->get_nrow()]();
    size_t num_ts = time_step.size();
    int_bbd ts=0;

    tmp_rhs = update_rhs(ts, BU_matrix, Ch_matrix, tmp_rhs, params->tran_method);
    float_bbd *x_sol=new float_bbd[n];

    int ret=solver.Initialize();

    solver.ctrl[3]=2;     // Configuring the Ordering Method for NICSLU
    if(ret<0){
        printf("Initialization Failed! \n"); 
    }
    else {
        printf("Solver Initialized! \n");
        solver.SetConfiguration(0, 1.);
    }

    solver.Analyze(n, ax, ai, ap, MATRIX_ROW_REAL);
    solver.CreateThreads(0);     //Using all threads we have
    solver.FactorizeMatrix(ax, 0);


    // For factor extraction 
    
    _double_t *lx,*ux;             //Allcoating space for the LU matrices
    _uint_t *li,*ui;
    _size_t *lp,*up;
    printf("%d %d ",(int)solver.GetInformation(9),(int)solver.GetInformation(10));
    lx=new _double_t[(int)solver.GetInformation(9)];
    ux=new _double_t[(int)solver.GetInformation(10)];
    lp=new _size_t[n+1];
    up=new _size_t[n+1];
    li=new _uint_t[(int)solver.GetInformation(9)];
    ui=new _uint_t[(int)solver.GetInformation(10)];
    printf("Memories are Allocated \n");

    solver.GetFactors(lx,li,lp,ux,ui,up,0,NULL,NULL,NULL,NULL); //Save LU to arrays
    printf("LU Factorization Done!  \n");
    FILE *fp;


    solver.Solve(tmp_rhs, x_sol);
    printf("solve time: %g\n", solver.GetInformation(2));
    for(int i=0;i<20;i++) printf("%d  ",x_sol[i]);

    // float_bbd total_time=0;
    // for(int ts=1;ts<10; ts++)
    // {
    //     tmp_rhs = update_rhs(ts, BU_matrix, Ch_matrix, tmp_rhs, params->tran_method);
    //     solver.Solve(tmp_rhs,x_sol);
    //     total_time=total_time+solver.GetInformation(2);
    // }
    //printf("======NICSLU Done! Total time = %g  ======\n",total_time);
    
    // delete[] tmp_rhs;
    // delete[] lx;
    // delete[] li;
    // delete[] lp;
    // delete[] ux;
    // delete[] up;
    // delete[] ui;

}




#endif
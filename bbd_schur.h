
#ifndef _BBD_SCHUR_H_
#define _BBD_SCHUR_H_

#include <vector>
#include <set>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <memory.h>
#include "sparse_matrix.h"
#include "./utils/pardiso_api.h"
#include "cholmod_lite.h"
#include "./extlibs/SuiteSparse/include/cs.h"
#include "mkl.h"
#include "bbd_type.h"
#include "./utils/bbd_blas.h"
#include "./extlibs/SuiteSparse/include/klu.h"

class BBDSchur {

private:
  int solver_type;

  int_bbd nrow;
  int_bbd ncol;

  int_bbd *colj;
  int_bbd *rowi;
  float_bbd *val;

  int_bbd nnz_org;
  int_bbd nnz_sym;

  int_bbd decompose_size;   // for BBD solver
  int_bbd nrow_schur;        

  SparseMatrix *A;          // [A B C D]

  SparseMatrix *C_Uinv;     // C x U-1
  SparseMatrix *Linv_B;     // L-1 x B
  SparseMatrix *Schur;      // C x A-1 x B = C x U-1 x L-1 x B

  SparseMatrix *L_part;     // lower triangular after LU, extracted from factorized A
  SparseMatrix *U_part;     // upper triangular after LU, extracted from factorized A
  SparseMatrix *B;


  /**********KLU*********/
  klu_symbolic *Symbolic;
  klu_numeric *Numeric;
  klu_common *Common;

  /**********NICSLU*********/


  CholLite *schur_solver;

  void shift_index(int n, int nnz, int* ia, int* ja, int value)
  {
    int i;
    for (i = 0; i < n+1; i++){
        ia[i] += value;
    }
    for (i = 0; i < nnz; i++){
        ja[i] += value;
    }
  }

  
public:

  BBDSchur(int type, int nrow_schur_, SparseMatrix *A_) {

    A = A_;   // [A B C D]

    solver_type = type;
    nrow_schur = nrow_schur_;
    decompose_size = A->get_nrow() - nrow_schur;
    nrow = A->get_nrow(); 
    ncol = A->get_ncol();
    nnz_org = A->get_nnz();

    /*********** schur matrix ***********/
    C_Uinv = NULL;     // C x U-1,  (nrow_schur, decompose_size)
    Linv_B = NULL;     // L-1 x B,  (decompose_size, nrow_schur)
    Schur = NULL;      // C x A-1 x B = C x U-1 x L-1 x B     (nrow_schur, nrow_schur)

    L_part = NULL;     // lower triangular after LU, extracted from factorized A
    U_part = NULL;     // upper triangular after LU, extracted from factorized A

    B = NULL;
    get_B();

    if(solver_type == SOLVER_TYPE_CHOLMOD_LITE) {
      schur_solver = new CholLite();
    }
    else if (solver_type == SOLVER_TYPE_BBD_KLU) {
      Common = new klu_common();
      klu_defaults (Common);
      Common->btf = 0;
      Common->ordering = 2;  /* 0: AMD, 1: COLAMD, 2: user P and Q,
                             * 3: user function */
      Common->scale = 0;
    }
    else if (solver_type == SOLVER_TYPE_BBD_NICSLU) {

    }
    else if (solver_type == SOLVER_TYPE_CHOLMOD) {
      
    }
    else {
      // support other solver in the future
    }

  }

  void symbolic_analysis() {
    if(solver_type == SOLVER_TYPE_CHOLMOD_LITE) {
        schur_solver->symbolic(A->get_nrow(), decompose_size, 
                                A->get_colj(), A->get_rowi());
    }
    else if (solver_type == SOLVER_TYPE_BBD_KLU) {
        Symbolic = klu_analyze (A->get_nrow(), A->get_colj(), 
                                        A->get_rowi(), Common);
    }
    // TODO
    else if(solver_type == SOLVER_TYPE_CHOLMOD) {

    }

  }

  void numerical_analysis(){
    if(solver_type==SOLVER_TYPE_CHOLMOD_LITE) {
        schur_solver->refactor(A->get_colj(), A->get_rowi(), A->get_value());
        A->update(nrow, ncol, schur_solver->get_nnz(), schur_solver->get_colj(), 
                            schur_solver->get_rowi(), schur_solver->get_value());

        // extract components
        get_CUinv(A);
        get_LinvB(A);
        get_CAinvB();
        get_lower_triangular(A);
        get_upper_triangular(A);
    }

    else if (solver_type == SOLVER_TYPE_BBD_KLU) {

      
        /* returns KLU_OK if OK, < 0 if error */
        Numeric = klu_factor (A->get_colj(), A->get_rowi(), A->get_value(), 
                                  Symbolic, Common);
        if(Numeric == NULL){
          printf("klu_factor fail!\n");
          return;
        }

        SparseMatrix *L_mtx = new SparseMatrix(ncol, ncol, Numeric->lnz);
        SparseMatrix *U_mtx = new SparseMatrix(ncol, ncol, Numeric->unz);
        SparseMatrix *F_mtx = new SparseMatrix(ncol, ncol, Numeric->nzoff);

        //AMD without scaling

        //get_Linv
        //get_Uinv
        //getLinvLinv()
        //getCAinvB

        int ok;
        int *P = new int[ncol];
        int *Pinv = new int[ncol];
        int *Q = new int[ncol];
        int *R = new int[ncol];
        float_bbd *Rs = new float_bbd[ncol];

        ok = klu_extract (Numeric, Symbolic, 
                          L_mtx->get_colj(), L_mtx->get_rowi(), L_mtx->get_value(), 
                          U_mtx->get_colj(), U_mtx->get_rowi(), U_mtx->get_value(), 
                          F_mtx->get_colj(), F_mtx->get_rowi(), F_mtx->get_value(), 
                          P, Q, Rs, R, Common);
        // L_mtx->save_file_matlab("extract_l.m");
        // U_mtx->save_file_matlab("extract_u.m");
        get_CUinv(L_mtx);
        get_LinvB(U_mtx);
        get_CAinvB();
        get_lower_triangular(L_mtx);
        get_upper_triangular(U_mtx);
        // save_all_matrix();
        
        delete L_mtx;
        delete U_mtx;
        delete F_mtx;
        delete P;
        delete Pinv;
        delete Q;
        delete R;
        delete Rs;
    }



    // TODO: Modify
    else if(solver_type == SOLVER_TYPE_CHOLMOD) { 
        // schur_solver->refactor(A->get_colj(), A->get_rowi(), 
        //                         A->get_value());

        // A->clear();
        // A->update(nrow, ncol, schur_solver->get_nnz(), schur_solver->get_colj(), schur_solver->get_rowi(), schur_solver->get_value());


    }
    
    else if(solver_type == SOLVER_TYPE_NONE) {

    }

  }

void getLinv(SparseMatrix *L_matrix)
{
  int new_ncol;
  int *new_colj, *new_rowi;
  float_bbd *new_val;

  new_ncol = L_matrix->get_ncol();
  new_colj = L_matrix->get_colj();
  new_rowi = L_matrix->get_rowi();
  new_val = L_matrix->get_value();



}



  void fact() {
    symbolic_analysis();
    numerical_analysis();
  }

  /**
   * this function should be called before factorization
  */
  void get_B() {
    
    if(solver_type == SOLVER_TYPE_CHOLMOD || solver_type == SOLVER_TYPE_CHOLMOD_LITE) {

      // get C, transpose
      cs *cs_b, *cs_c;
      int *colj, *rowi, *colj_c, *rowi_c;
      double *val, *val_c;

      colj = A->get_colj(); rowi = A->get_rowi(); val = A->get_value();

      SparseMatrix *C_mtx = new SparseMatrix(nrow_schur, decompose_size);
      colj_c = C_mtx->get_colj(); rowi_c = C_mtx->get_rowi(); val_c = C_mtx->get_value();    

      int_bbd nnz  = 0;
      colj_c[0] = 0;
      for(int_bbd j=0; j<decompose_size; j++) {
        for(int_bbd p=colj[j]; p<colj[j+1]; p++) {
          int_bbd row_idx = rowi[p];
          if(row_idx >= decompose_size){
            if(C_mtx->realloc(j)){         // make sure we have enough memory; if reallocated, we need to update pointers
                colj_c = C_mtx->get_colj(); rowi_c = C_mtx->get_rowi(); val_c = C_mtx->get_value();    
            }

            rowi_c[nnz] = row_idx-decompose_size;
            val_c[nnz++] = val[p];
          }

        }

        colj_c[j+1] = nnz;
      }

      C_mtx->set_nnz(nnz);

      cs_c = C_mtx->to_csparse();
      cs_b = cs_transpose(cs_c, 1);

      B = new SparseMatrix(cs_b);

   }

   else {
        int_bbd *colj, *rowi, *colj_b, *rowi_b;
        float_bbd *val, *val_b;

        colj = A->get_colj(); 
        rowi = A->get_rowi(); 
        val = A->get_value();

        B = new SparseMatrix(decompose_size, nrow_schur);
        colj_b = B->get_colj(); 
        rowi_b = B->get_rowi(); 
        val_b = B->get_value();    

        int_bbd nnz  = 0;
        colj_b[0] = 0;
        for(int j=decompose_size; j<ncol; j++) {
          for(int p=colj[j]; p<colj[j+1]; p++) {
            int row_idx = rowi[p];
            if(row_idx < decompose_size) {
              if(B->realloc(j-decompose_size)){         // make sure we have enough memory; if reallocated, we need to update pointers
                  colj_b = B->get_colj(); 
                  rowi_b = B->get_rowi(); 
                  val_b = B->get_value();    
              }
              rowi_b[nnz] = row_idx;
              val_b[nnz++] = val[p];
            }

          }
          colj_b[j+1-decompose_size] = nnz;
        }
        B->set_nnz(nnz);

    }


  }

  void get_CUinv(SparseMatrix *A_) {

    if (solver_type == SOLVER_TYPE_CHOLMOD || solver_type == SOLVER_TYPE_CHOLMOD_LITE || solver_type == SOLVER_TYPE_BBD_KLU) {

      int *colj, *rowi, *colj_c, *rowi_c;
      double *val, *val_c;

      colj = A_->get_colj(); 
      rowi = A_->get_rowi(); 
      val = A_->get_value();

      C_Uinv = new SparseMatrix(nrow_schur, decompose_size);
      colj_c = C_Uinv->get_colj(); 
      rowi_c = C_Uinv->get_rowi(); 
      val_c = C_Uinv->get_value();    

      int_bbd nnz  = 0;
      colj_c[0] = 0;
      for(int j=0; j<decompose_size; j++){
        for(int p=colj[j]; p<colj[j+1]; p++){
          int row_idx = rowi[p];
          if(row_idx >= decompose_size){
            if(C_Uinv->realloc(j)){         // make sure we have enough memory; if reallocated, we need to update pointers
                colj_c = C_Uinv->get_colj(); 
                rowi_c = C_Uinv->get_rowi(); 
                val_c = C_Uinv->get_value();    
            }

            rowi_c[nnz] = row_idx-decompose_size;
            val_c[nnz++] = val[p];
          }

        }

        colj_c[j+1] = nnz;
      }
      C_Uinv->set_nnz(nnz);

   }

   else {
      printf("Please compelte get_CUinv function for this new solver first!!!\n");
    }



  }

  void get_LinvB(SparseMatrix *A_){

    if(solver_type == SOLVER_TYPE_CHOLMOD || solver_type == SOLVER_TYPE_CHOLMOD_LITE) {
      // L-1B = transpose(CU-1)
      cs *cs_mtx, *cs_linvb;
      if(C_Uinv == NULL)
          get_CUinv(A_);
      cs_mtx = C_Uinv->to_csparse();
      cs_linvb = cs_transpose(cs_mtx, 1);
      Linv_B = new SparseMatrix(cs_linvb);
    }
    else if (solver_type == SOLVER_TYPE_BBD_KLU) {

        int_bbd *colj, *rowi, *colj_b, *rowi_b;
        float_bbd *val, *val_b;

        colj = A_->get_colj(); 
        rowi = A_->get_rowi(); 
        val = A_->get_value();

        Linv_B = new SparseMatrix(decompose_size, nrow_schur);
        colj_b = Linv_B->get_colj(); 
        rowi_b = Linv_B->get_rowi(); 
        val_b = Linv_B->get_value();    

        int_bbd nnz  = 0;
        colj_b[0] = 0;
        for(int j=decompose_size; j<ncol; j++) {
          for(int p=colj[j]; p<colj[j+1]; p++) {
            int row_idx = rowi[p];
            if(row_idx < decompose_size) {
              if(Linv_B->realloc(j-decompose_size)){         // make sure we have enough memory; if reallocated, we need to update pointers
                  colj_b = Linv_B->get_colj(); 
                  rowi_b = Linv_B->get_rowi(); 
                  val_b = Linv_B->get_value();    
              }
              rowi_b[nnz] = row_idx;
              val_b[nnz++] = val[p];
            }

          }
          colj_b[j+1-decompose_size] = nnz;
        }
        Linv_B->set_nnz(nnz);
    }

    else {
      printf("Please complete get_LinvB function for this new solver first!!!\n");
    }

  }

  // CA-1B  == CU-1 x L-1B
  void get_CAinvB() {
    cs *cu, *lb, *cs_schur;
    cu = C_Uinv->to_csparse();
    lb = Linv_B->to_csparse();
    cs_schur = cs_multiply(cu, lb);
    Schur = new SparseMatrix(cs_schur);
  }

  // CA-1B
  SparseMatrix *get_Schur(){
    return Schur;
  }

  // Lx = rhs,  x = L-1 x rhs
  void l_solve(float_bbd *rhs, float_bbd *x){
    if (L_part){
      cs *cs_l = L_part->to_csparse();
      cs_lsolve(cs_l, x);
    } else {
      printf("L_part of matrix does not exist, please factorize matrix firstly!!\n");
    }

  }
  // Ux = rhs
  void u_solve(float_bbd *rhs, float_bbd *x) {

    if (U_part){
      cs *cs_u = U_part->to_csparse();
      cs_usolve(cs_u, x);
    } else {
      printf("U_part of matrix does not exist, please factorize matrix firstly!!\n");
    }

  }

  float_bbd *solve(float_bbd *rhs, float_bbd *x) {
    // in-place operation
    l_solve(rhs, x);
    u_solve(x, x);

    return x;
  }

  // bi - Bi * xc

  void update_rhs(float_bbd *xi, float_bbd *xc) {
    float_bbd *tmp_res;
    int_bbd size = B->get_nrow();
    tmp_res = bbd_spmv(1.0, B, xc, MKL_BLAS);
    xi = bbd_dense_add(size, tmp_res, xi, -1, MKL_BLAS);
  }


  void get_lower_triangular(SparseMatrix *A_) {

    if(solver_type == SOLVER_TYPE_CHOLMOD || solver_type == SOLVER_TYPE_CHOLMOD_LITE || solver_type == SOLVER_TYPE_BBD_KLU) {

      int *colj, *rowi, *colj_l, *rowi_l;
      double *val, *val_l;

      colj = A_->get_colj(); rowi = A_->get_rowi(); val = A_->get_value();

      L_part = new SparseMatrix(decompose_size, decompose_size);
      colj_l = L_part->get_colj(); rowi_l = L_part->get_rowi(); val_l = L_part->get_value();    

      int nnz  = 0;
      colj_l[0] = 0;
      for(int j=0; j<decompose_size; j++){
        for(int p=colj[j]; p<colj[j+1]; p++){
          int row_idx = rowi[p];
          if(row_idx < j)
            continue;

          if(L_part->realloc(j)){    // make sure we have enough memory; after reallocated, we need to update pointers
              colj_l = L_part->get_colj(); 
              rowi_l = L_part->get_rowi(); 
              val_l = L_part->get_value();    
          }
          if(row_idx < decompose_size) {
              rowi_l[nnz] = row_idx;
              val_l[nnz++] = val[p];
          }
        }
        colj_l[j+1] = nnz;
      }

      L_part->set_nnz(nnz);

    }

    // else if(solver_type == SOLVER_TYPE_BBD_KLU) {

      // int *colj, *rowi, *colj_l, *rowi_l;
      // double *val, *val_l;

      // colj = A_->get_colj(); rowi = A_->get_rowi(); val = A_->get_value();

      // L_part = new SparseMatrix(decompose_size, decompose_size);
      // colj_l = L_part->get_colj(); rowi_l = L_part->get_rowi(); val_l = L_part->get_value();    

      // int nnz  = 0;
      // colj_l[0] = 0;
      // for(int j=0; j<decompose_size; j++){
      //   for(int p=colj[j]; p<colj[j+1]; p++){
      //     int row_idx = rowi[p];
      //     if(row_idx < j)
      //       continue;

      //     if(L_part->realloc(j)){    // make sure we have enough memory; after reallocated, we need to update pointers
      //         colj_l = L_part->get_colj(); 
      //         rowi_l = L_part->get_rowi(); 
      //         val_l = L_part->get_value();    
      //     }
      //     if(row_idx < decompose_size) {
      //         rowi_l[nnz] = row_idx;
      //         val_l[nnz++] = val[p];
      //     }
      //   }
      //   colj_l[j+1] = nnz;
      // }

      // L_part->set_nnz(nnz);

    // }

    else {
      printf("Please complete get_lower_triangular function !!!\n");
    }

  }


  void get_upper_triangular(SparseMatrix *A_){

    if(solver_type == SOLVER_TYPE_CHOLMOD || solver_type == SOLVER_TYPE_CHOLMOD_LITE) {
      // U_part = transpose(L_part)
      cs *cs_mtx, *cs_u;
      cs_mtx = L_part->to_csparse();
      cs_u = cs_transpose(cs_mtx, 1);
      U_part = new SparseMatrix(cs_u);
    }
    else if(solver_type == SOLVER_TYPE_BBD_KLU) {
      int *colj, *rowi, *colj_u, *rowi_u, u_ncol;
      double *val, *val_u;

      colj = A_->get_colj(); rowi = A_->get_rowi(); val = A_->get_value();
      u_ncol = A_->get_ncol();
      // assert(u_ncol == decompose_size);

      U_part = new SparseMatrix(decompose_size, decompose_size);
      colj_u = U_part->get_colj(); rowi_u = U_part->get_rowi(); val_u = U_part->get_value();    

      int nnz  = 0;
      colj_u[0] = 0;
      for(int j=0; j<decompose_size; j++) {
        for(int p=colj[j]; p<colj[j+1]; p++){
          int row_idx = rowi[p];
          if(row_idx <= j) {
              if(U_part->realloc(j)){         // make sure we have enough memory; after reallocated, we need to update pointers
                  colj_u = U_part->get_colj(); 
                  rowi_u = U_part->get_rowi(); 
                  val_u = U_part->get_value();    
              }
              rowi_u[nnz] = row_idx;
              val_u[nnz++] = val[p];
            }
            colj_u[j+1] = nnz;
        }

      }

      U_part->set_nnz(nnz);
    }

    else {
      printf("Please complete get_upper_triangular function !!!\n");
    }

  }

  // CA-1b = CU-1 x L-1b
float_bbd *get_Schur_RHS(float_bbd *xi_rhs){

    cs *cs_l, *cs_cu;
    float_bbd *res;

    int_bbd size = C_Uinv->get_ncol();
    float_bbd *xi_rhs_copy = new float_bbd[size];
    memcpy(xi_rhs_copy, xi_rhs, sizeof(float_bbd)*size);

    cs_l = L_part->to_csparse();
    cs_lsolve(cs_l, xi_rhs_copy);

    res = bbd_spmv(1.0, C_Uinv, xi_rhs_copy, MKL_BLAS);

    delete []xi_rhs_copy;

    return res;
    
  }

// SparseMatrix *get_xi(SparseMatrix *rhs_mtx){

//     int_bbd *new_colj, new_rowi;
//     float_bbd *new_val;

//     new_val = rhs_mtx->get_value();

//     new_colj = new int_bbd[2];
//     new_colj[0] = 0;
//     new_colj[1] = rhs_mtx->get_nrow()-rhs_mtx->get_schur_size();

//     SparseMatrix *new_rhs_mtx = new SparseMatrix(cs_schur_rhs);
//     rhs_mtx->clear();

//     return new_rhs_mtx;
    
//   }



void clear(){
  
  // int *rowi;
  // int *colj;
  // double *val;

  // SparseMatrix *A;        // [A B C D]
  // SparseMatrix *C_Uinv;
  // SparseMatrix *Linv_B;
  // SparseMatrix *Schur;    // CA-1B

  // SparseMatrix *L_part;
  // SparseMatrix *U_part;
  // // float_bbd *schur_rhs;    // CA-1b      nrow_schur x 1

  // CholLite *schur_solver;

}


void save_all_matrix(){

    C_Uinv->save_file_matlab("CUinv.m");
    Linv_B->save_file_matlab("LinvB.m");
    Schur->save_file_matlab("schur.m");
    L_part->save_file_matlab("L_part.m");
    U_part->save_file_matlab("U_part.m");

}

  ~BBDSchur() {
    clear();
  }

};

#endif

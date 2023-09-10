/**
author: dztu
date: Oct. 07, 2022
version: v3.0
**/

#ifndef _SPARSEMATRIX_H_
#define _SPARSEMATRIX_H_


#include "./extlibs/SuiteSparse/include/cs.h"
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <map>
#include "mkl.h"
#include "bbd_type.h"

// #include "./extlibs/Eigen/Sparse"

// #define SOLVER_TYPE_NONE    0
// #define SOLVER_TYPE_CHOLMOD 1
// #define SOLVER_TYPE_KLU     2
// #define SOLVER_TYPE_NICSLU  3
// #define SOLVER_TYPE_PARDISO 4
// #define SOLVER_TYPE_SUPERLU 5
// #define SOLVER_TYPE_GLU     6
// #define SOLVER_TYPE_CHOLMOD_LITE 7   // without reordering


class SparseMatrix {
private:
  // int  solver_type;     // cholmod, klu or other solvers
  bool is_symmetric;    // if it is a symmetric matrix. We only save lower triangle for symmetrix matrices

  // matrix in CSC format
  int_bbd nrow;
  int_bbd ncol;
  int_bbd nnz;
  size_t nz_max;      // allocate extra memory for sparse if needed
  int_bbd *colj;       // size = ncol+1
  int_bbd *rowi;       // size = nnz
  float_bbd *val;      // size = nnz


  int_bbd schur_size;
  
  // map format to build random matrix
  std::vector <std::map<int_bbd, float_bbd>> *val_map;  // the data is stored in column major form. each column has a map

  // some other stats
  // sparsity
  // nflops
  // runtime tracing

public:
  SparseMatrix();
  SparseMatrix(int_bbd nrow_, int_bbd ncol_);
  SparseMatrix(int_bbd nrow_, int_bbd ncol_, int_bbd nnz_);   // just allocate memory
  SparseMatrix(int_bbd nrow_, int_bbd ncol_, float_bbd *val_, bool trans);  // dense formate actually
  SparseMatrix(int_bbd nrow_, int_bbd ncol_, int_bbd nnz_, int_bbd *p, int_bbd *i, float_bbd *data);
  SparseMatrix(cs *cs_mtx);



  int_bbd get_nrow(){
    return nrow;
  }

  int_bbd get_ncol(){
    return ncol;
  }

  int_bbd get_nnz(){
    return nnz;
  }

  int_bbd get_nz_max(){
    return nz_max;
  }

  bool get_symmetric(){
    return is_symmetric;
  }

  void set_nnz(int_bbd nnz_){
    nnz = nnz_;
  }

  int_bbd *get_colj(){
    return colj;  // ncol +1 
  }

  int_bbd *get_rowi(){
    return rowi;  // nnz
  }
 
  float_bbd *get_value(){
    return val;   //nnz 
  }

  int_bbd get_schur_size(){
    return schur_size;
  }

  void set_schur_size(int_bbd schur_size_){
    schur_size = schur_size_;
  }


  SparseMatrix *get_sym_pattern(){


    SparseMatrix *sym_pat = new SparseMatrix();
    sym_pat->init(nrow, ncol, false);

    for(unsigned j=0; j<ncol; j++) {
      for(unsigned p=colj[j]; p<colj[j+1]; p++) {
        unsigned ridx = rowi[p];
        if( ridx == j) {
          sym_pat->add_entry(j, j, val[p]);
        }

        else {
          sym_pat->add_entry(ridx, j, val[p]);
          sym_pat->add_entry(j, ridx, 0);
        }

      }
    }
    sym_pat->compact();


    clear();

    colj = sym_pat->get_colj();
    rowi = sym_pat->get_rowi();
    val = sym_pat->get_value();

    nnz = sym_pat->get_nnz();
    nz_max = nnz;


    return sym_pat;

  }
  int_bbd get_diag_count(){

    unsigned diag_count = 0;

    for(unsigned j=0; j<ncol; j++) {
      for(unsigned p=colj[j]; p<colj[j+1]; p++) {
        if(rowi[p] == j) {
            diag_count++;
        }
      }
    }

    if(diag_count < ncol){
      printf("this is not SPD!!!\n");
    }
    else {
      printf("Great. This is SPD!!\n");
    }


    return diag_count;
  }



  // SparseMatrix* get_inverse() {

  //   Eigen::Map<Eigen::SparseMatrix<double> > A_eigen(nrow, ncol, nnz, colj, rowi, val); //RW
  //   Eigen::SparseMatrix<double> eye_I(nrow, ncol);
  //   Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

    // eye_I.setIdentity();
    // solver.compute(A_eigen);
    // A_eigen.
    // // solver.analyzePattern(A); 
    // // solver.factorize(A); 

    // auto A_inv = solver.solve(eye_I);

    // SparseMatrix *A_inv_sp = new SparseMatrix();
    // return A_inv_sp;
  // }


  // SparseMatrix *get_a_col(size_t j){

  //   size_t nnz; 
  //   nnz = colj[j+1] - colj[j];

  //   int_bbd *new_colj = new int_bbd[2];
  //   int_bbd *new_rowi = new int_bbd[nnz];
  //   float_bbd *new_val = new float_bbd[nnz];

  //   new_colj[0] = 0;
  //   new_colj[1] = nnz;

  //   memcpy(new_rowi, &rowi[colj[j]], sizeof(int_bbd)*nnz);
  //   memcpy(new_val, &val[colj[j]], sizeof(float_bbd)*nnz);

  //   SparseMatrix *col = new SparseMatrix(nrow, 1,  nnz, 
  //                                        new_colj,  new_rowi,  new_val);
  //   return col;
  // }

    float_bbd *get_a_col(size_t j) {

      float_bbd *one_col = new float_bbd[nrow];
      memset(one_col, 0, sizeof(float_bbd) * nrow);

      for(int_bbd p=colj[j]; p<colj[j+1]; p++) {
        one_col[rowi[p]] = val[p];
      }
      
      return one_col;
  }

  /************ intialization ************/
  void init(int_bbd m, int_bbd n, bool is_sym);
  void add_entry(int_bbd row, int_bbd col, float_bbd d);  // add an element to matrix
  void compact(); // convert to CSC format
  void clear();
  
  bool realloc(int_bbd j);
  void update(int_bbd nrow_, int_bbd ncol_, int_bbd nnz_, 
              int_bbd *p, int_bbd *i, float_bbd *data);
  /************ factorize ************/
  bool factor();
  bool solve(std::vector<float_bbd> &r, std::vector<float_bbd> &x);
  void sparsify(SparseMatrix &tgt, float_bbd tol);


  /************ BLAS ************/
  bool multiply_add(std::vector<float_bbd> &x, std::vector<float_bbd> &y);   // y += A * x
  bool multiply_sub(std::vector<float_bbd> &x, std::vector<float_bbd> &y);   // y -= A * x


  /************ stats ************/
  float_bbd get_density();


  /************ save matrix ************/
  void save_file_matlab (const char filename[]);
  void save_file_coo (const char filename[]);

  /************ stats ************/
  

  /************ Runtime log ************/

  /************ format transformation ************/
  cs *to_csparse(){

    cs *cs_mtx = new cs;

    cs_mtx->m = nrow;
    cs_mtx->n = ncol; 
    cs_mtx->nz = -1;
    cs_mtx->nzmax = nnz;

    cs_mtx->p = colj; 
    cs_mtx->i = rowi;
    cs_mtx->x = val;

    return cs_mtx;
}

  void to_full() {

    if (val_map)
      delete val_map;

    val_map = new std::vector <std::map<int_bbd, float_bbd>>;
    val_map->resize(ncol);

    for(int_bbd j=0; j<ncol; j++) {
      for(int_bbd p=colj[j]; p<colj[j+1]; p++) {
          int_bbd ridx = rowi[p];
          float_bbd value = val[p];
          if(ridx != j) {
            add_entry(ridx, j, val[p]);
            add_entry(j, ridx, val[p]);
          }
          else {
            add_entry(ridx, j, val[p]);
          }
      }
    }

    is_symmetric = false;

    if (colj) {
      delete [] colj;
      colj = NULL;
    }
    if (rowi) {
      delete [] rowi;
      rowi = NULL;
    }
    if (val) {
      delete [] val;
      val = NULL;
    }

    compact();

}

  void to_dense(){
    // col-major layout
    nnz = nrow * ncol;

    int_bbd *new_colj = new int_bbd[ncol+1];
    int_bbd *new_rowi = new int_bbd[nnz];
    float_bbd *new_val = new float_bbd[nnz];

    nz_max = nnz;

    memset(new_val, 0.0, sizeof(float_bbd) * nnz);

    new_colj[0] = 0;
    for(int_bbd j=0; j<ncol; j++){

      for(int_bbd p=colj[j]; p<colj[j+1]; p++){
          int_bbd row_idx = rowi[p];
          new_val[j*nrow+row_idx] = val[p];
      }

      for(int_bbd i=0; i<nrow; i++) {
        new_rowi[j*nrow+i] = i;
      }

      new_colj[j+1] = nrow * (j+1);

    }

    clear(); // release old memory

    colj = new_colj;
    rowi = new_rowi;
    val = new_val;

  }
  
  void init_rand_matrix(int_bbd nrow=6, int_bbd ncol=6, bool is_sym=false) {

    srand(1);

    init(nrow, ncol, is_sym);    // true: symmetric ; false: non-symmetric




    if(nrow == ncol) {
      for (int_bbd i=0; i<nrow; i++)
        add_entry(i, i, ((float_t)rand()) / RAND_MAX);

      for (int i=0; i<2* nrow; i++) {
        int_bbd ri = rand() % nrow;
        int_bbd cj = rand() % ncol;
        float_bbd g = ((float_bbd)rand()) / RAND_MAX;
        add_entry(ri, ri, g);
        add_entry(cj, cj, g);
        add_entry(ri, cj, -g);
        add_entry(cj, ri, -g);
      }

      schur_size = 3;
    }
    else {

      for (int i=0; i<100 * nrow; i++) {
        int_bbd ri = rand() % nrow;
        int_bbd cj = rand() % ncol;
        float_bbd g = ((float_bbd)rand()) / RAND_MAX;
        add_entry(ri, cj, -g);
      }
      for (int i=0; i<100 * nrow; i++) {
        int_bbd ri = rand() % nrow;
        int_bbd cj = rand() % ncol;
        float_bbd g = ((float_bbd)rand()) / RAND_MAX;
        add_entry(ri, cj, g);
      }

    }
 
    compact();

  }
  
  // convert, test


  SparseMatrix *to_SPD() {
    
    SparseMatrix *new_A = new SparseMatrix();
    new_A->init(nrow, ncol, false);    // true: symmetric ; false: non-symmetric

    for(int_bbd j=0; j<ncol; j++) {
      new_A->add_entry(j, j, 1000);
    }

    for(int_bbd j=0; j<ncol; j++){
      for(int_bbd p=colj[j]; p<colj[j+1]; p++) {
            int ridx = rowi[p];
            new_A->add_entry(ridx, j, val[p]);
          }
      }

      new_A->compact();

      clear();
      return new_A;
  }

  void to_lower(){

    nz_max = nnz;   // new nnz must be less than nnz/2
    int_bbd *new_colj = new int_bbd[ncol+1];
    int_bbd *new_rowi = new int_bbd[nz_max];
    float_bbd *new_val = new float_bbd[nz_max];

    int_bbd new_nnz = 0;
    new_colj[0] = 0;
    for(int_bbd j=0; j<ncol; j++){
      for(int_bbd p=colj[j]; p<colj[j+1]; p++){
          int_bbd ridx = rowi[p];
          if(ridx >= j){
            new_rowi[new_nnz] = ridx;
            new_val[new_nnz++] = val[p];
          }
      }
      new_colj[j+1] = new_nnz;
    }

    clear();

    is_symmetric = true;
    nnz = new_nnz;
    colj = new_colj;
    rowi = new_rowi;
    val = new_val;
    
  }


  SparseMatrix *to_csc() {

    int_bbd *csc_colj = (int *)malloc(sizeof(int)*(ncol+1));
    int_bbd *csc_rowi = (int *)malloc(sizeof(int)*(nnz));
    float_bbd *csc_val = (double *)malloc(sizeof(double)*nnz);

    int info = 0;
    int job[6] = {0, // if job(1)=1, the matrix in the CSC format is converted to the CSR format.
                     0,//If job(2)=0, zero-based indexing for the matrix in CSR format is used;
                     0,//If job(3)=0, zero-based indexing for the matrix in the CSC format is used;
                     0,
                     nnz,
                     1 //If job(6)≠0, all output arrays acsc, ja1, and ia1 are filled in for the output storage.
                     };


    mkl_dcsrcsc(job, &nrow, csc_val, csc_rowi, csc_colj, val, rowi, colj, &info);

    if(info !=0){
        printf("Fail to convert CSR to CSC\n");
        exit(-1);
    }

    printf("CSR -> CSC Done!\n"); 

    SparseMatrix *A_csc = new SparseMatrix(nrow, ncol, nnz, csc_colj, csc_rowi, csc_val);
    return A_csc;
  }

  SparseMatrix *to_csr() {

    // CSC => CSR
    int_bbd *csr_rowi = (int *)malloc(sizeof(int)*(nrow+1));
    int_bbd *csr_colj = (int *)malloc(sizeof(int)*nnz);
    float_bbd *csr_val = (double *)malloc(sizeof(double)*nnz);

    int info=0;
    int job[6] = {1, // if job(1)=1, the matrix in the CSC format is converted to the CSR format.
                     0,//If job(2)=0, zero-based indexing for the matrix in CSR format is used;
                     0,//If job(3)=0, zero-based indexing for the matrix in the CSC format is used;
                     0,
                     nnz,
                     1 //If job(6)≠0, all output arrays acsc, ja1, and ia1 are filled in for the output storage.
                     };


    mkl_dcsrcsc(job, &nrow, csr_val, csr_colj, csr_rowi, val, rowi, colj, &info);

    if(info !=0){
        printf("Fail to convert CSC to CSR\n");
        exit(-1);
    }

    printf("CSC -> CSR Done!\n"); 


    SparseMatrix *A_csr = new SparseMatrix(nrow, ncol, nnz, csr_colj, csr_rowi, csr_val);

    // delete []val;
    // delete []rowi;
    // delete []colj;

    // val = csr_val;
    // rowi = csr_rowi;
    // colj = csr_colj;

    // nz_max = nnz;

    return A_csr;

  }

  /************ free space ************/
  ~SparseMatrix(){
    // clear();
  }

};



    
#endif


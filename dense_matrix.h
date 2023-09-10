/**
author: modified by dztu
date: Sep 30, 2022
version: v2.0
**/

#ifndef _DENSEMATRIX_H_
#define _DENSEMATRIX_H_

#include <map>
#include <vector>
#include <assert.h>
#include <stdio.h>
#include "./bbd_type.h"
#include "./extlibs/gmres/vector.h" // added by cc
#include "sparse_matrix.h"

class DenseMatrix {
private:
  bool is_symmetric;    // if it is a symmetric matrix, we only save low triangle for symmetrix matrices

  // linearized data in compact format
  int_bbd nrow;
  int_bbd ncol;
  int_bbd val_size;
  std::vector <float_bbd> val;

public:
  DenseMatrix();
  DenseMatrix(size_t nrow_, size_t ncol_, std::vector <float_bbd> &val_);
  DenseMatrix(size_t nrow_, size_t ncol_, float_bbd *val_);
  


  /************ intialization ************/
  void init(int_bbd m, int_bbd n, bool iss);
  void add_entry(int_bbd row, int_bbd col, double d);   // add an element to matrix
  void clear();


  /************ format conversion ************/
  SparseMatrix *to_sparse(){

    int_bbd *colj = new int_bbd[nrow+1];
    int_bbd *rowi = new int_bbd[val_size];

    colj[0] = 0;
    for(int j=1; j<(ncol+1); j++){
      colj[j] = j*ncol;
    }

    for(int j=0; j<ncol; j++){
      for(int i=0; i<nrow; i++){
          rowi[j*nrow+i] = i;
      }
    }

    SparseMatrix *sp_mtx = new SparseMatrix(nrow, ncol, val_size, colj, rowi, val.data());
    return sp_mtx;

  }


  /************ factorize ************/
  // use MKL BLAS or MKL LAPACK
  bool factor(bool save_original);
  bool solve(std::vector<double> &r, std::vector<double> &x);


  /************ BLAS ************/
  bool multiply_add(std::vector<double> &x, std::vector<double> &y);   // y += A * x
  bool multiply_sub(std::vector<double> &x, std::vector<double> &y);   // y -= A * x

  /************ save matrix ************/
  void save_file_matlab (char filename[]);
  void save_file_coo (char filename[]);


  /************ free space ************/
  ~DenseMatrix() {
    clear();
  }


  // added by cc
  /************ print the matrix ************/
  void print_matrix();

  // added by cc
  /************ set the entry of matrix ************/
  void set_entry(int row, int col, float_bbd d);

  // added by cc
  /************ get the entry of matrix ************/
  float_bbd get_entry(int row, int col) const;

  // added by cc
  /************ get the ith column of matrix ************/
  Vector get_colomn(int column_index) const;

  // added by cc
  /************ matrix multiply vector ************/
  Vector operator*(const Vector &x) const;

  // added by cc 
  int_bbd get_nrow() {
    return nrow;
  }

};

#endif

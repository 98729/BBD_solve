/**
author: dztu
date: Sep 30, 2022
version: v2.0
**/

#include "dense_matrix.h"
#include <iostream> // added by cc

DenseMatrix::DenseMatrix()
{}

DenseMatrix::DenseMatrix(size_t nrow_, size_t ncol_, std::vector <float_bbd> &val_)
{
  nrow = nrow_;
  ncol = ncol_;

  val = val_;
  val_size = val.size();
}

DenseMatrix::DenseMatrix(size_t nrow_, size_t ncol_, float_bbd *val_)
{
  nrow = nrow_;
  ncol = ncol_;
  val_size = nrow * ncol;
  val.resize(val_size);
  for(int i=0; i<val_size; i++){
    val[i] = val_[i];
  }

}


void DenseMatrix::init(int m, int n, bool iss=true)
{
    is_symmetric = iss;
    nrow = m;
    ncol = n;
    val_size = m * n;
    if (is_symmetric) {
      assert(m == n);
      val_size = m * (m + 1) / 2;
    }
    val.resize(val_size, 0.0);
}


void DenseMatrix::add_entry(int row, int col, double d)
{
    if (is_symmetric) {
      if (col > row)  
        return;
      else {
        int n1 = nrow;
        int n2 = nrow - col + 1;
        int offset = (n1 + n2) * col / 2 + row - col;
        val[offset] += d;
      }
    }
    else
      val[col * nrow + row] += d;
}

void DenseMatrix::clear()
{
    val.clear();
    val.resize(val_size, 0.0);
}


  // use MKL BLAS or MKL LAPACK 
bool DenseMatrix::factor(bool save_original = false) 
{
    return true;
}

  // use MKL BLAS or MKL LAPACK 
bool DenseMatrix::solve(std::vector<double> &r, std::vector<double> &x)
{
    return true;
}

// y += A * x
bool DenseMatrix::multiply_add(std::vector<double> &x, std::vector<double> &y)
{
    // use blas
    return true;
}

// y -= A * x
bool DenseMatrix::multiply_sub(std::vector<double> &x, std::vector<double> &y)
{
    // use blas
    return true;
}

// 1-based index
void DenseMatrix::save_file_matlab (char filename[]) 
{
    FILE *fp;
    fp = fopen(filename, "w");

    if (!is_symmetric) {
      for (int j=0; j<ncol; j++) {
        for (int i=0; i<nrow; i++) 
          fprintf(fp, "A(%d, %d) = %g\n", i+1, j+1, val[j*nrow+i]);
      }
    }
    else {
      int_bbd idx = 0;
      for (int_bbd j=0; j<ncol; j++) {
        for (int_bbd i=j; i<nrow; i++)
          fprintf(fp, "A(%d, %d) = %g\n", i+1, j+1, val[idx++]); 
      }
    }

    fclose(fp);
}

// 0-based index
void DenseMatrix::save_file_coo(char filename[])
{
    FILE *fp;
    fp = fopen(filename, "w");

    fprintf(fp, "%d, %d, %d\n", nrow, ncol, nrow*ncol);

    if (!is_symmetric) {
      for (int j=0; j<ncol; j++) {
        for(int i=0; i<nrow; i++){
                fprintf(fp, "%d, %d, %g\n", i, j,  val[j*nrow+i]);
            }
        }
      }
    else {
        int idx = 0;
        for (int j=0; j<ncol; j++) {
          for (int i=j; i<nrow; i++){
            fprintf(fp, "A(%d, %d) = %g\n", i, j, val[idx++]);
          }
        }
      }

    fclose(fp);
}



// added by cc
/************ print the matrix ************/
void DenseMatrix::print_matrix()
{
  for (int rowi=0; rowi<nrow; rowi++){
    for (int colj=0; colj<ncol; colj++){
      std::cout << val[colj*nrow+rowi] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// added by cc
/************ set the entry of matrix ************/
void DenseMatrix::set_entry(int row, int col, float_bbd d)
{
  if (is_symmetric) {
    if (col > row)  
      return;
    else {
      int n1 = nrow;
      int n2 = nrow - col + 1;
      int offset = (n1 + n2) * col / 2 + row - col;
      val[offset] = d;
    }
  }
  else
    val[col * nrow + row] = d;
}

// added by cc
/************ get the entry of matrix ************/
float_bbd DenseMatrix::get_entry(int row, int col) const
{
  return val[col * nrow + row];
}

// added by cc
/************ get the ith column of matrix ************/
Vector DenseMatrix::get_colomn(int column_index) const
{
  Vector tmp = Vector(nrow);
  int index = 0;
  int start_idx = column_index*nrow;
  int end_idx = start_idx+nrow;
  for (int i=start_idx; i<end_idx; i++) {
    tmp.set_entry(index, val[i]);
    index++;
  }
  return tmp;
}

// added by cc
/************ matrix multiply vector ************/
Vector DenseMatrix::operator*(const Vector &x) const
{
  Vector tmp = Vector(nrow);
  for (int i=0; i<ncol; i++) {
    tmp += get_colomn(i) * x.get_entry(i);
  }
  return tmp;
}
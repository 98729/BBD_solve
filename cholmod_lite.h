/*******************************************************
* author: Xiaolve Lai, modified by dztu
* date: Oct. 07, 2022
* version: v2.0
* function: a simple solver for PD symmetric matrices with NO reordering NO permutation
*******************************************************/


#ifndef _CHOLMOD_LITE_H_
#define _CHOLMOD_LITE_H_


#include <vector>
#include <set>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <memory.h>
#include "sparse_matrix.h"

class CholLite {
private:
  int size;
  int decompose_size;
  int nnz;
  int nnz_orig;
  int nnz_schur;
  int *col_idx;
  int *row_idx;
  double *data;

  int current_alloc;
  int *row_row_idx;     // compacted row major pattern
  int *row_col_idx;

  void realloc(int i) {
    int s = col_idx[i] + size;
    if (s >= current_alloc) {
      int new_alloc = current_alloc * 2;
      int *tmp = new int[new_alloc];
      memcpy(tmp, row_idx, current_alloc * sizeof(int));
      delete[] row_idx;
      row_idx = tmp;
      current_alloc = new_alloc;
    }
  }

  // you can make r==x for inplace solve
  bool l_solve(double *x, double *r) {
    if (r != x) 
      memcpy(x, r, decompose_size * sizeof(double));
    
    x[0] = r[0] / data[0];             // 1st row
    for (int i=1; i<decompose_size; i++) {
      int start = col_idx[i-1] + 1;    // skip the diagonal entry
      int stop  = col_idx[i];
      double d = x[i-1];
      for (int j=start; j<stop; j++) {
        int row = row_idx[j];
        if (row >= decompose_size)
          break;
        x[row] -= d * data[j];
      }
      x[i] /= data[col_idx[i]];
    }
    return true;
  }

  // you can make r==x for inplace solve
  bool u_solve(double *x, double *r) {
    x[decompose_size-1] = r[decompose_size-1] / data[col_idx[decompose_size-1]];  // last row
    for (int i=decompose_size-2; i>=0; i--) {
      int start = col_idx[i] + 1;
      int stop  = col_idx[i+1];
      double d = r[i];
      for (int j=start; j<stop; j++) {
        int row = row_idx[j];
        if (row >= decompose_size)
          break;
        d -= x[row] * data[j];
      }
      x[i] = d / data[col_idx[i]];
    }
    
    return true;
  }

  // y -= CU^-1 * x,  for rhs compression
  bool cmult_sub(double *y, double *x) {
    for (int i=0; i<decompose_size; i++) {
      int start = col_idx[i];
      int stop  = col_idx[i+1];
      double d = x[i];
      for (int j=stop-1; j>=start; j--) {
        int row = row_idx[j];
        if (row < decompose_size)
          break;
        y[row] -= d * data[j];
      }
    }
    return true;
  }

  // y -= L^-1C * x,  for solution propagation
  bool c_transpose_mult_sub(double *y, double *x) {
    for (int i=0; i<decompose_size; i++) {
      int start = col_idx[i];
      int stop  = col_idx[i+1];
      for (int j=stop-1; j>=start; j--) {
        int row = row_idx[j];
        if (row < decompose_size)
          break;
        y[i] -= data[j] * x[row]; 
      } 
    }
    return true;   
  } 


  
public:

  CholLite() {
    col_idx = NULL;
    row_idx = NULL;
    data = NULL;
    row_row_idx = NULL;
    row_col_idx = NULL;
  }


  int_bbd get_nrow(){
    return size;
  }

  int_bbd get_ncol(){
    return size;
  }

  int_bbd get_nnz(){
    return nnz;
  }

  int_bbd *get_colj(){
    return col_idx;
  }

  int_bbd *get_rowi(){
    return row_idx;
  }
 
  float_bbd *get_value(){
    return data;
  }


  bool symbolic(int size_, int ds, int *cidx, int *ridx) {
    size = size_;
    decompose_size = ds;
    clear(); 
    col_idx = new int[size+1];
    nnz_orig = cidx[size];
    current_alloc = nnz_orig * 5;
    row_idx = new int[current_alloc];           // assume 4x fillins, we don't like reallocation

    std::vector <std::vector<int>> idx_tran;    // row major pattern
    idx_tran.resize(size);
    std::vector <bool> visited(size, false);    // true if a row was stamped
    std::vector <int> cur_pos(size);
    col_idx[0] = 0;

    int *cur_col;
    int  csize;

    for (int i=0; i<decompose_size; i++) {
      realloc(i);                               // make sure there has enough memory for a new column
      std::vector<int> &tidx = idx_tran[i];     // transposed matrix pattern, row compressed format
      cur_col = row_idx + col_idx[i];
      csize = 0;
      cur_pos[i] = col_idx[i] + 1;

      // collect original elements
      int start = cidx[i];
      int stop  = cidx[i+1];
      for (int j=start; j<stop; j++) {
        cur_col[csize++] = ridx[j];             // add rows for column i
        idx_tran[ridx[j]].push_back(i);         // add columns to transposed pattern, to rows
        visited[ridx[j]] = true;
      }

      // create fillins
      for (size_t j=0; j<tidx.size(); j++) {
        int cc = tidx[j];
        if (cc < i) {
          start = cur_pos[cc];
          stop  = col_idx[cc+1];
          for (int k=start; k<stop; k++) {
            int rr = row_idx[k];
            if (visited[rr]) 
              continue;
            cur_col[csize++] = rr;
            idx_tran[rr].push_back(i);
            visited[rr] = true;
          }
          cur_pos[cc]++;
        }
      }

      std::sort(cur_col, cur_col+csize);
      col_idx[i+1] = col_idx[i] + csize;
      for (int k=0; k<csize; k++)
        visited[cur_col[k]] = false; 
    }
    
    // TODO: create schur pattern
    nnz_schur = 0;
    for (int i=decompose_size; i<size; i++) {
      realloc(i);
      std::vector<int> &tidx = idx_tran[i];  // transposed matrix pattern, row compressed format
      cur_col = row_idx + col_idx[i];
      csize = 0;

      // collect original elements
      int start = cidx[i];
      int stop  = cidx[i+1];
      for (int j=start; j<stop; j++) {
        cur_col[csize++] = ridx[j];             // add rows for column i
        visited[ridx[j]] = true;
      }

      for (size_t j=0; j<tidx.size(); j++) {
        int cc = tidx[j];
        assert(cc < decompose_size);
//        if (cc < decompose_size) {
          start = cur_pos[cc]; //col_idx[cc];
          stop  = col_idx[cc+1];
          for (int k=start; k<stop; k++) {
            int rr = row_idx[k];
            if (!visited[rr]) {
              cur_col[csize++] = rr;
              visited[rr] = true;
            }
          }
//        }
        cur_pos[cc]++;
      }
      std::sort(cur_col, cur_col+csize);
      col_idx[i+1] = col_idx[i] + csize;
      for (int k=0; k<csize; k++)
        visited[cur_col[k]] = false;
      nnz_schur += csize;
//      printf("csize = %d\n", csize);
    }

    nnz = col_idx[size];
    printf("original nnz=%d, new nnz=%d, schur size=%d, schur nnz=%d\n", nnz_orig, nnz, size-decompose_size, nnz_schur);

    // TODO: clean up memory, truncate the col_idx to size.
    int *tmp = new int[nnz];
    for (int i=0; i<nnz; i++)
      tmp[i] = row_idx[i];
    delete []row_idx;
    row_idx = tmp;
    data = new double[nnz];

    // compact row major pattern
    row_row_idx = new int[size+1];
    row_col_idx = new int[nnz];
    row_row_idx[0] = 0;
    for (int i=0; i<size; i++) {
      std::vector<int> &tidx = idx_tran[i];  // transposed matrix pattern, row compressed format
      std::sort(tidx.begin(), tidx.end());
      int start = row_row_idx[i];
      unsigned  k = 0;
      for (k=0; k<tidx.size(); k++ ) 
        row_col_idx[start+k] = tidx[k];
      row_row_idx[i+1] = row_row_idx[i] + k;
    }
//    assert(row_row_idx[size] == nnz);

    return true;
  }

  bool refactor(int *cidx, int *ridx, double *data_) {
    std::vector <double> tmp(size, 0.0);          // column buffer 
    std::vector <int> cur_pos(size);              // current row position
    int start, stop, start1, stop1;

    for (int i=0; i<decompose_size; i++) {
      
      cur_pos[i] = col_idx[i] + 1;

      // copy original data to tmp 
      start = cidx[i];
      stop  = cidx[i+1];
      for (int k=start; k<stop; k++) 
        tmp[ridx[k]] = data_[k];

      start = row_row_idx[i];
      stop  = row_row_idx[i+1] - 1;   // the last one is on diagonal, skip it
      for (int k=start; k<stop; k++) {
        int col = row_col_idx[k];
        double d = data[cur_pos[col]];
        start1 = cur_pos[col];
        stop1 = col_idx[col+1];
        for (int l=start1; l<stop1; l++) {
          tmp[row_idx[l]] -= data[l] * d;
        }
        cur_pos[col]++;
      }

      double diag = tmp[i];
      if(diag <= 0.0) {
        printf("the matrix is not PD (diag=%g, column=%d)!!\n", diag, i);
        return false;
      }

      diag = 1.0 / sqrt(diag);
      start = col_idx[i];
      stop  = col_idx[i+1];
      for (int k=start; k<stop; k++) {
        data[k] = tmp[row_idx[k]] * diag;
        tmp[row_idx[k]] = 0.0;
      }
    }

    //schur
    for (int i=decompose_size; i<size; i++) {
      // copy original data to tmp 
      start = cidx[i];
      stop  = cidx[i+1];
      for (int k=start; k<stop; k++)
        tmp[ridx[k]] = data_[k];

      start = row_row_idx[i];
      stop  = row_row_idx[i+1];
      for (int k=start; k<stop; k++) {
        int col = row_col_idx[k];
        assert(col < decompose_size);
//        if (col >= decompose_size) {
//          printf("row=%d, col=%d\n", i, col);
//          break;
//        }
        double d = data[cur_pos[col]];
        for (int l=cur_pos[col]; l<col_idx[col+1]; l++) {
          tmp[row_idx[l]] -= data[l] * d;
        }
        cur_pos[col]++;
      }

      start = col_idx[i];
      stop  = col_idx[i+1];
      for (int k=start; k<stop; k++) {
        data[k] = tmp[row_idx[k]];
        tmp[row_idx[k]] = 0.0;
      }
    }

    delete[] row_row_idx;
    row_row_idx = NULL;
    delete[] row_col_idx;
    row_col_idx = NULL;

    return true;
  }

  bool full_factor(int size_, int ds, int *cidx, int *ridx, double *data) {
    symbolic(size_, ds, cidx, ridx);
    refactor(cidx, ridx, data);
    return true;
  }

  bool solve (std::vector <double> &x, std::vector <double> &r) {
    l_solve(&x[0], &r[0]);    
    u_solve(&x[0], &x[0]);    
    return true;
  }

  // rhs_schur = rhs_schur - C * U^-1 * L^-1 * r_internal
  bool compress_rhs(std::vector <double> &r, std::vector <double> &x) {
    if (r.empty())
      return false;
    l_solve(&x[0], &r[0]);  
    cmult_sub(&r[0], &x[0]);
    return true;
  }

  // x_internal = U^-1 (L^-1 r_internal - L^-1 * C * x_schur)  
  bool propagate_sol(std::vector <double> &x) {
    if (x.empty())
      return false;
    c_transpose_mult_sub(&x[0], &x[0]);
    u_solve(&x[0], &x[0]);
    return true;
  }

  void clear() {
    if (col_idx)
      delete[] col_idx;
    if (row_idx)
      delete row_idx;
    if (row_col_idx)
      delete[] row_col_idx;
    if (row_row_idx)
      delete row_row_idx;
     if (data)
      delete data;
  }

  ~CholLite() {
    clear();
  }

  // collect schur to higher level, we should merge this function to refactor()
  void collect_schur(SparseMatrix &schur, std::vector <int> &port_map) {
    for (int col=decompose_size; col<size; col++) {
      int start = col_idx[col];
      int stop  = col_idx[col+1];
      for (int j=start; j<stop; j++) {
        int row = row_idx[j];
        if (row < decompose_size)
          continue;
        int col2 = port_map[col-decompose_size];   // we need to understand mapping step
        int row2 = port_map[row-decompose_size];
        if (col2 < row2)
          schur.add_entry(row2, col2, data[j]);
        else
          schur.add_entry(col2, row2, data[j]);
      }
    }
  }

  // print matrix in to a matlab file
  void print (FILE *fh = stdout) {
    fprintf(fh, "L = zeros(%d, %d);\n", size, size);
    for (int i=0; i<size; i++) {
      for (int j=col_idx[i]; j<col_idx[i+1]; j++) {
        fprintf(fh, "L(%d, %d) = %g;\n", row_idx[j]+1, i+1, data[j]);
      }
    }
  }

};

#endif

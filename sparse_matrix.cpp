/**
author: dztu
date: Oct. 7, 2022
version: v3.0
**/

#include "sparse_matrix.h"
#include "mkl.h"



SparseMatrix::SparseMatrix()
{
    nrow = 0; ncol = 0;
    nnz = 0;      // equal to colj[nrow]
    nz_max = 0;   // maximum allocated memory 

    colj = NULL; rowi = NULL; val = NULL;

    val_map = NULL;
    is_symmetric = false;
}

SparseMatrix::SparseMatrix(cs *cs_mtx)
{
    nrow = cs_mtx->m;
    ncol = cs_mtx->n;
    nnz = cs_mtx->p[cs_mtx->n];
    nz_max = nnz;

    colj = cs_mtx->p;
    rowi = cs_mtx->i;
    val = cs_mtx->x;

    val_map = NULL;
    is_symmetric = false;
}

// we assume the matrix is dense, but we need to store it in sparse formate
SparseMatrix::SparseMatrix(int_bbd nrow_, int_bbd ncol_, float_bbd *val_, bool trans)
{
    nrow = nrow_;
    ncol = ncol_;
    nnz = nrow * ncol;
    nz_max = nnz;

    if(trans){
      // switch row-major layout to col-major layout
        mkl_dimatcopy ('R',   // row-major layout
                      'T',    // transpose
                      nrow, 
                      ncol,  
                      1.0,    // scale input matrix
                      val_, 
                      ncol, 
                      nrow);
    }

    val = val_;

    colj = new int_bbd[ncol+1];
    rowi = new int_bbd[nnz];

    colj[0] = 0;
    for(int j=1; j<(ncol+1); j++) {
      colj[j] = j*nrow;
      // printf("col[%d] = %d\n", j, j*nrow);
    }

    for(int j=0; j<ncol; j++) {
      for(int i=0; i<nrow; i++){
          rowi[j*nrow+i] = i;
      }
    }


    val_map = NULL;
    is_symmetric = false;

}
SparseMatrix::SparseMatrix(int_bbd nrow_, int_bbd ncol_, int_bbd nnz_)
{
    nrow = nrow_;
    ncol = ncol_;
    nz_max = nnz_ > 1 ? nnz_ : 1;
    nnz = nnz_;
    
    colj = new int_bbd[ncol+1];
    rowi = new int_bbd[nz_max];
    val = new float_bbd[nz_max];

    val_map = NULL;
    is_symmetric = false;
}

SparseMatrix::SparseMatrix(int_bbd nrow_, int_bbd ncol_, int_bbd nnz_, 
                            int_bbd *p, int_bbd *i, float_bbd *data)
{
    nrow = nrow_;
    ncol = ncol_;
    nnz = nnz_;
    nz_max = nnz_;

    colj = p;
    rowi = i;
    val = data;

    val_map = NULL;
    is_symmetric = false;

    // assert(nnz == colj[ncol]);
}

SparseMatrix::SparseMatrix(int_bbd nrow_, int_bbd ncol_)
{
    nrow = nrow_;
    ncol = ncol_;
    nnz = 0;

    // default sparsity 50%
    int_bbd tmp_nz = int_bbd(nrow * ncol * 0.2)>100 ? int_bbd(nrow * ncol * 0.1): 100;  // TODO, modify 0.5 to suitable value
    // int_bbd tmp_nz = 100;

    colj = new int_bbd[ncol+1];
    rowi = new int_bbd[tmp_nz];
    val = new float_bbd[tmp_nz];

    nz_max = tmp_nz;
    val_map = NULL;
    is_symmetric = false;
}


bool SparseMatrix::realloc(int_bbd j) {

  int_bbd s = colj[j] + nrow;
  if (s >= nz_max) {
    size_t new_alloc = nz_max * 2;

    int_bbd *tmp_rowi = new int_bbd[new_alloc];
    float_bbd *tmp_val = new float_bbd[new_alloc];

    memcpy(tmp_rowi, rowi, nz_max * sizeof(int_bbd));
    memcpy(tmp_val, val, nz_max * sizeof(float_bbd));

    delete[] rowi;
    delete[] val;
    
    rowi = tmp_rowi;
    val = tmp_val;

    nz_max = new_alloc;

    return true;
  }
  return false;

}


void SparseMatrix::update(int_bbd nrow_, int_bbd ncol_, int_bbd nnz_, int_bbd *p, int_bbd *i, float_bbd *data){

    clear();

    nrow = nrow_;
    ncol = ncol_;
    nnz = nnz_;
    nz_max = nnz_;

    colj = p;
    rowi = i;
    val = data;

    val_map = NULL;

}


void SparseMatrix::init(int_bbd m, int_bbd n, bool is_sym)
{
    if (colj)
      delete [] colj;
    if (rowi)
      delete [] rowi;
    if (val)
      delete [] val;
    if (val_map)
      delete val_map;

    is_symmetric = is_sym;

    nrow = m;
    ncol = n;
    nnz = 0;
    nz_max = 0;

    colj = NULL;
    rowi = NULL;
    val = NULL;

    val_map = new std::vector <std::map<int_bbd, float_bbd>>;
    val_map->resize(ncol);
}

void SparseMatrix::clear()
{
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
    if (val_map) {
      delete val_map;
      val_map = NULL;
    }
}


void SparseMatrix::add_entry(int_bbd row, int_bbd col, float_bbd d)
{
    if (is_symmetric && col > row)  
      return;
    (*val_map)[col][row] += d;
}


void SparseMatrix::compact() 
{
    std::map<int, float_bbd>::iterator it;
    nnz = 0;
    for (int i=0; i<ncol; i++)
      nnz += (*val_map)[i].size();

    colj = new int[ncol+1];
    rowi = new int[nnz];
    val = new float_bbd[nnz];
    
    int idx = 0;
    colj[0] = 0;
    for (int i=0; i<ncol; i++) {
      for (it = (*val_map)[i].begin(); it != (*val_map)[i].end(); ++it) {
        rowi[idx] = it->first;
        val[idx++] = it->second;
      }
      colj[i+1] = idx;
    }
    assert(idx == nnz);

    delete val_map;
    val_map = NULL;
}



float_bbd SparseMatrix::get_density()
{
    if (nnz > 0) {
      return nnz / (1.0 * nrow * ncol);
    }
    return 0.0;
}


void SparseMatrix::sparsify(SparseMatrix &tgt, float_bbd tol) {
    for (int i=0; i<ncol; i++) {
      int start = colj[i];
      int stop  = colj[i+1];

      float_bbd diag = val[start];
      tgt.add_entry(i,i,diag);

      for (int j=start+1; j<stop; j++) {
        int row  = rowi[j];
        float_bbd d = val[j];
        if (fabs(d) > diag * tol) {
          tgt.add_entry(row, i, d);
        }
        else {
          tgt.add_entry(i, i, d);
          tgt.add_entry(row, row, d);
        }
      }
    }
    tgt.compact();
  }



// 1-based index in MATLAB format
void SparseMatrix::save_file_matlab(const char filename[]) 
{
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "A = zeros(%d, %d);\n", nrow, ncol);
    for(int j=0; j<ncol; j++){
        for(int i=colj[j]; i<colj[j+1]; i++){
            fprintf(fp, "A(%d, %d)=%g;\n", rowi[i]+1, j+1, val[i]);
        }
    }
    fclose(fp);
}



// 1-based index in MatrixMarket format
void SparseMatrix::save_file_coo(const char filename[]) {
    FILE *fp;
    fp = fopen(filename, "w");

    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%%-------------------------------------------------------------------------------\n");
    fprintf(fp, "%% name: test sparse matrix\n");
    fprintf(fp, "%% kind: circuit simulation problem\n");
    fprintf(fp, "%%-------------------------------------------------------------------------------\n");
    fprintf(fp, "%d %d %d\n", nrow, ncol, nnz);

    for(int j=0; j<ncol; j++){
        for(int i=colj[j]; i<colj[j+1]; i++){
            fprintf(fp, "%d %d %g\n", rowi[i]+1, j+1, val[i]);
        }
    }
    fclose(fp);
}
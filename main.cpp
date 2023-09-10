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
#include "./bbd_seq_cholmod_lite_wrapper.h"
#include "klu_solver_wrapper.h"
#include "nicslu_solver_wrapper.h"
#include "bbd_seq_klu_solver_wrapper.h"
#include "bbd_parallel_klu_solver_wrapper.h"
#include "./extlibs/nicslu/nicslu202110/include/nicslu.h"

using namespace std;
// Debug flag

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);
  int comm_rank, comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  const int root = 0;

  Timer2 total_timer, tmp_timer;
  total_timer.start();

  SparseMatrix *A_org, *BU_matrix, *Ch_matrix; // BxU, C/h
  std::vector<int_bbd> node_index;             // required node for output
  std::vector<float_bbd> time_step;            // required simluation time steps
  float_bbd h_step;                            // required time step, h = time_step[1] - time_step[0];

  /**************** CMD line parser ****************/
  Params *params;
  params = parse_cmdline(argc, argv);

  /**************** read & transform sparse matrix ****************/
  tmp_timer.start();

  if (comm_rank == 0)
    printf("==============Reading Matrix==============\n");
  if (params->mtx_filepath != NULL)
  {

    A_org = read_mtx_matrix(params->mtx_filepath);

    // A_org = new SparseMatrix();
    // A_org->init_rand_matrix(5000, 5000, false);
    // A_org->save_file_coo("mytest.mtx");

    // attention: only for runtime test
    A_org = A_org->to_SPD();


    BU_matrix = new SparseMatrix();
    Ch_matrix = new SparseMatrix();

    BU_matrix->init_rand_matrix(A_org->get_nrow(), 401, false);
    Ch_matrix->init_rand_matrix(A_org->get_nrow(), A_org->get_ncol(), false);
  }
  else
  {

    SparseMatrix *G_matrix, *C_matrix, *B_matrix, *U_matrix;

    // use openmp read matrix,  4-thread reading
    string g_filepath = string(params->dirname) + "/CSC_G.txt";
    G_matrix = read_one_matrix(g_filepath.c_str());

    string c_filepath = string(params->dirname) + "/CSC_C.txt";
    C_matrix = read_one_matrix(c_filepath.c_str());

    string b_filepath = string(params->dirname) + "/CSC_B.txt";
    B_matrix = read_one_matrix(b_filepath.c_str());

    string u_filepath = string(params->dirname) + "/u_t.txt";
    U_matrix = read_u_matrix(u_filepath.c_str(), node_index, time_step);

    printf("G matrix size: %d x %d, nnz: %d\n", G_matrix->get_nrow(), G_matrix->get_ncol(), G_matrix->get_nnz());
    printf("C matrix size: %d x %d, nnz: %d\n", C_matrix->get_nrow(), C_matrix->get_ncol(), C_matrix->get_nnz());
    printf("B matrix size: %d x %d, nnz: %d\n", B_matrix->get_nrow(), B_matrix->get_ncol(), B_matrix->get_nnz());
    printf("U matrix size: %d x %d, nnz: %d\n", U_matrix->get_nrow(), U_matrix->get_ncol(), U_matrix->get_nnz());

    printf("# of test nodes: %d \n", node_index.size());
    printf("# of time steps: %d \n", time_step.size());

    tmp_timer.elapsedWallclockTime(params->read_time);
    printf("==============Reading Matrix Done, Runtime: %g ms==============\n", params->read_time);

    printf("==============Stamping Matrix Start==============\n");
    tmp_timer.start();

    h_step = time_step[1] - time_step[0];
    printf("time step h: %g \n", h_step);

    if (params->tran_method == 0)
    {
      A_org = bbd_mm_add(G_matrix, C_matrix, 1.0, 1 / h_step, CSPARSE_BLAS); // G + C / h
      BU_matrix = bbd_mm_mult_add(B_matrix, U_matrix, 1, CSPARSE_BLAS);      // B x U
      Ch_matrix = bbd_m_scale(C_matrix, 1 / h_step);                         // C/h        shallow copy
    }
    if (params->tran_method == 1)
    {
      A_org = bbd_mm_add(G_matrix, C_matrix, 1.0, 2 / h_step, CSPARSE_BLAS);      // G + 2 * C / h
      BU_matrix = bbd_mm_mult_add(B_matrix, U_matrix, 1, CSPARSE_BLAS);           // B x U
      Ch_matrix = bbd_mm_add(G_matrix, C_matrix, -1.0, 2 / h_step, CSPARSE_BLAS); // -G + 2 * C / h
    }


    // attention: only for runtime test
    A_org = A_org->to_SPD();


    printf("G+C/h matrix size: %d x %d, nnz: %d\n", A_org->get_nrow(), A_org->get_ncol(), A_org->get_nnz());
    printf("BxU matrix size: %d x %d, nnz: %d\n", BU_matrix->get_nrow(), BU_matrix->get_ncol(), BU_matrix->get_nnz());
    printf("C/h matrix size: %d x %d, nnz: %d\n", Ch_matrix->get_nrow(), Ch_matrix->get_ncol(), Ch_matrix->get_nnz());

    G_matrix->clear();
    // C_matrix->clear();  // attention: C matrix should not be cleared, since C/h is stored in C matrix
    B_matrix->clear();
    U_matrix->clear();

    tmp_timer.elapsedWallclockTime(params->stamp_time);
    printf("==============Stamping Matrix Done, Runtime: %g ms==============\n", params->stamp_time);
  }

  if (params->solver_type == -1)
  { // default
    // TODO: auto select solver according to historical statistics

    // SPD detection
    params->solver_type = SOLVER_TYPE_BBD_KLU;
  }

  if (params->solver_type == SOLVER_TYPE_KLU)
  {
    bbd_klu_solve(A_org, Ch_matrix, BU_matrix,
                  node_index, time_step, params);
  }
  else if (params->solver_type == SOLVER_TYPE_NICSLU)
  {
    bbd_nicslu_solve(A_org, Ch_matrix, BU_matrix,
                     node_index, time_step, params);
  }
  else if (params->solver_type == SOLVER_TYPE_BBD_KLU)
  {
    bbd_seq_klu_solve(A_org, Ch_matrix, BU_matrix,
                      node_index, time_step, params);
  }
  else if (params->solver_type == SOLVER_TYPE_PARALLEL_BBD_KLU)
  {
    bbd_parallel_klu_solve(A_org, Ch_matrix, BU_matrix,
                           node_index, time_step, params);
  }

  else if (params->solver_type == SOLVER_TYPE_CHOLMOD_LITE)
  {
    printf("Wrong solver, SOLVER_TYPE_CHOLMOD_LITE has been deleted!!!\n");
    exit(-1);
    // bbd_seq_cholmod_lite_solve (A_org, Ch_matrix, BU_matrix,
    //                         node_index, time_step, params);
  }

  total_timer.elapsedWallclockTime(params->total_time);

  if (comm_rank == 0)
    printf("==============!!!BBD Solver Success!!! Runtime: %g s==============\n", params->total_time / 1000.0);

  // TODO: memory leakage check
  // TODO: catch exception and solve

  MPI_Finalize();

  return 0;
}

// for future MPI scheduler
// std::vector<std::vector<int>> scheduler_map;
// std::vector<int> map0 = {1, 3, 4, 7, 7, 7};
// std::vector<int> map1 = {2, 5, 7, 7};
// std::vector<int> map2 = {6, 7, 8, 9, 7, 7, 7, 7};
// scheduler_map.push_back(map0);          // 0: master process -> process partition [1, 3, 4] with solver [7, 7, 7]
// scheduler_map.push_back(map1);          // 1: slave process -> process partition [2, 5] with solver [7, 7]
// scheduler_map.push_back(map2);          // 2: slave process -> process partition [6, 7, 8, 9] with solver [7, 7, 7, 7]

// create pipeline, refer to https://isocpp.org/files/papers/n3534.html

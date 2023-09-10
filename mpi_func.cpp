#include "mpi_func.h"

void gather_schur_rhs(
    float_bbd *schur_rhs_sum,
    float_bbd *schur_rhs_arr_single,
    int_bbd schur_size)
{
    int root = 0;
    MPI_Reduce(schur_rhs_arr_single, schur_rhs_sum, schur_size, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
}

void gather_schur_mtx(
    std::vector<SparseMatrix *> &schur_mtx_arr,
    SparseMatrix *schur_mtx_arr_single,
    int comm_rank, int comm_size)
{
    const int root = 0;
    int_bbd nnz = schur_mtx_arr_single->get_nnz();
    int_bbd ncol = schur_mtx_arr_single->get_ncol();
    int_bbd *colj = schur_mtx_arr_single->get_colj();   // ncol + 1
    int_bbd *rowi = schur_mtx_arr_single->get_rowi();   // nnz
    float_bbd *val = schur_mtx_arr_single->get_value(); // nnz

    int k, k1, k2, k3, k4;
    MPI_Pack_size(2, MPI_INT, MPI_COMM_WORLD, &k1); // pack nnz & ncol(int)
    MPI_Pack_size(ncol + 1, MPI_INT, MPI_COMM_WORLD, &k2);
    MPI_Pack_size(nnz, MPI_INT, MPI_COMM_WORLD, &k3);
    MPI_Pack_size(nnz, MPI_DOUBLE, MPI_COMM_WORLD, &k4);
    k = k1 + k2 + k3 + k4;
    char *packbuf = new char[k];
    int position = 0;
    MPI_Pack(&nnz, 1, MPI_INT, packbuf, k, &position, MPI_COMM_WORLD);
    MPI_Pack(&ncol, 1, MPI_INT, packbuf, k, &position, MPI_COMM_WORLD);
    MPI_Pack(colj, ncol + 1, MPI_INT, packbuf, k, &position, MPI_COMM_WORLD);
    MPI_Pack(rowi, nnz, MPI_INT, packbuf, k, &position, MPI_COMM_WORLD);
    MPI_Pack(val, nnz, MPI_DOUBLE, packbuf, k, &position, MPI_COMM_WORLD);

    if (comm_rank != 0)
    {
        MPI_Gather(&position, 1, MPI_INT, NULL, 0,
                   MPI_DATATYPE_NULL, root, MPI_COMM_WORLD);
        MPI_Gatherv(packbuf, position, MPI_PACKED, NULL,
                    NULL, NULL, MPI_DATATYPE_NULL, root, MPI_COMM_WORLD);
    }
    else
    {
        int counts[comm_size];
        MPI_Gather(&position, 1, MPI_INT, counts, 1,
                   MPI_INT, root, MPI_COMM_WORLD);
        int displs[comm_size];
        displs[0] = 0;
        for (int i = 1; i < comm_size; i++)
            displs[i] = displs[i - 1] + counts[i - 1];
        int totalcount = displs[comm_size - 1] + counts[comm_size - 1];
        char *recvBuf = new char[totalcount];
        MPI_Gatherv(packbuf, position, MPI_PACKED, recvBuf,
                    counts, displs, MPI_PACKED, root, MPI_COMM_WORLD);

        /***** unpack all *****/
        for (int i = 0; i < comm_size; i++)
        {
            position = 0;
            MPI_Unpack(recvBuf + displs[i], totalcount - displs[i],
                       &position, &nnz, 1, MPI_INT, MPI_COMM_WORLD);
            MPI_Unpack(recvBuf + displs[i], totalcount - displs[i],
                       &position, &ncol, 1, MPI_INT, MPI_COMM_WORLD);
            printf("rank 0 got nnz = %d from rank %d. ", nnz, i);
            schur_mtx_arr[i] = new SparseMatrix(ncol, ncol, nnz);
            printf("schur_mtx_arr[%d] has nnz = %d. ", i, schur_mtx_arr[i]->get_nnz());
            MPI_Unpack(recvBuf + displs[i], totalcount - displs[i],
                       &position, schur_mtx_arr[i]->get_colj(),
                       ncol + 1, MPI_INT, MPI_COMM_WORLD);
            MPI_Unpack(recvBuf + displs[i], totalcount - displs[i],
                       &position, schur_mtx_arr[i]->get_rowi(),
                       nnz, MPI_INT, MPI_COMM_WORLD);
            MPI_Unpack(recvBuf + displs[i], totalcount - displs[i],
                       &position, schur_mtx_arr[i]->get_value(),
                       nnz, MPI_DOUBLE, MPI_COMM_WORLD);
            printf("schur_mtx_arr[%d] has nnz = %d. ", i, schur_mtx_arr[i]->get_nnz());
            printf("now nnz = %d.\n", nnz);
        }
    }
}

void gather_xi(
    float_bbd *xi_arr,
    float_bbd *xi_arr_single, int *rhs_recvCount, int *displs,
    int sendnum)
{
    MPI_Gatherv(xi_arr_single, sendnum, MPI_DOUBLE, xi_arr,
                rhs_recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
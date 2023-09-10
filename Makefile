# CXX       := g++
CXX       := mpigxx
CXX_FLAGS := -std=c++17 -ggdb -g -fopenmp -Wall -w
RELEASE_CXX_FLAGS := -std=c++17 -o3 -fopenmp

BIN     := 
SRC     := ./utils/*.cpp ./utils/*.c
INCLUDE := -I/opt/intel/oneapi/mkl/2022.2.0/include 

# -I/usr/include/eigen3/Eigen


LIBRARIES  := -L/opt/intel/oneapi/mkl/2022.2.0/lib/intel64 -lmkl_rt \
			-L./extlibs/SuiteSparse/lib -lklu -lamd -lbtf -lcolamd -lcxsparse \
			-L./extlibs/SuiteSparse/SuiteSparse_config -lsuitesparseconfig \
			-L./extlibs/metis-5.1.0/build/Linux-x86_64/libmetis -lmetis \
			-L/usr/lib -lgfortran -lgomp -lpthread \
			-L./extlibs/gmres -lgmres \
			-L./extlibs/nicslu/nicslu_for_smash7.5.x/linux/lib_centos6_x64_gcc482_fma/ -lnicslu -lpthread -lm -ldl \
			-L./ -lhsl_mc64	
# -L ./extlibs/hsl_mc64-2.4.0/src/libs -lhsl_mc64


EXECUTABLE  := main

# enter BTF make 
# enter AMD make
# enter CHOLMOD make
# enter METIS make


all: clean
	$(CXX) $(CXX_FLAGS) $(INCLUDE) $(SRC) *.cpp -o $(EXECUTABLE) $(LIBRARIES)

release: clean
	$(CXX) $(RELEASE_CXX_FLAGS) $(INCLUDE) $(SRC) *.cpp -o $(EXECUTABLE) $(LIBRARIES)

run:
	./$(EXECUTABLE) -d matrix_case/matrix1 -p 5 -s 2

matrix1:
	./$(EXECUTABLE) -d matrix_case/matrix1 -p 5 -s 2

matrix2:
	./$(EXECUTABLE) -d matrix_case/matrix2 -p 5 -s 2

klu:
	./$(EXECUTABLE) -d matrix_case/matrix1 -p 5 -s 2

nicslu:
	./$(EXECUTABLE) -d matrix_case/matrix1 -p 5 -s 3

test:
	./$(EXECUTABLE) -m matrix_case/benchmark/oscil_trans_01.mtx -p 2 -s 6

bbdklu:
	./$(EXECUTABLE) -d matrix_case/matrix1 -p 5 -s 6


mpitest:
	/opt/intel/oneapi/mpi/2021.7.0/bin/mpirun -np 2 ./$(EXECUTABLE) -m ./matrix_part_0.mtx -p 4 -s 8


###################################### Performance Testing ##########################################
###################################### -p should be multiple of -np ##########################################
mpi_matrix1:
	/opt/intel/oneapi/mpi/2021.7.0/bin/mpirun -np 1 ./$(EXECUTABLE) -d matrix_case/matrix1 -p 2 -s 8

mpi_matrix2:
	/opt/intel/oneapi/mpi/2021.7.0/bin/mpirun -np 8 ./$(EXECUTABLE) -d matrix_case/matrix2 -p 64 -s 8

benchmark: all
	/opt/intel/oneapi/mpi/2021.7.0/bin/mpirun -np 1 ./$(EXECUTABLE) -m matrix_case/benchmark/rajat05.mtx -p 2 -s 8


# bbdklu:
# 	./$(EXECUTABLE) -d matrix_case/matrix1 -p 5 -s 6

clean:
	rm -rf *.o $(EXECUTABLE) *.m *.graph *.coo *.png *.csv *.mtx

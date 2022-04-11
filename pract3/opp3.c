#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>

#define K 500
#define N1 K*24 
#define N2 K*10
#define N3 K*24

double random(double mod){
    double res = (double)rand() / RAND_MAX * mod;
    res = rand() % 2? res : -res;
    return res;
}

void init(int n, int m, double* A){
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			A[i*m + j] = random(100);
}

void mul(const double* A, const double* B, double* C, int n1, int n2, int n3){
	for (int i = 0; i < n1; i++){
		for (int k = 0; k < n3; k++)
			C[i*n3 + k] = 0;
		for (int j = 0; j < n2; j++)
			for (int k = 0; k < n3; k++)
				C[i*n3 + k] += A[i*n2 + j] * B[j*n3 + k];
	}
}

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);
	int p1 = atoi(argv[1]), p2 = atoi(argv[2]);
	int K1 = N1 / p1, K2 = N3 / p2;
	double* A = (double*)malloc(K1 * N2 * sizeof(double));
	double* B = (double*)malloc(K2 * N2 * sizeof(double));
	double* C = (double*)malloc(K1 * K2 * sizeof(double));
	double* fullA, * fullB, * fullC;
	double starttime, endtime;
	MPI_Comm cart, gorCom, verCom;
	MPI_Datatype col, coltype, block, blocktype;
	int dims[2] = {p1, p2}, periods[2] = {0, 0}, coords[2];
	int rcount[p1 * p2], displs[p1 * p2];
	int reorder = 0;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart);
	MPI_Cart_get(cart, 2, dims, periods, coords);
	if (!coords[0] && !coords[1]){
		fullA = (double*)malloc(N1 * N2 * sizeof(double));
		fullB = (double*)malloc(N2 * N3 * sizeof(double));
		fullC = (double*)malloc(N1 * N3 * sizeof(double));
		init(N1, N2, fullA);
		init(N2, N3, fullB);
	}
	starttime = MPI_Wtime();
	MPI_Comm_split(cart, coords[0], coords[1], &gorCom);
	MPI_Comm_split(cart, coords[1], coords[0], &verCom);
	if (!coords[1])
		MPI_Scatter(fullA, K1 * N2, MPI_DOUBLE, A, K1 * N2, MPI_DOUBLE, 0, verCom);
	MPI_Type_vector(N2, K2, N3, MPI_DOUBLE, &coltype);
	MPI_Type_commit(&coltype);
	MPI_Type_create_resized(coltype, 0, K2 * sizeof(double), &col);
	MPI_Type_commit(&col);
	if (!coords[0])
		MPI_Scatter(fullB, 1, col, B, K2 * N2, MPI_DOUBLE, 0, gorCom);
	MPI_Bcast(A, K1 * N2, MPI_DOUBLE, 0, gorCom);
	MPI_Bcast(B, K2 * N2, MPI_DOUBLE, 0, verCom);
	mul(A, B, C, K1, N2, K2);
	MPI_Type_vector(K1, K2, N3, MPI_DOUBLE, &blocktype);
	MPI_Type_commit(&blocktype);
	MPI_Type_create_resized(blocktype, 0, K2 * sizeof(double), &block);
	MPI_Type_commit(&block);
	for (int i = 0; i < p1; i++)
		for (int j = 0; j < p2; j++){
			rcount[i*p2 + j] = 1;
			displs[i*p2 + j] = p2 * K1 * i + j;
	}
	MPI_Gatherv(C, K1 * K2, MPI_DOUBLE, fullC, rcount, displs, block, 0, cart);
	endtime = MPI_Wtime();
	if (!coords[0] && !coords[1]){
		printf("Multiply done!\nTime: %lf\n", endtime - starttime);
		free(fullA);
		free(fullB);
		free(fullC);
	}
	free(A);
	free(B);
	free(C);
	MPI_Type_free(&block);
	MPI_Type_free(&blocktype);
	MPI_Type_free(&col);
	MPI_Type_free(&coltype);
	MPI_Comm_free(&verCom);
	MPI_Comm_free(&gorCom);
	MPI_Comm_free(&cart);
	MPI_Finalize();
	return 0;
}

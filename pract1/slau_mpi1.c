#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>

#define K 10 //число членов для проверки сходимости
#define L 1000000 //ограничение сверху на норму
#define eps0 0.00000000000001 //epsilon
#define tay 0.001 //tay

double partNorm(const double* u, int begin, int end){
    double res = 0;
    for (int i = begin; i < end; i++)
        res += u[i] * u[i];
    return res;
}

void mul(const double* A, const double* u, double* Au, int begin, int end, int N){
    for (int i = 0; i < end - begin; i++){
        Au[i + begin] = 0;
        for (int k = 0; k < N; k++)
            Au[i + begin] += A[i*N + k] * u[k];
    }
}

void sub(const double* f, double koef, const double* s, double* res, int begin, int end){
    for (int i = begin; i < end; i++)
        res[i] = f[i] - koef * s[i];
}

int converg(int n, int l, const double* seqv, double eps){
    if (l < n)
        return 0;
    for (int i = 0; i < n; i++)
        if (seqv[i] > eps)
            return 0;
    return 1;
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
	int size, rank, beginPart, endPart, N, iter;
	double Nb, pNb, pNx, eps, starttime, endtime;
    double* b, * A, * fullA, * x, * z;
    double Nirm[K];

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int sendcmatr[size], dispmatr[size], recvcvec[size], dispvec[size];
    if (!rank){
        FILE* fin = fopen("data.txt", "r");
        fscanf(fin, "%d", &N);
        fullA = (double*)malloc(N * N * sizeof(double));
        b = (double*)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                fscanf(fin, "%lf", &fullA[i*N + j]);
        for (int i = 0; i < N; i++)
            fscanf(fin, "%lf", &b[i]);
        fclose(fin);
    }
    starttime = MPI_Wtime();
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int countSmallProc = size - N % size;
    
    for (int i = 0; i < size; i++){
        if (i < countSmallProc){
            sendcmatr[i] = (N / size) * N;
            dispmatr[i] = i * sendcmatr[i];
            recvcvec[i] = N / size;
            dispvec[i] = i * recvcvec[i];
        }
        else{
            sendcmatr[i] = (N / size + 1) * N;
            dispmatr[i] = i * sendcmatr[i] - countSmallProc * N;
            recvcvec[i] = N / size + 1;
            dispvec[i] = i * recvcvec[i] - countSmallProc;
        }
    }

    if (rank < countSmallProc){
        int k = N / size;
        beginPart = k * rank;
        endPart = beginPart + k;
    }
    else{
        int k = N / size + 1;
        beginPart = rank * k - countSmallProc;
        endPart = beginPart + k;
    }

    A = (double*)malloc((endPart - beginPart) * N * sizeof(double));
    x = (double*)malloc(N * sizeof(double));
    z = (double*)malloc(N * sizeof(double));
    if (rank) 
        b = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
        x[i] = 0;
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullA, sendcmatr, dispmatr, MPI_DOUBLE, A, (endPart - beginPart) * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    pNb = partNorm(b, beginPart, endPart);
    MPI_Allreduce(&pNb, &Nb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    eps = Nb * eps0 * eps0;
    Nirm[0] = 1;
    for (iter = 1; Nirm[(iter - 1) % K] < L && !converg(K, iter, Nirm, eps); iter++){
        mul(A, x, z, beginPart, endPart, N);
        sub(z, 1, b, z, beginPart, endPart);
        pNx = partNorm(z, beginPart, endPart);
        MPI_Allreduce(&pNx, &Nirm[iter % K], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        sub(x, tay, z, z, beginPart, endPart);
        MPI_Allgatherv(&z[beginPart], endPart - beginPart, MPI_DOUBLE, x, recvcvec, dispvec, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    endtime = MPI_Wtime();
    if (!rank){
        if (Nirm[(iter - 1) % K] < L)
            for (int i = 0; i < N; i++)
                printf("x[%d] = %lf\n", i, x[i]);
        else
            printf("No solution found\n");
        printf("TIME: %lf sec\n", endtime - starttime);
        free(fullA);
    }
    free(b);
    free(A);
    free(x);
    free(z);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize(); 
 	return 0;
}

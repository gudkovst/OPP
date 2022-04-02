#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <omp.h>

#define K 10 //число членов для проверки сходимости
#define L 1000000 //ограничение сверху на норму
#define eps0 0.00000000000001 //epsilon
#define tay 0.001 //tay

int converg(int n, int l, const double* seqv, double eps){
    if (l < n)
        return 0;
    for (int i = 0; i < n; i++)
        if (seqv[i] > eps)
            return 0;
    return 1;
}

int main(int argc, char** argv){
    double starttime, endtime;
    double eps, Nb = 0, norm = 0;
    int N, nth = atoi(argv[1]);
    FILE* fin = fopen("data.txt", "r");
    fscanf(fin, "%d", &N);
    double Nirm[K];
    double* Axb = (double*)malloc(N * sizeof(double));
    double* x = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* A = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            fscanf(fin, "%lf", &A[i*N + j]);
    for (int i = 0; i < N; i++)
        fscanf(fin, "%lf", &b[i]);
    starttime = omp_get_wtime();
    Nirm[0] = 1;
    omp_set_num_threads(nth);
    #pragma omp parallel 
    {
        #pragma omp for reduction (+:Nb)
            for (int i = 0; i < N; i++)
                Nb += b[i] * b[i];
        #pragma omp master
            eps = Nb * eps0 * eps0;
        #pragma omp for
            for (int i = 0; i < N; i++) {
                Axb[i] = -b[i];
                x[i] = 0;
            }
        for (int l = 1; Nirm[(l-1) % K] < L  && !converg(K, l, Nirm, eps); l++){
            #pragma omp for
                for (int i = 0; i < N; i++)
                    x[i] = x[i] - tay * Axb[i];
            #pragma omp for
                for (int i = 0; i < N; i++){
                    Axb[i] = 0;
                    for (int k = 0; k < N; k++)
                        Axb[i] += A[i*N + k] * x[k];
                }
            #pragma omp for
                for (int i = 0; i < N; i++)
                    Axb[i] = Axb[i] - b[i];
            #pragma omp for reduction (+:norm)
                for (int i = 0; i < N; i++)
                    norm += Axb[i] * Axb[i];
            #pragma omp single
            {
                Nirm[l % K] = norm;
                norm = 0;
            }
        }
    }
    endtime = omp_get_wtime();
    if (converg(K, K, Nirm, eps))
        for (int i = 0; i < N; i++)
            printf("%lf\n", x[i]);
    else
        printf("No solution found\n");
    
    printf("TIME: %lf sec\n", endtime - starttime);
    free(Axb);
    free(x);
    free(b);
    free(A);
    return 0;
}
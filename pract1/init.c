#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

double random(double mod, int sgn){
    double res = (double)rand() / RAND_MAX * mod;
    if (sgn)
        res = rand() % 2? res : -res;
    return res;
}

int main(int argc, char** argv){
    int N = atoi(argv[1]);
    double* A = (double*)malloc(N * N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++){
        b[i] = random(50, 1);
        for (int j = 0; j < i; j++){
            A[i*N + j] = random(1, 0);
            A[j*N + i] = A[i*N + j];
        }
        A[i*N + i] = random(1, 0) + 25;
    }
    freopen("data.txt", "w", stdout);
    printf("%d\n", N);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++)
            printf("%lf ", A[i*N + j]);
        printf("\n"); 
    }
    for (int i = 0; i < N; i++)
        printf("%lf ", b[i]);
    return 0;
}


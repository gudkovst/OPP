#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/times.h>
#include <unistd.h>

#define K 10 //число членов для проверки сходимости
#define L 1000000 //ограничение сверху на норму
#define eps0 0.00000000000001 //epsilon
#define tay 0.001 //tay

double norm(const double* u, int N){
    double res = 0;
    for (int i = 0; i < N; i++)
        res += u[i] * u[i];
    return res;
}

void mul(const double* A, const double* u, double* Au, int N){
    for (int i = 0; i < N; i++){
        Au[i] = 0;
        for (int k = 0; k < N; k++)
            Au[i] += A[i*N + k] * u[k];
    }
}

void sub(const double* f, double koef, const double* s, double* res, int N){
    for (int i = 0; i < N; i++)
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

double* iterate(const double* A, const double* b, double eps, int N){
    int s = 0, l;
    double Nirm[K];
    double* Axb = (double*)malloc(N * sizeof(double));
    double* sol;
    double* x [2];
    x[0] = (double*)malloc(N * sizeof(double));
    x[1] = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        Axb[i] = -b[i];
        x[0][i] = 0;
    }
    Nirm[0] = 1;
    for (l = 1; Nirm[(l-1) % K] < L  && !converg(K, l, Nirm, eps); s = !s, l++){
        sub(x[s], tay, Axb, x[!s], N);
        mul(A, x[!s], Axb, N);
        sub(Axb, 1, b, Axb, N);
        Nirm[l % K] = norm(Axb, N);
    }
    if (Nirm[(l-1) % K] < L)
        sol = x[s];
    else
        sol = NULL;
    free(Axb);
    free(x[!s]);
    return sol;
}

int main(){
    struct tms starttime, endtime;
    long clk_per_sec = sysconf(_SC_CLK_TCK);
    long clocks;
    double eps;
    int N;
    FILE* fin = fopen("data.txt", "r");
    fscanf(fin, "%d", &N);
    double* sol;
    double* b = (double*)malloc(N * sizeof(double));
    double* A = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            fscanf(fin, "%lf", &A[i*N + j]);
    for (int i = 0; i < N; i++)
        fscanf(fin, "%lf", &b[i]);
    times(&starttime);
    eps = norm(b, N) * eps0 * eps0;
    sol = iterate(A, b, eps, N);
    times(&endtime);
    if (sol) {
        for (int i = 0; i < N; i++)
            printf("%lf\n", sol[i]);
        free(sol);
    }
    else
        printf("No solution found\n");
    clocks = endtime.tms_utime - starttime.tms_utime;
    printf("TIME: %lf sec\n", (double)clocks / clk_per_sec);
    free(b);
    free(A);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/times.h>
#include <unistd.h>

#define K 100000 //макс число итераций

int* codingMap(const short* map, int size){
    int codelen = sizeof(int) * 8;
    int k = size / codelen;
    int* res = (int*)malloc((k + 1) * sizeof(int));
    for (int i = 0; i < k; i++){
        res[i] = 0;
        for (int j = 0; j < codelen; j++)
            res[i] = (res[i] << 1) + map[codelen * i + j];
    }
    res[k] = 0;
    for (int j = k * codelen; j < size; j++)
        res[k] = (res[k] << 1) + map[j];
    return res;
}

int proverka(const int* old, const int* pres, int size){
    int codelen = sizeof(int) * 8;
    int k = size / codelen + 1;
    for (int i = 0; i < k; i++)
        if (pres[i] - old[i])
            return 0;
    return 1;
}

short sost(const short* map, int i, int j, int n, int m){
    short res = map[i*m + (m + j - 1) % m];
    res += map[i*m + (j + 1) % m];
    res += map[((n + i - 1) % n) * m + j];
    res += map[((i + 1) % n) * m + j];
    res += map[((n + i - 1) % n) * m + (m + j - 1) % m];
    res += map[((n + i - 1) % n) * m + (j + 1) % m];
    res += map[((i + 1) % n) * m + (m + j - 1) % m];
    res += map[((i + 1) % n) * m + (j + 1) % m];
    return (res == 3 || map[i*m + j] + res == 3);
}

int main(int argc, char** argv){
	int count, N = atoi(argv[1]), M = atoi(argv[2]);
    struct tms starttime, endtime;
    long clk_per_sec = sysconf(_SC_CLK_TCK);
    long clocks;
	short* map[2];
	int* hist[K];
    FILE* fin = fopen("data.txt", "r");
    map[0] = (short*)malloc(N * M * sizeof(short));
    map[1] = (short*)malloc(N * M * sizeof(short));
    for (int i = 0; i < N * M; i++)
         fscanf(fin, "%hu", &map[0][i]);
    fclose(fin);
    times(&starttime);
    
    for (int iter = 0; iter < K; iter++){
        hist[iter] = codingMap(map[iter % 2], N * M);
        for (int i = 0; i < iter; i++)
            if (proverka(hist[i], hist[iter], N * M)){
                count = iter;
                iter = K;
                break;
            }
        if (iter < K)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < M; j++)
                    map[!(iter % 2)][i*M + j] = sost(map[iter % 2], i, j, N, M);
    }
    times(&endtime);
    clocks = endtime.tms_utime - starttime.tms_utime;
    printf("Work finished after %d iteration\nTIME: %lf\n", count, (double)clocks / clk_per_sec);
    free(map[0]);
    free(map[1]);
    for (int i = 0; i < count; i++)
        free(hist[i]); 
    return 0;
}
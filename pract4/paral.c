#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>

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

short proverka(const int* old, const int* pres, int size){
    int codelen = sizeof(int) * 8;
    int k = size / codelen + 1;
    for (int i = 0; i < k; i++)
        if (pres[i] - old[i])
            return 0;
    return 1;
}

int cycle(int size, const short* sumflags, int iter){
    for (int i = 0; i < iter; i++)
        if (sumflags[i] == size)
            return 1;
    return 0;
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
	MPI_Init(&argc, &argv);
	int size, rank, kstr, countSmallProc, iter = 0, N = atoi(argv[1]), M = atoi(argv[2]);
    double starttime, endtime;
	short* map[2], * birth;
    short* sumflags = (short*)malloc(K * sizeof(short));
	int* hist[K];
    short flags[K] = {0};
	MPI_Request reqtop, reqdown, sendtop, senddown;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rcount[size], displs[size];
    if (!rank){
    	FILE* fin = fopen("data.txt", "r");
    	birth = (short*)malloc(N * M * sizeof(short));
    	for (int i = 0; i < N * M; i++)
    		fscanf(fin, "%hu", &birth[i]);
    	fclose(fin);
    }
    countSmallProc = size - N % size;
    for (int i = 0; i < size; i++){
    	if (i < countSmallProc){
            rcount[i] = N / size * M;
            displs[i] = i * rcount[i];
        }
        else{
            rcount[i] = (N / size + 1) * M;
            displs[i] = i * rcount[i] - countSmallProc * M;
        }
    }
    kstr = (rank < countSmallProc)? N / size + 2 : N / size + 3;
    map[0] = (short*)malloc(kstr * M * sizeof(short));
    map[1] = (short*)malloc(kstr * M * sizeof(short));
    starttime = MPI_Wtime();
    MPI_Scatterv(birth, rcount, displs, MPI_SHORT, map[0] + M, (kstr - 2) * M, MPI_SHORT, 0, MPI_COMM_WORLD);
    for (; iter < K; iter++){
    	MPI_Isend(map[iter % 2] + M, M, MPI_SHORT, (size + rank - 1) % size, 1, MPI_COMM_WORLD, &sendtop);
    	MPI_Isend(map[iter % 2] + (kstr - 2) * M, M, MPI_SHORT, (rank + 1) % size, 2, MPI_COMM_WORLD, &senddown);
    	MPI_Irecv(map[iter % 2] + (kstr - 1) * M, M, MPI_SHORT, (rank + 1) % size, 1, MPI_COMM_WORLD, &reqdown);
    	MPI_Irecv(map[iter % 2], M, MPI_SHORT, (size + rank - 1) % size, 2, MPI_COMM_WORLD, &reqtop);
        hist[iter] = codingMap(map[iter % 2] + M, (kstr - 2) * M);
        for (int i = 0; i < iter; i++)
            flags[i] = proverka(hist[i], hist[iter], (kstr - 2) * M);
        MPI_Allreduce(flags, sumflags, iter, MPI_SHORT, MPI_SUM, MPI_COMM_WORLD);
        if (cycle(size, sumflags, iter)){
            MPI_Wait(&reqtop, MPI_STATUS_IGNORE);
            MPI_Wait(&reqdown, MPI_STATUS_IGNORE);
            MPI_Wait(&senddown, MPI_STATUS_IGNORE);
            MPI_Wait(&sendtop, MPI_STATUS_IGNORE);
            break;
        }
        for (int i = 2; i < kstr - 2; i++)
            for (int j = 0; j < M; j++)
                map[!(iter % 2)][i*M + j] = sost(map[iter % 2], i, j, kstr, M);
        MPI_Wait(&reqtop, MPI_STATUS_IGNORE);
        for (int j = 0; j < M; j++)
            map[!(iter % 2)][M + j] = sost(map[iter % 2], 1, j, kstr, M);
        MPI_Wait(&reqdown, MPI_STATUS_IGNORE);
        for (int j = 0; j < M; j++)
            map[!(iter % 2)][(kstr - 2) * M + j] = sost(map[iter % 2], kstr - 2, j, kstr, M);
        MPI_Wait(&sendtop, MPI_STATUS_IGNORE);
        MPI_Wait(&senddown, MPI_STATUS_IGNORE);
    }
    endtime = MPI_Wtime();
    if (!rank){
        printf("Work finished after %d iteration\nTIME: %lf\n", iter, endtime - starttime);
        free(birth);
    }
    free(sumflags);
    free(map[0]);
    free(map[1]);
    for (int i = 0; i < iter; i++)
        free(hist[i]);
    MPI_Finalize(); 
    return 0;
}
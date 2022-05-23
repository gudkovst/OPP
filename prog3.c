#include <pthread.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#define L 1000 //параметр веса задачи
#define P 0.25  //параметр алгоритма балансировки (0,1)
#define CountTasks 100 //начальное число задач для каждого процесса
#define NumTask 10 //число итераций

//Коды сообщений:
#define REPORT 1 //отчёт о сделанной работе
#define REQUEST 2 //запрос на новую работу
#define FINISH 3 //сигнал о завершении работы
#define TASKS 4 //задачи

//Коды получателей(теги)
#define PROCESS 1
#define BALANCER 2

typedef struct List{
	int data;
	struct List* next;
}list;

typedef struct Queue{
	int len;
	list* fst;
	list* last;
}queue;

//Глобальные переменные
int ids[4] = {0, 1, 2, 3};
int rank, size, count, flag, req;
int* countsT;
double* times, *results;
double start, end, globres;
queue* tasks;
pthread_t thrs[4];
pthread_mutex_t taskMutex, condMutex;
pthread_cond_t cond;
pthread_attr_t attrs;

queue* cQueue(){
	queue* q = (queue*)malloc(sizeof(queue));
	q->len = 0;
	q->fst = NULL;
	q->last = NULL;
	return q;
}

void push(queue* q, int d){
	list* node = (list*)malloc(sizeof(list));
	node->data = d;
	node->next = NULL;
	if (q->len)
		q->last->next = node;
	else
		q->fst = node;
	q->last = node;
	q->len++;
}

int pop(queue* q){
	if (!q->len)
		return -1;
	int res = q->fst->data;
	list* head = q->fst->next;
	free(q->fst);
	q->fst = head;
	q->len--;
	return res;
}

void deQueue(queue* q){
	while(pop(q) != -1);
}

int calcWt(int i, int iter, int rank, int size){
	int k1 = abs(CountTasks / 2 - i % CountTasks);
	int k2 = abs(rank - iter % size);
	return k1 * k2 * L;
}

double work(int wt){
	double res = 0;
	for(int i = 0; i < wt; i++)
		res += sin(i);
	return res;
}

void* worker(void* me){
	start = MPI_Wtime();
	while (flag || tasks->len){
		pthread_mutex_lock(&taskMutex);
		int wtask = pop(tasks);
		pthread_mutex_unlock(&taskMutex);
		globres += work(wtask);
		count += !!(wtask + 1);
		pthread_mutex_lock(&condMutex);
		if (!(count % 5) || !tasks->len)
			pthread_cond_signal(&cond);
		pthread_mutex_unlock(&condMutex);
	}
	end = MPI_Wtime();
	pthread_mutex_lock(&condMutex);
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&condMutex);
}

void* recvmanager(void* me){
	int buflen = CountTasks / 4 + 2;
	int* buf = (int*)malloc(buflen * sizeof(int));
	while(flag){
		MPI_Recv(buf, buflen, MPI_INT, MPI_ANY_SOURCE, PROCESS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (buf[0] == REQUEST){
			int k = P * tasks->len;
			int* bufTasks = (int*)malloc((k + 2) * sizeof(int));
			bufTasks[0] = TASKS;
			bufTasks[1] = k;
			pthread_mutex_lock(&taskMutex);
			for (int i = 0; i < k; i++)
				bufTasks[i + 2] = pop(tasks);
			pthread_mutex_unlock(&taskMutex);
			MPI_Send(bufTasks, k + 2, MPI_INT, buf[1], PROCESS, MPI_COMM_WORLD);
		}
		if (buf[0] == TASKS){
			pthread_mutex_lock(&taskMutex);
			for (int i = 0; i < buf[1]; i++)
				push(tasks, buf[i + 2]);
			pthread_mutex_unlock(&taskMutex);
			req = 0;
		}
		if (buf[0] == FINISH)
			flag = 0;
	}
	
}

void* sendmanager(void* me){
	int msg;
	while(flag){
		pthread_mutex_lock(&condMutex);
		pthread_cond_wait(&cond, &condMutex);
		pthread_mutex_unlock(&condMutex);
		msg = REPORT;
		MPI_Send(&msg, 1, MPI_INT, 0, BALANCER, MPI_COMM_WORLD);
		if (flag && tasks->len <= 0.1 * CountTasks && !req){
			msg = REQUEST;
			MPI_Send(&msg, 1, MPI_INT, 0, BALANCER, MPI_COMM_WORLD);
			req = 1;
		}
	}
}

void* balancer(void* me){
	int sost[size] = {0}, load[size] = {0};
	int buf, msg[2], flagcount = 0;
	MPI_Status stat;
	while(flagcount < size){
		MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, BALANCER, MPI_COMM_WORLD, &stat);
		if (buf == REPORT)
			load[stat.MPI_SOURCE] += 5;
		if (buf == REQUEST && !sost[stat.MPI_SOURCE]){
			int indMaxLoad = 0, numMaxLoad = load[0];
			for (int i = 1; i < size; i++)
				if (i != stat.MPI_SOURCE && load[i] < numMaxLoad && !sost[i]){
					numMaxLoad = load[i];
					indMaxLoad = i;
				}
			if (indMaxLoad == stat.MPI_SOURCE || numMaxLoad > (1 - P)*CountTasks || sost[indMaxLoad]){
				msg[0] = FINISH;
				flagcount += !sost[stat.MPI_SOURCE];
				sost[stat.MPI_SOURCE] = 1;
				MPI_Send(msg, 1, MPI_INT, stat.MPI_SOURCE, PROCESS, MPI_COMM_WORLD);
			}
			else{
				msg[0] = REQUEST;
				msg[1] = stat.MPI_SOURCE;
				MPI_Send(msg, 2, MPI_INT, indMaxLoad, PROCESS, MPI_COMM_WORLD);
			}
		}
	}
}

int main(int argc, char** argv){
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	pthread_mutex_init(&taskMutex, NULL);
	pthread_mutex_init(&condMutex, NULL);
	pthread_cond_init(&cond, NULL);
	pthread_attr_init(&attrs);
	pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);
	countsT = (int*)malloc(size * sizeof(int));
	results = (double*)malloc(size * sizeof(double));
	times = (double*)malloc(size * sizeof(double));
	tasks = cQueue();
	for(int iter = 0; iter < NumTask; iter++){
		for (int i = 0; i < CountTasks; i++)
			push(tasks, calcWt(i, iter, rank, size));
		globres = 0;
		count = 0;
		flag = 1;
		req = 0;
		pthread_create(&thrs[0], &attrs, worker, &ids[0]);
		pthread_create(&thrs[1], &attrs, sendmanager, &ids[1]);
		pthread_create(&thrs[2], &attrs, recvmanager, &ids[2]);
		if (!rank)
			pthread_create(&thrs[3], &attrs, balancer, &ids[3]);
		for (int i = 0; i < 3; i++)
			pthread_join(thrs[i], NULL);
		if (!rank)
			pthread_join(thrs[3], NULL);
		double time = end - start;
		MPI_Gather(&count, 1, MPI_INT, countsT, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Gather(&globres, 1, MPI_DOUBLE, results, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(&time, 1, MPI_DOUBLE, times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (!rank){
			double delta = 0, maxTime = 0;
			for (int i = 0; i < size; i++)
				printf("Process %d:\ncount: %d\nresult: %lf\ntime: %lf\n", i, countsT[i], results[i], times[i]);
			for (int i = 0; i < size; i++){
				if (times[i] > maxTime)
					maxTime = times[i];
				for (int j = i + 1; j < size; j++)
					if (fabs(times[i] - times[j]) > delta)
						delta = fabs(times[i] - times[j]);
			}
			printf("Time disbalance: %lfs\nProportion disbalance: %lf\n", delta, delta / maxTime * 100);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	pthread_mutex_destroy(&taskMutex);
	pthread_mutex_destroy(&condMutex);
	pthread_cond_destroy(&cond);
	pthread_attr_destroy(&attrs);
	deQueue(tasks);
	MPI_Finalize();
	return 0;
}
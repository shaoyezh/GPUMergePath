#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define N_THREADS 1024
#define N_BLOCKS 64
#define SIZEA 65536
#define SIZEB 65536

__global__ void merge(int *A, int *B, int *m, int *A_idx, int *B_idx){
	int startA, endA;
	int startB, endB;
	
	__shared__ int A_shared[1024]; // SIZEA / block size 
	__shared__ int B_shared[1024]; // SIZEB / block size

	if (blockIdx.x == 0){
		startA = 0;
		endA = A_idx[blockIdx.x];
		startB = 0;
		endB = B_idx[blockIdx.x];
	}
	else if (blockIdx.x == N_BLOCKS-1){
		startA = A_idx[blockIdx.x-1];
		endA = SIZEA;
		startB = B_idx[blockIdx.x-1];
		endB = SIZEB;
	}
	else{
		startA = A_idx[blockIdx.x-1];
		endA = A_idx[blockIdx.x];
		startB = B_idx[blockIdx.x-1];
		endB = B_idx[blockIdx.x];
	}

	int sizeAshared = endA - startA; 
	int sizeBshared = endB - startB; 

	int index = blockIdx.x * blockdim.x + threadIdx.x; 

	A_shared[startA + threadIdx]


 

}

__global__ void pathPartition(int *A, int *B, int *A_idx, int *B_idx){
	int index = (SIZEA + SIZEA) / N_BLOCKS * (blockIdx.x + 1); 
	if (blockIdx.x == N_BLOCKS - 1){
		return;
	}
	int atop, btop, abot, bbot, offset;
	atop = index; 
	btop = 0; 
	// Binary Search 
	if(index > SIZEA){
		atop = SIZEA; 
		btop = index - SIZEA;
	}
	abot = btop; 
	while(1){
		offset = abs(atop - abot) / 2;
		int i = atop - offset; 
		int j = btop + offset; 
		if(j > 0 && A[i] > B[j-1]){
			if(i > 0 && (A[i-1] < B[j]|| A[i-1] == B[j])){
				A_idx[blockIdx.x] = i; 
				B_idx[blockIdx.x] = j; 
				break;
			}else{
				atop = i - 1;
				btop = j + 1;
			}
		}else{
			abot = i + 1;
		}
	}
}

int main(){
	int *A = (int*) malloc(sizeof(int) * SIZEA);
	for(int i = 0; i < SIZEA; i++){
		A[i] = 2 * i;
	}

	int *B = (int*) malloc(sizeof(int) * SIZEB);
	for(int i = 0; i < SIZEB; i++){
		B[i] = 2 * i + 1;
	}
	int *mHost = (int*) malloc(sizeof(int) * (SIZEA + SIZEB));

	// int A_idx[N_BLOCKS]; 
	// int B_idx[N_BLOCKS]; 
	
	int *A_dev, *B_dev, *m_dev, *A_idx_dev, *B_idx_dev; 

	cudaMalloc((void**) &A_dev, SIZEA * sizeof(int));
	cudaMalloc((void**) &B_dev, SIZEB * sizeof(int));
	cudaMalloc((void**) &m_dev, (SIZEA + SIZEB) * sizeof(int));
	cudaMalloc((void**) &A_idx_dev, N_BLOCKS * sizeof(int));
	cudaMalloc((void**) &B_idx_dev, N_BLOCKS * sizeof(int));

	cudaMemcpy(A_dev, A, SIZEA * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev, B, SIZEB * sizeof(int), cudaMemcpyHostToDevice);

	pathPartition<<<N_BLOCKS, 1>>>(A_dev, B_dev, A_idx_dev, B_idx_dev); 


}
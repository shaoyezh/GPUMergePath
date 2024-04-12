#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define X 0
#define Y 1
#define SIZEA 65536
#define SIZEB 65336

#define N_BLOCKS 64
#define N_THREADS 2

__global__ void mergeBig_k(int *A, int *B, int *M, int *A_idx, int *B_idx){

	// Shared memory on which we will work
	__shared__ int A_shared[1024]; // SIZEA / block size 
	__shared__ int B_shared[1024]; // SIZEB / block size

	__shared__ int biaisA;
	__shared__ int biaisB;

	// (endA-startA) : size of A in partition
	// (endB-startB) : size of B in partition
	int startA, endA;
	int startB, endB;
	
	// We retrieve the indexes of the start and end of A and B in relation to the global table
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

	// Article ratings
	// There are N elements to merge
	// N = SIZEA + SIZEB
	// Each partition contains N/p elements, each block processes a partition
	// N / p = (endB-startB) + (endA-startA) = (SIZEA+SIZEB) / N_BLOCKS
	// If Z is the number of threads
	// We are going to merge Z elements at a time
	// So we need to do it (N / p) / Z times
	// We are going to move the sliding window (N / p) / Z times
	int iter_max = (blockDim.x - 1 + (endB-startB) + (endA-startA)) / blockDim.x;
	int iter = 0;

	biaisA = 0;
	biaisB = 0;
	do{
		// To synchronize the biases
		__syncthreads();

		// Loading values ​​into shared memory
		if (startA + biaisA + threadIdx.x < endA){
			A_shared[threadIdx.x] = A[startA + biaisA + threadIdx.x];
		}

		if (startB + biaisB + threadIdx.x < endB){
			B_shared[threadIdx.x] = B[startB + biaisB + threadIdx.x];	
		}

		// To synchronize shared memory
		__syncthreads();

		// Get the size of the sliding window
		// In general it is the number of threads (blockDim.x), i.e. We are in a Z * Z square normally
		// But the size can be smaller if there are fewer blockDim.x elements to load
		int sizeAshared = endA-startA - biaisA;
		int sizeBshared = endB-startB - biaisB;
		if (sizeAshared < 0)
			sizeAshared = 0;
		if (sizeAshared > blockDim.x && sizeAshared != 0)
			sizeAshared = blockDim.x;
		if (sizeBshared < 0)
			sizeBshared = 0;
		if (sizeBshared > blockDim.x && sizeBshared != 0)
			sizeBshared = blockDim.x;

		// Binary search
		int i = threadIdx.x;

		if (i < sizeAshared + sizeBshared){
			int K[2];
			int P[2];

			if (i > sizeAshared) {
				K[X] = i - sizeAshared;
				K[Y] = sizeAshared;
				P[X] = sizeAshared;
				P[Y] = i - sizeAshared;
			}
			else {
				K[X] = 0;
				K[Y] = i;
				P[X] = i;
				P[Y] = 0;
			}

			while (1) {
				int offset = (abs(K[Y] - P[Y]))/2;
				int Q[2] = {K[X] + offset, K[Y] - offset};

				if (Q[Y] >= 0 && Q[X] <= sizeBshared && (Q[Y] == sizeAshared || Q[X] == 0 || A_shared[Q[Y]] > B_shared[Q[X]-1])) {
					if (Q[X] == sizeBshared || Q[Y] == 0 || A_shared[Q[Y]-1] <= B_shared[Q[X]]) {
						int idx = startA + startB + i + iter * blockDim.x;
						if (Q[Y] < sizeAshared && (Q[X] == sizeBshared || A_shared[Q[Y]] <= B_shared[Q[X]]) ) {
							M[idx] = A_shared[Q[Y]];
							atomicAdd(&biaisA, 1);	//  Bias to increment 
						}
						else {
							M[idx] = B_shared[Q[X]];
							atomicAdd(&biaisB, 1); //  Bias to increment
						}
						//printf("blockIdx.x = %d threadIdx.x = %d idx = %d m = %d biaisA = %d\n", blockIdx.x, threadIdx.x, idx, M[idx], biaisA);
						break ;
					}
					else {
						K[X] = Q[X] + 1;
						K[Y] = Q[Y] - 1;
					}
				}
				else {
					P[X] = Q[X] - 1;
					P[Y] = Q[Y] + 1 ;
				}
			}
		}
		iter = iter + 1;
	} while(iter < iter_max);
}

__global__ void pathBig_k(int *A, int *B, int *M, int *A_idx, int *B_idx){

	// In this kernel, we will simply look for diagonal N_BLOCKS
	// such that each block will process N/N_BLOCKS elements in the second kernel
	int i = (SIZEA + SIZEB)/N_BLOCKS * (blockIdx.x + 1);
	if (blockIdx.x == N_BLOCKS-1){
		return;
	}

	// Binary search
	int K[2];
	int P[2];

	if (i > SIZEA) {
		K[X] = i - SIZEA; //btop / abottom
		K[Y] = SIZEA; // atop 
		P[X] = SIZEA; // atop /bbottom
		P[Y] = i - SIZEA; //btop / abottom
	}
	else {
		K[X] = 0;
		K[Y] = i;
		P[X] = i;
		P[Y] = 0;
	}
	// abottom = btop
	while (1) {

		int offset = (abs(K[Y] - P[Y]))/2;  // (atop - abottom) / 2
		int Q[2] = {K[X] + offset, K[Y] - offset};

		if (Q[Y] >= 0 && Q[X] <= SIZEB && (Q[Y] == SIZEA || Q[X] == 0 || A[Q[Y]] > B[Q[X]-1])) {
			if (Q[X] == SIZEB || Q[Y] == 0 || A[Q[Y]-1] <= B[Q[X]]) {
				if (Q[Y] < SIZEA && (Q[X] == SIZEB || A[Q[Y]] <= B[Q[X]]) ) {
					M[i] = A[Q[Y]];
				}
				else {
					M[i] = B[Q[X]];
				}
				A_idx[blockIdx.x] = Q[Y];
				B_idx[blockIdx.x] = Q[X];
				// printf("blockIdx.x = %d | Aidx[%d] = %d | Bidx[%d] = %d \n", blockIdx.x, blockIdx.x, Q[Y], blockIdx.x, Q[X]);
				break ;
			}
			else {
				K[X] = Q[X] + 1;
				K[Y] = Q[Y] - 1;
			}
		}
		else {
			P[X] = Q[X] - 1;
			P[Y] = Q[Y] + 1;
		}
	}
}

int main(){

	// Allocation de la mémoire, remplissage du tableau
	int *A = (int*) malloc(sizeof(int) * SIZEA);
	for (int i = 0; i < SIZEA; i++){
		A[i] = 2 * i;
	}
	int *B = (int*) malloc(sizeof(int) * SIZEB);
	for (int i = 0; i < SIZEB; i++){
		B[i] = 2 * i + 1;
	}
	int mHost[SIZEA + SIZEB];		// Tableau merged	

	int A_idx[N_BLOCKS];			// Merge path
	int B_idx[N_BLOCKS];			// Merge path
	int *aDevice, *bDevice, *mDevice, *A_idxDevice, *B_idxDevice;

	// GPU global memory allocation
	cudaMalloc( (void**) &aDevice, SIZEA * sizeof(int) );
	cudaMalloc( (void**) &bDevice, SIZEB * sizeof(int) );
	cudaMalloc( (void**) & , (SIZEA+SIZEB) * sizeof(int) );
	cudaMalloc( (void**) &A_idxDevice, N_BLOCKS * sizeof(int) );
	cudaMalloc( (void**) &B_idxDevice, N_BLOCKS * sizeof(int) );

	// Copy arrays to GPU
	cudaMemcpy( aDevice, A, SIZEA * sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( bDevice, B, SIZEB * sizeof(int), cudaMemcpyHostToDevice );

	// Run the kernel to find an array partition
	// (SIZEA+SIZEB) / N_BLOCKS elements to process for each block in the kernel
	pathBig_k<<<N_BLOCKS, 1>>>(aDevice, bDevice, mDevice, A_idxDevice, B_idxDevice);

//	cudaMemcpy( mHost, mDevice, (SIZEA+SIZEB) * sizeof(int), cudaMemcpyDeviceToHost );
//	cudaMemcpy( A_idx, A_idxDevice, N_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost );
//	cudaMemcpy( B_idx, B_idxDevice, N_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost );

//	A_idx[N_BLOCKS-1] = SIZEA;
//	B_idx[N_BLOCKS-1] = SIZEB;

//	cudaMemcpy( A_idxDevice, A_idx, N_BLOCKS * sizeof(int), cudaMemcpyHostToDevice );
//	cudaMemcpy( B_idxDevice, B_idx, N_BLOCKS * sizeof(int), cudaMemcpyHostToDevice );

	// Sliding window to load elements into shared memory
	mergeBig_k<<<N_BLOCKS, N_THREADS>>>(aDevice, bDevice, mDevice, A_idxDevice, B_idxDevice);

	cudaMemcpy( mHost, mDevice, (SIZEA+SIZEB) * sizeof(int), cudaMemcpyDeviceToHost );
	for (int i = 0; i < SIZEA+SIZEB; i ++){
		printf("m[%d] = %d\n", i, mHost[i]);
	}


	free(A);
	free(B);
	cudaFree(aDevice);
	cudaFree(bDevice);
	cudaFree(mDevice);
	cudaFree(A_idxDevice);
	cudaFree(B_idxDevice);

	return 0;
}


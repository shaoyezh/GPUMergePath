#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define N_THREADS 32
#define N_BLOCKS 16
#define SIZEA 1024
#define SIZEB 1024

__global__ void pathMerge(int *A, int *B, int *m, int *A_idx, int *B_idx){
	int startA, endA;
	int startB, endB;
	
	__shared__ int A_shared[1024]; // SIZEA / block size 
	__shared__ int B_shared[1024]; // SIZEB / block size
	__shared__ int stepA;
	__shared__ int stepB; 
	stepA = 0;
	stepB = 0; 
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

	// sliding window:
	// iteration times = (N/p) / Z 
  int iterations= ((endB-startB) + (endA-startA)+blockDim.x-1) / blockDim.x;
	int iter = 0;
	while(iter < iterations){
		__syncthreads(); // sync stepA and stepB 

		// loading data to shared memoery 

    if(startA + stepA + threadIdx.x < endA){
        A_shared[threadIdx.x] = A[startA + stepA + threadIdx.x];
    }
    if(startB + stepB + threadIdx.x < endB){
        B_shared[threadIdx.x] = B[startB + stepB + threadIdx.x];
    }
		
		

		__syncthreads(); // sync shared memory
		int sizeAshared = endA - startA - stepA; 
		int sizeBshared = endB - startB - stepB; 
		if (sizeAshared > blockDim.x && sizeAshared != 0)
			sizeAshared = blockDim.x;
		if (sizeBshared > blockDim.x && sizeBshared != 0)
			sizeBshared = blockDim.x;
		
		int index = threadIdx.x; 

		int atop, btop, abot, bbot, offset;
		atop = index; 
		btop = 0; 
		// Binary Search 
		if(index > sizeAshared){
			atop = sizeAshared; 
			btop = index - sizeAshared;
		}
		abot = btop; 
		while(1){
			offset = abs(atop - abot) / 2;
			int i = atop - offset; 
			int j = btop + offset; 
			if(i == sizeAshared || j == 0 || A_shared[i] > B_shared[j-1]){
        // i: 16, j: 15, A[i]: 1, B[j]: 2047
				if(i == 0 || j == sizeBshared || A_shared[i-1] <= B_shared[j]){
					// solution found 
					int idx = startA + startB + index + iter * blockDim.x;

					if(i < sizeAshared && (j == sizeBshared || A_shared[i] <= B_shared[j])){
            // if i > sizeAshared, then A_shared[i] doesn't exist in current block, so m[idx] = B_shared[j];
            // if j == sizeBshared, then Bshared[j] doesn't exist, so m[idx] = Ashared[i]
						m[idx] = A_shared[i];
						atomicAdd(&stepA, 1); 
					}else{
						m[idx] = B_shared[j]; 
						atomicAdd(&stepB, 1);
					}
					break;
				}else{
					// solution in bot left section;
					atop = i - 1;
					btop = j + 1;
				}
			}else{
				// solution in top right section 
				abot = i + 1 ; 
				bbot = j - 1; 
			}
		}
		iter += 1;
	}

}
 
__global__ void pathPartition(int *A, int *B, int *A_idx, int *B_idx){

	if (blockIdx.x == N_BLOCKS - 1){
		return;
	}
	int index = (SIZEA + SIZEA) / N_BLOCKS * (blockIdx.x + 1); 
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
		if(j == 0 || A[i] > B[j-1]){
			if(i == 0 || A[i-1] <= B[j]){
				// solution found 
				A_idx[blockIdx.x] = i; 
				B_idx[blockIdx.x] = j; 
				break;
			}else{
				// solution in bot left section;
				atop = i - 1;
				btop = j + 1;
			}
		}else{
			// solution in top right section 
			abot = i + 1 ; 
			bbot = j - 1; 
		}
	}
}

int main(){
	int *A = (int*) malloc(sizeof(int) * SIZEA);
	for(int i = 0; i < SIZEA; i++){
		A[i] = 2 * i +1 ;
	}

	int *B = (int*) malloc(sizeof(int) * SIZEB);
	for(int i = 0; i < SIZEB; i++){
		B[i] = 2 * i;
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

	pathMerge<<<N_BLOCKS, N_THREADS>>>(A_dev, B_dev, m_dev, A_idx_dev, B_idx_dev); 

	cudaMemcpy(mHost, m_dev, (SIZEA + SIZEB) * sizeof(int), cudaMemcpyDeviceToHost); 

	for(int i = SIZEA + SIZEB - 100; i < SIZEA + SIZEB; i++){
		printf("m[%d]: %d\n", i, mHost[i]);
	}

	free(A);
	free(B);
	free(mHost);
	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(m_dev);
	cudaFree(A_idx_dev); 
	cudaFree(B_idx_dev);

	return 0;

}
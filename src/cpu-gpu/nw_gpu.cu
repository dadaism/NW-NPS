/*
* File: nw_gpu.cu
* Author: Da Li
* Email:  da.li@mail.missouri.edu
* Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
*
* Description: This file defines all the wrapper functions GPU implementations.
*
*/

#include <cuda_runtime_api.h>
#include "nw_gpu.h"

#include "nw_kernel_diagonal.cu"
#include "nw_kernel_tile.cu"


inline void cudaCheckError(int line, cudaError_t ce)
{ 
    if (ce != cudaSuccess){
        printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
        exit(1);
    }
}

void nw_gpu_allocate(int stream_num)
{
	/* GPU memory allocation */
	int i = stream_num;
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set1[i], sizeof(char)*pos1[i][pair_num_gpu[i]] ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set2[i], sizeof(char)*pos2[i][pair_num_gpu[i]] ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_score_matrix[i], sizeof(int)*pos_matrix[i][pair_num_gpu[i]]) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos1[i], sizeof(unsigned int)*(pair_num_gpu[i]+1) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos2[i], sizeof(unsigned int)*(pair_num_gpu[i]+1) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos_matrix[i], sizeof(unsigned int)*(pair_num_gpu[i]+1) ) );	
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_dim_matrix[i], sizeof(unsigned int)*(pair_num_gpu[i]+1) ) );

	/* Memcpy to device */
	cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set1[i], sequence_set1[i], sizeof(char)*pos1[i][pair_num_gpu[i]], cudaMemcpyHostToDevice ) );
   	cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set2[i], sequence_set2[i], sizeof(char)*pos2[i][pair_num_gpu[i]], cudaMemcpyHostToDevice ) );
   	cudaCheckError( __LINE__, cudaMemcpy( d_pos1[i], pos1[i], sizeof(unsigned int)*(pair_num_gpu[i]+1), cudaMemcpyHostToDevice ) );
   	cudaCheckError( __LINE__, cudaMemcpy( d_pos2[i], pos2[i], sizeof(unsigned int)*(pair_num_gpu[i]+1), cudaMemcpyHostToDevice ) );
   	cudaCheckError( __LINE__, cudaMemcpy( d_pos_matrix[i], pos_matrix[i], sizeof(unsigned int)*(pair_num_gpu[i]+1), cudaMemcpyHostToDevice ) );
	cudaCheckError( __LINE__, cudaMemcpy( d_dim_matrix[i], dim_matrix[i], sizeof(unsigned int)*(pair_num_gpu[i]+1), cudaMemcpyHostToDevice ) );
}

void nw_gpu_destroy(int stream_num)
{
    /* GPU memory allocation */
    int i = stream_num;
    cudaCheckError( __LINE__, cudaFree(d_sequence_set1[i]) );
    cudaCheckError( __LINE__, cudaFree(d_sequence_set2[i]) );
    cudaCheckError( __LINE__, cudaFree(d_score_matrix[i]) );
    cudaCheckError( __LINE__, cudaFree(d_pos1[i]) );
    cudaCheckError( __LINE__, cudaFree(d_pos2[i]) );
    cudaCheckError( __LINE__, cudaFree(d_pos_matrix[i]) );
    cudaCheckError( __LINE__, cudaFree(d_dim_matrix[i]) );
}

void nw_gpu(char * sequence_set1, char * sequence_set2, unsigned int * pos1, unsigned int * pos2, 
			int * score_matrix, unsigned int * pos_matrix, unsigned int pair_num,
			int * d_score_matrix, cudaStream_t stream, int stream_num, int kernel_type)
{
    cudaError_t ce;
	//int i = stream_num;
	//printf("Kernel type: %d\n", kernel_type);
	switch(kernel_type) {
		case 0: nw_cuda_diagonal(stream, stream_num);
				break;
		case 1: nw_cuda_tile(stream, stream_num);
				break;
		default:
				break;
	}
	ce = cudaGetLastError();
	if ( ce != cudaSuccess) {
		fprintf(stdout, "Error: %s\n", cudaGetErrorString(ce));
	}
}

void nw_gpu_copyback(int *score_matrix, int *d_score_matrix, unsigned int *pos_matrix, unsigned int pair_num, int stream_num)
{
	int i = stream_num;
	/* Memcpy to host */
	printf("Stream %d : %d pairs\n", i, pair_num);
    cudaCheckError(__LINE__,cudaMemcpy(score_matrix,d_score_matrix,sizeof(int)*pos_matrix[pair_num],cudaMemcpyDeviceToHost ) );
}

void nw_cuda_diagonal( cudaStream_t stream, int stream_num)
{
	int i = stream_num;
	needleman_cuda_diagonal<<<config.num_blocks, config.num_threads, 0, stream>>>(	
							d_sequence_set1[i], d_sequence_set2[i], d_pos1[i], d_pos2[i], 
							d_score_matrix[i], d_pos_matrix[i],	pair_num_gpu[i], config.penalty);
}

void nw_cuda_tile( cudaStream_t stream, int stream_num)
{
	int maxLength = config.length;
	int i = stream_num;
	int tile_size = TILE_SIZE;
	int iteration = maxLength / tile_size + 1;
	if ( maxLength%tile_size==0 )
		iteration--;
	dim3 dimGrid(1,1);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	needleman_cuda_init<<< pair_num_gpu[i], 256, 0, stream>>>(d_score_matrix[i], d_pos_matrix[i], d_dim_matrix[i], config.penalty);
	//process top-left matrix
	for( int j = 1; j <= iteration; ++j) {
		dimGrid.x = pair_num_gpu[i];
		dimGrid.y = j;
		needleman_cuda_tile_upleft<<<config.num_blocks, config.num_threads, 0, stream>>>(
									d_sequence_set1[i], d_sequence_set2[i], d_pos1[i], d_pos2[i],
									d_score_matrix[i], d_pos_matrix[i], d_dim_matrix[i], pair_num_gpu[i], j, config.penalty);
	}
	//process bottom-right matrix
	for( int j = iteration - 1; j >= 1 ; j--){
    	dimGrid.x = pair_num_gpu[i];
		dimGrid.y = j;
		needleman_cuda_tile_bottomright<<<config.num_blocks, config.num_threads, 0, stream>>>(
										d_sequence_set1[i], d_sequence_set2[i], d_pos1[i], d_pos2[i],
										d_score_matrix[i], d_pos_matrix[i], d_dim_matrix[i], pair_num_gpu[i], j, config.penalty);
	}
}


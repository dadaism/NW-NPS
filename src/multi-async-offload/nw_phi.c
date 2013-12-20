/*
* File: nw_gpu.cu
* Author: Da Li
* Email:  da.li@mail.missouri.edu
* Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
*
* Description: This file defines all the wrapper functions GPU implementations.
*
*/

#include "nw_phi.h"
#include "nw_kernel_phi.c"



void nw_phi(char * sequence_set1, char * sequence_set2, unsigned int * pos1, unsigned int * pos2, 
			int * score_matrix, unsigned int * pos_matrix, unsigned int pair_num,
			int stream_num, int kernel_type)
{
	//printf("Kernel type: %d\n", kernel_type);
	switch(kernel_type) {
		//case 0: nw_cuda_diagonal(stream, stream_num);
		//		break;
		//case 1: nw_cuda_tile(stream, stream_num);
		//		break;
		default:
				break;
	}
}
/*

void nw_cuda_diagonal( cudaStream_t stream, int stream_num)
{
	int i = stream_num;
	needleman_cuda_diagonal<<<config.num_blocks, config.num_threads, 0, stream>>>(	
							d_sequence_set1[i], d_sequence_set2[i], d_pos1[i], d_pos2[i], 
							d_score_matrix[i], d_pos_matrix[i],	pair_num[i], config.penalty);
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
    needleman_cuda_init<<< pair_num[i], 256, 0, stream>>>(d_score_matrix[i], d_pos_matrix[i], d_dim_matrix[i], config.penalty);
    //process top-left matrix
    for( int j = 1; j <= iteration; ++j) {
        dimGrid.x = pair_num[i];
        dimGrid.y = j;
        needleman_cuda_tile_upleft<<<config.num_blocks, config.num_threads, 0, stream>>>(
									d_sequence_set1[i], d_sequence_set2[i], d_pos1[i], d_pos2[i],
                                    d_score_matrix[i], d_pos_matrix[i], d_dim_matrix[i], pair_num[i], j, config.penalty);
    }
    //process bottom-right matrix
    for( int j = iteration - 1; j >= 1 ; j--){
        dimGrid.x = pair_num[i];
        dimGrid.y = j;
        needleman_cuda_tile_bottomright<<<config.num_blocks, config.num_threads, 0, stream>>>(
										d_sequence_set1[i], d_sequence_set2[i], d_pos1[i], d_pos2[i],
                                        d_score_matrix[i], d_pos_matrix[i], d_dim_matrix[i], pair_num[i], j, config.penalty);
    }

}
*/

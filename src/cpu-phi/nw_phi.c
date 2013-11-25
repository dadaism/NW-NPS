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

//#include "nw_kernel_diagonal.cu"
//#include "nw_kernel_tile.cu"

void nw_gpu_destroy(int stream_num)
{
    /* GPU memory allocation */
   /* int i = stream_num;
    cudaCheckError( __LINE__, cudaFree(d_sequence_set1[i]) );
    cudaCheckError( __LINE__, cudaFree(d_sequence_set2[i]) );
    cudaCheckError( __LINE__, cudaFree(d_score_matrix[i]) );
    cudaCheckError( __LINE__, cudaFree(d_pos1[i]) );
    cudaCheckError( __LINE__, cudaFree(d_pos2[i]) );
    cudaCheckError( __LINE__, cudaFree(d_pos_matrix[i]) );
    cudaCheckError( __LINE__, cudaFree(d_dim_matrix[i]) );
*/}
/*
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
}*/



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

#include "needle_kernel_rodinia.cu"
#include "needle_kernel_rodinia_opt.cu"
#include "needle_kernel_diagonal.cu"
#include "needle_kernel_tile.cu"
#include "needle_kernel_row.cu"

/* Global variable for Rodinia */
extern int * reference[MAX_SEQ_NUM];
extern int * reference_cuda[MAX_SEQ_NUM];
extern int * matrix_cuda[MAX_SEQ_NUM];
extern int max_cols, max_rows;

/* Global variable for other kernels */
extern unsigned int pos1[MAX_SEQ_NUM];
extern unsigned int pos2[MAX_SEQ_NUM];
extern unsigned int pos_matrix[MAX_SEQ_NUM];
extern unsigned int dim_matrix[MAX_SEQ_NUM];

extern char * d_sequence_set1;
extern char * d_sequence_set2;
extern unsigned int * d_pos1;
extern unsigned int * d_pos2;
extern int * d_score_matrix;
extern unsigned int * d_pos_matrix;
extern unsigned int * d_dim_matrix;
extern int pair_num;
extern int penalty;
extern int maxLength;

inline void cudaCheckError(int line, cudaError_t ce)
{ 
    if (ce != cudaSuccess){
        printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
        exit(1);
    }
}

void nw_gpu(char * sequence_set1, char * sequence_set2, 
			unsigned int * pos1, unsigned int * pos2, 
			int * score_matrix, unsigned int * pos_matrix,
			unsigned int * dim_matrix, int mem_type, 
			int kernel_type, int dev_num)
{
    cudaError_t ce;

	cudaCheckError( __LINE__, cudaSetDevice(dev_num) );

	/* GPU memory allocation */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set1, sizeof(char)*pos1[pair_num] ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set2, sizeof(char)*pos2[pair_num] ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_score_matrix, sizeof(int)*pos_matrix[pair_num]) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos1, sizeof(unsigned int)*(pair_num+1) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos2, sizeof(unsigned int)*(pair_num+1) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos_matrix, sizeof(unsigned int)*(pair_num+1) ) );	
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_dim_matrix, sizeof(unsigned int)*(pair_num+1) ) );	

	/* Memcpy to device */
	cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set1, sequence_set1, sizeof(char)*pos1[pair_num], cudaMemcpyHostToDevice ) );
   	cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set2, sequence_set2, sizeof(char)*pos2[pair_num], cudaMemcpyHostToDevice ) );
   	cudaCheckError( __LINE__, cudaMemcpy( d_pos1, pos1, sizeof(unsigned int)*(pair_num+1), cudaMemcpyHostToDevice ) );
   	cudaCheckError( __LINE__, cudaMemcpy( d_pos2, pos2, sizeof(unsigned int)*(pair_num+1), cudaMemcpyHostToDevice ) );
   	cudaCheckError( __LINE__, cudaMemcpy( d_pos_matrix, pos_matrix, sizeof(unsigned int)*(pair_num+1), cudaMemcpyHostToDevice ) );
   	cudaCheckError( __LINE__, cudaMemcpy( d_dim_matrix, dim_matrix, sizeof(unsigned int)*(pair_num+1), cudaMemcpyHostToDevice ) );
	
	if ( kernel_type==0 ) {	// Extra code for Rodinia kernel
		for (int i=0; i<pair_num; ++i){
			/* GPU memory allocation */
			cudaCheckError( __LINE__, cudaMalloc( (void**)&reference_cuda[i], sizeof(int)*max_cols*max_rows ) );
		
			/* Memcpy to device */
    		cudaCheckError( __LINE__, cudaMemcpy( d_score_matrix, score_matrix, sizeof(int)*pos_matrix[pair_num], cudaMemcpyHostToDevice ) );
			cudaCheckError( __LINE__,cudaMemcpy(reference_cuda[i], reference[i], sizeof(int)*max_cols*max_rows, cudaMemcpyHostToDevice) );
			matrix_cuda[i] = d_score_matrix+pos_matrix[i];
		}
	}
	//printf("Kernel type: %d\n", kernel_type);
	switch(kernel_type) {
		case 0: nw_cuda_rodinia();
				break;
		case 1: nw_cuda_rodinia_opt();
				break;
		case 2: nw_cuda_diagonal();
				break;
		case 3: nw_cuda_tile();
				break;
		case 4: nw_cuda_row();
				break;
		default:
				break;
	}
	ce = cudaGetLastError();
	if ( ce != cudaSuccess) {
		fprintf(stdout, "Error: %s\n", cudaGetErrorString(ce));
	}
	
	/* Memcpy to host */
    cudaCheckError( __LINE__, cudaMemcpy( score_matrix, d_score_matrix, sizeof(int)*pos_matrix[pair_num], cudaMemcpyDeviceToHost ) );


}

void nw_cuda_rodinia()
{
	dim3 dimGrid;
    dim3 dimBlock(BLOCK_SIZE, 1);
    int block_width = ( max_cols - 1 )/BLOCK_SIZE;
	// Sequencially compare multiple sequences
    for (int n=0; n<pair_num; ++n){
        //printf("Processing top-left matrix\n");
        //process top-left matrix
        for( int i = 1 ; i <= block_width ; i++){
            dimGrid.x = i;
            dimGrid.y = 1;
            needle_cuda_shared_1<<<dimGrid, dimBlock>>>(reference_cuda[n], 
														matrix_cuda[n],
														max_cols, -penalty, 
														i, block_width);
        }
        //printf("Processing bottom-right matrix\n");
        //process bottom-right matrix
        for( int i = block_width - 1  ; i >= 1 ; i--){
            dimGrid.x = i;
            dimGrid.y = 1;
            needle_cuda_shared_2<<<dimGrid, dimBlock>>>(reference_cuda[n], 
														matrix_cuda[n],
														max_cols, -penalty, 
														i, block_width);
        }
    }
}

void nw_cuda_rodinia_opt()
{
	// the threads in block should equal to the STRIDE_SIZE
    int tileSize = TILE_SIZE;
    int iteration = maxLength / tileSize + 1;
    if ( maxLength%tileSize==0 )
        iteration--;
    dim3 dimGrid(1,1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    for (int j=0; j<pair_num; ++j) {
        needleman_cuda_init<<< 1, 512>>>(d_score_matrix, d_pos_matrix, d_dim_matrix, penalty, j);
        //process top-left matrix
        for( int i = 1; i <= iteration; ++i) {
            dimGrid.y = i;
            needleman_cuda_tile_upleft<<<dimGrid, dimBlock>>>(d_sequence_set1+pos1[j], d_sequence_set2+pos2[j],
                                                              pos1[j+1]-pos1[j], pos2[j+1]-pos2[j],
                                                              d_score_matrix+pos_matrix[j], i, penalty, dim_matrix[j]+1);
		}
        //process bottom-right matrix
        for( int i = iteration - 1; i >= 1 ; i--){
            dimGrid.y = i;
			needleman_cuda_tile_bottomright<<<dimGrid, dimBlock>>>(d_sequence_set1+pos1[j], d_sequence_set2+pos2[j],
                                                                    pos1[j+1]-pos1[j], pos2[j+1]-pos2[j],
                                                                    d_score_matrix+pos_matrix[j], i, penalty, dim_matrix[j]+1);
        }
    }
}

void nw_cuda_diagonal()
{

	needleman_cuda_diagonal<<<pair_num,512>>>(	d_sequence_set1, d_sequence_set2,
                                       			d_pos1, d_pos2,
                                       			d_score_matrix, d_pos_matrix,
                                       			pair_num, penalty);

}


void nw_cuda_tile()
{
	//printf("The max length of sequence is %d\n", maxLength);
	//printf("Pair number is %d\n", pair_num);
	int tileSize = TILE_SIZE;
    int iteration = maxLength / tileSize + 1;
    if ( maxLength%tileSize==0 )
        iteration--;
    dim3 dimGrid(1,1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    needleman_cuda_init<<< pair_num, 512>>>(d_score_matrix, d_pos_matrix, d_dim_matrix, penalty);
    //process top-left matrix
    for( int i = 1; i <= iteration; ++i) {
        dimGrid.x = pair_num;
        dimGrid.y = i;
        needleman_cuda_tile_upleft<<<dimGrid, dimBlock>>>(d_sequence_set1, d_sequence_set2, d_pos1, d_pos2,
                                                        d_score_matrix, d_pos_matrix, d_dim_matrix, i, penalty);
    }
    //process bottom-right matrix
    for( int i = iteration - 1; i >= 1 ; i--){
        dimGrid.x = pair_num;
        dimGrid.y = i;
        needleman_cuda_tile_bottomright<<<dimGrid, dimBlock>>>(d_sequence_set1, d_sequence_set2, d_pos1, d_pos2,
                                                            d_score_matrix, d_pos_matrix, d_dim_matrix, i, penalty);
    }
	//printf("Exit nw_cuda_tile\n");
}

void nw_cuda_row()
{
	/* PAIR_IN_BLOCK and STRIDE_SIZE are defined in kernel file */
	dim3 dimGrid( pair_num/PAIR_IN_BLOCK, 1);
	dim3 dimBlock(STRIDE_SIZE,1);
	if ( pair_num % PAIR_IN_BLOCK != 0 )
		dimGrid.x = dimGrid.x + 1;
	/* the threads in block should equal to the STRIDE_SIZE */
	/* the number of pair should be less than #block * PAIR_IN_BLOCK */
	needleman_cuda_row<<<dimGrid, dimBlock>>>(	d_sequence_set1, d_sequence_set2,
												d_pos1, d_pos2,
												d_score_matrix, d_pos_matrix,
												pair_num, penalty);
}


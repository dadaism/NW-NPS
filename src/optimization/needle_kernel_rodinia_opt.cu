/*
* Copyright (c) 2013 Da Li, and NPS Lab, University of Missouri – Columbia.
* All rights reserved
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*    1. Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*    2. Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*    3. The name of the author or University of Missouri may not be used
*       to endorse or promote products derived from this source code
*       without specific prior written permission.
*    4. Conditions of any other entities that contributed to this are also
*       met. If a copyright notice is present from another entity, it must
*       be maintained in redistributions of the source code.
*
* THIS INTELLECTUAL PROPERTY (WHICH MAY INCLUDE BUT IS NOT LIMITED TO SOFTWARE,
* FIRMWARE, VHDL, etc) IS PROVIDED BY  THE AUTHOR AND UNIVERSITY OF MISSOURI
* ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
* TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR UNIVERSITY OF MISSOURI
* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS INTELLECTUAL PROPERTY, EVEN IF
* ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* */

/*
* File: needle_kernel_rodinia_opt.cu 
* Author: Da Li
* Email: da.li@mail.missouri.edu
* Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
*
* Description: This file defines optimized kernel functions from Rodinia benchmark.
*
*/


#include <stdio.h>
#include <cuda.h>
#include "needle.h"

#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

__global__ void needleman_cuda_init(int *score_matrix, unsigned int *pos_matrix, unsigned int *dim_matrix, int penalty, int pair_no)
{
	int tid = threadIdx.x;
	int *matrix = score_matrix+pos_matrix[pair_no];
	unsigned int row_size = dim_matrix[pair_no]+1;	// 1 element margin
	unsigned int stride = blockDim.x;

	int iteration = row_size / stride;
	// init first row
	for (int i=0; i<=iteration ; ++i)
	{	
		int index = (tid+stride*i);
		if ( index<row_size ){
			matrix[ index ] = index * penalty;
		}
	}

	// init first column
	for (int i=0; i<=iteration ; ++i)
	{
		int index = row_size * (tid+stride*i);
		if ( (tid+stride*i)<row_size){
			matrix[ index ] = (tid+stride*i)*penalty;
		}
	}
}

__global__ void needleman_cuda_tile_upleft( char *seq1, char *seq2, 
									   		int seq1_len, int seq2_len,
									   		int *matrix, int iter_no, int penalty,
											int row_size)
{
	// 4KB, seq1[], sqe2[], tile[][]
	__shared__ char s_seq1[TILE_SIZE];
	__shared__ char s_seq2[TILE_SIZE];
	__shared__ int s_tile[(TILE_SIZE+1)][(TILE_SIZE+1)];

	int tile_no = blockIdx.y;
	int tid = threadIdx.y*blockDim.x + threadIdx.x;

	// calculate index, what are index_x & index_y
	int index_x = TILE_SIZE*tile_no + 1;	// 2-D matrix starts from (1,1)
	int index_y = TILE_SIZE*(iter_no-1) - TILE_SIZE*tile_no + 1;

	// load seq1
	int seq_index = index_x - 1 + tid; 	
	if ( tid<TILE_SIZE && seq_index<seq1_len )
		s_seq1[tid] = seq1[seq_index];
	// load seq2
	seq_index = index_y - 1 + tid;
	if ( tid<TILE_SIZE && seq_index<seq2_len )
		s_seq2[tid] = seq2[seq_index];

	// load boundary of tile
	if ( tid<TILE_SIZE ){
		int index = (index_y-1)*row_size + index_x + tid;	// x-index in 1-D array
		s_tile[0][tid+1] = matrix[index];

		index = (index_y+tid)*row_size + index_x - 1;		// y-index in 1-D array
		s_tile[tid+1][0] = matrix[index];
	}
	if ( tid==0 ) {
		int index = (index_y-1)*row_size + index_x-1;
		s_tile[tid][0] = matrix[index];
	}
	__syncthreads();
	// calculate
	for( int i = 0 ; i < TILE_SIZE ; i++){   
	  if ( tid <= i ){
		  index_x =  tid + 1;
		  index_y =  i - tid + 1;
          s_tile[index_y][index_x] = maximum(s_tile[index_y-1][index_x] + penalty,	//	up
											s_tile[index_y][index_x-1] + penalty,	//	left
						 					s_tile[index_y-1][index_x-1]+blosum62[s_seq2[index_y-1]][s_seq1[index_x-1]]);
	  }
	  __syncthreads();  
    }
	for( int i = TILE_SIZE - 1 ; i >=0 ; i--){   
	  if ( tid <= i){
		  index_x =  tid + TILE_SIZE - i ;
		  index_y =  TILE_SIZE - tid;
          s_tile[index_y][index_x] = maximum(s_tile[index_y-1][index_x] + penalty,	//	up
											s_tile[index_y][index_x-1] + penalty,	//	left
						 					s_tile[index_y-1][index_x-1]+blosum62[s_seq2[index_y-1]][s_seq1[index_x-1]]);
	  }
	  __syncthreads();
  }
	// write to global mem
	index_x = TILE_SIZE*tile_no + 1 + threadIdx.x;	// 2-D matrix starts from (1,1)
	index_y = TILE_SIZE*(iter_no-1) - TILE_SIZE*tile_no + 1 + threadIdx.y;
	/* 1-1 mapping between threads and tile elements */
	matrix[index_x + index_y * row_size] = s_tile[threadIdx.y+1][threadIdx.x+1];
}

__global__ void needleman_cuda_tile_bottomright(char *seq1, char *seq2, 
									   			int seq1_len, int seq2_len,
									   			int *matrix, int iter_no, int penalty,
												int row_size)
{
	// 4KB, seq1[], sqe2[], tile[][]
	__shared__ char s_seq1[TILE_SIZE];
	__shared__ char s_seq2[TILE_SIZE];
	__shared__ int s_tile[(TILE_SIZE+1)][(TILE_SIZE+1)];

	int tile_no = blockIdx.y;
	int tid = threadIdx.y*blockDim.x + threadIdx.x;

	// calculate index
	//int index_x = TILE_SIZE*(tile_no+1) + 1;	// 2-D matrix starts from (1,1)
	int index_x = row_size - TILE_SIZE*iter_no + TILE_SIZE*tile_no;	// 2-D matrix starts from (1,1)
	int index_y = row_size - TILE_SIZE - TILE_SIZE*tile_no;

	// load seq1
	int seq_index = index_x -1 + tid; 	
	if ( tid<TILE_SIZE && seq_index<seq1_len )
		s_seq1[tid] = seq1[seq_index];
	// load seq2
	seq_index = index_y -1 + tid;
	if ( tid<TILE_SIZE && seq_index<seq2_len )
		s_seq2[tid] = seq2[seq_index];

	// load boundary of tile
	if ( tid<TILE_SIZE ) {
		int index = (index_y-1)*row_size + index_x + tid;	// x-index in 1-D array
		s_tile[0][tid+1] = matrix[index];

		index = (index_y+tid)*row_size + index_x - 1;		// y-index in 1-D array
		s_tile[tid+1][0] = matrix[index];
	}
	if ( tid==0 ) {
		int index = (index_y-1)*row_size + index_x-1;
		s_tile[tid][0] = matrix[index];
	}
	__syncthreads();
	// calculate
	for( int i = 0 ; i < TILE_SIZE ; i++){
		if ( tid <= i ){
			index_x =  tid + 1;
			index_y =  i - tid + 1;
			s_tile[index_y][index_x] = maximum(s_tile[index_y-1][index_x] + penalty,//	up
											s_tile[index_y][index_x-1] + penalty,	//	left
						 					s_tile[index_y-1][index_x-1]+blosum62[s_seq2[index_y-1]][s_seq1[index_x-1]]);
		}
		__syncthreads();  
	}
	for( int i = TILE_SIZE - 1 ; i >=0 ; i--){   
		if ( tid <= i){
			index_x =  tid + TILE_SIZE - i ;
			index_y =  TILE_SIZE - tid;
			s_tile[index_y][index_x] = maximum(s_tile[index_y-1][index_x] + penalty,//	up
											s_tile[index_y][index_x-1] + penalty,	//	left
											s_tile[index_y-1][index_x-1]+blosum62[s_seq2[index_y-1]][s_seq1[index_x-1]]);
		}
		__syncthreads();
	}

	// write to global mem
	//index_x = TILE_SIZE*(tile_no+1) + 1 + threadIdx.x;	// 2-D matrix starts from (1,1)
	index_x = row_size - TILE_SIZE*iter_no + TILE_SIZE*tile_no + threadIdx.x;
	index_y = row_size - TILE_SIZE- TILE_SIZE*tile_no + threadIdx.y;
	/* 1-1 mapping between threads and tile elements */
	matrix[index_x + index_y * row_size] = s_tile[threadIdx.y+1][threadIdx.x+1];
}
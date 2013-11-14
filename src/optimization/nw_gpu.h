/*
* File:  nw_gpu.h
* Author: Da Li
* Email:  da.li@mail.missouri.edu
* Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
*
* Description: This file has all the declarations of GPU related functions.
*
*/

#ifndef __NW_GPU_H__
#define __NW_GPU_H__

//#include "global.h"

void nw_cuda_rodinia();
void nw_cuda_rodinia_opt();
void nw_cuda_diagonal();
void nw_cuda_tile();
void nw_cuda_row();

void nw_gpu(char * sequence_set1, char * sequence_set2, 
            unsigned int * pos1, unsigned int * pos2, 
            int * score_matrix, unsigned int * pos_matrix, 
            unsigned int * dim_matrix, int mem_type, 
			int kernel_type, int dev_num);
#endif


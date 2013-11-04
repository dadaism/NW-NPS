/*
* File:  nw-nps.cu
* Author: Da Li
* Email:  da.li@mail.missouri.edu
* Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
*
* Description: This file is the framework of program.
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "global.h"
#include "nw_cpu.h"
#include "nw_gpu.h"

//#define DEBUG
//#define TRACEBACK

extern char blosum62_cpu[24][24];

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <length> <pair number> <penalty> <memory> <kernel> <device>\n", argv[0]);
    fprintf(stderr, "\t<length>  - x and y length\n");
    fprintf(stderr, "\t<pair number>  - number of pairs, 0 means using 70%% of GPU's memory\n");
    fprintf(stderr, "\t<penalty> - penalty (negative integer)\n");
    fprintf(stderr, "\t<memory> - 0: unpined 1: pinned\n");
    fprintf(stderr, "\t<kernel> - 0: Rodinia 1: Optimized Rodinia 2: DScan 3: TDScan 4: RScan\n");
    fprintf(stderr, "\t<device> - device ID\n");
    exit(1);
}

void output_config(int argc, char **argv)
{
    fprintf(stderr, "sequence length: %s\n", argv[1]);
    fprintf(stderr, "pair number: %d\n", pair_num);
    fprintf(stderr, "penalty: %s\n", argv[3]);
    fprintf(stderr, "memory: %s - 0: unpined 1: pinned 2: double buffering\n", argv[4]);
    fprintf(stderr, "kernel: %s - 0: Rodinia 1: Optimized Rodinia 2: DScan 3: TDScan 4: RScan\n", argv[5]);
    fprintf(stderr, "device: %s\n", argv[6]);
}

int validation(int *score_matrix_cpu, int *score_matrix, unsigned int length)
{
    unsigned int i = 0;
    while (i!=length){
        if ( score_matrix_cpu[i]==score_matrix[i] ){
            ++i;
            continue;
        }
        else {
            printf("i = %d\n",i);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv)
{
	char sequence_set1[MAX_SEQ_LEN*MAX_SEQ_NUM] = {0}, sequence_set2[MAX_SEQ_LEN*MAX_SEQ_NUM] = {0};
	int mem_type = 0, kernel_type = 2, dev_count = 0, dev_num = 0;
	int seq1_len, seq2_len, seq_len;
	int mem_per_pair = 0;
	int * score_matrix;
	
	/* Parse option */
	if ( argc>=7 ) {
		maxLength = seq_len = atoi(argv[1]);
		pair_num = atoi(argv[2]);
		penalty = atoi(argv[3]);
		mem_type = atoi(argv[4]);
		kernel_type = atoi(argv[5]);
		dev_count = atoi(argv[6]);
	}
	else {
		usage(argc, argv);
		exit(1);
	}
	/* Validate the input */
	if ( seq_len > MAX_SEQ_LEN )
	{
		printf("The maximum sequence length is %d\n", MAX_SEQ_LEN);
		exit(1);
	}
	if ( pair_num<0 )
	{
		printf("Number of pairs should not be negative\n");
		exit(1);
	}
	/* Choose between explicit number and 70% memory model */
	if ( pair_num==0 )
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, dev_num);
		mem_per_pair += sizeof(char)*seq_len*2 + sizeof(int)*(seq_len+1)*(seq_len+1);
		if ( kernel_type == 0 ) // NW from Rodinia needs more global memory 
			mem_per_pair += sizeof(int)*(seq_len+1)*(seq_len+1);
		pair_num = (int) ( 0.7 * (float)devProp.totalGlobalMem / (float)mem_per_pair );
		//printf("Memory per pair: %d\n", mem_per_pair);
		//printf("Total global memory on GPU: %d\n", devProp.totalGlobalMem);
	}
	if ( pair_num> MAX_SEQ_NUM )
	{
		printf("The maximum number of pairs is %d\n", MAX_SEQ_NUM);
		printf("You can modify it in global.h\n");
		printf("However, you may need to execute \"ulimit -s unlimited\" before run the code\n");
		exit(1);
	}
	cudaGetDeviceCount(&dev_count);
	if ( dev_num>=dev_count )
	{
		printf("%d device(s) available. Starting from 0.\n", dev_count);
		exit(1);
	}
	if ( mem_type>1 )
	{
		printf("You set %d as memory type\n", mem_type);
		printf("Current version only supports 0: unpinned 1: pinned memory\n");
		exit(1);
	}
	/* Generate data */
	srand ( 7 );
    pos_matrix[0] = pos1[0] = pos2[0] = 0;
	/* Output execution configuration */
	output_config(argc, argv);
	//printf("Total %d pairs\n", pair_num);
    for (int i=0; i<pair_num; ++i){
        //please define your own sequence 1
        seq1_len = seq_len; //64+rand() % 20;
        //printf("Seq1 length: %d\n", seq1_len);    
        for (int j=0; j<seq1_len; ++j)
            sequence_set1[ pos1[i] + j ] = rand() % 20 + 1;
        pos1[i+1] = pos1[i] + seq1_len;
        //please define your own sequence 2.
        seq2_len = seq_len;//64+rand() % 20;       
        //printf("Seq2 length: %d\n\n", seq2_len);      
        for (int j=0; j<seq2_len; ++j)
            sequence_set2[ pos2[i] +j ] = rand() % 20 + 1;
        pos2[i+1] = pos2[i] + seq2_len;
        //printf("Matrix size increase: %d\n", (seq1_len+1) * (seq2_len+1));
        pos_matrix[i+1] = pos_matrix[i] + (seq1_len+1) * (seq2_len+1);
		dim_matrix[i] = (unsigned int)ceil( (float)seq_len/TILE_SIZE )*TILE_SIZE;
    }

	if ( mem_type==0 )
		score_matrix = (int *)malloc( pos_matrix[pair_num]*sizeof(int) );
	else
		cudaHostAlloc( &score_matrix, pos_matrix[pair_num]*sizeof(int), cudaHostAllocDefault );
		
	if ( kernel_type==0 ) { // Extra code for Rodinia
		max_rows = max_cols = seq_len + 1;
		for ( int n=0; n<pair_num; ++n ) {
			if ( mem_type==0 )
				reference[n] = (int *)malloc( max_rows*max_cols*sizeof(int) );
			else
				cudaHostAlloc( &reference[n], max_rows*max_cols*sizeof(int), cudaHostAllocDefault );
			score_matrix[ pos_matrix[n] ] = 0;
			for (int i=1; i<max_rows; ++i)
				for (int j=1; j<max_cols; ++j) {
					reference[n][i*max_cols+j] = blosum62_cpu[ sequence_set1[ pos1[n]+j-1 ] ][ sequence_set2[ pos2[n]+i-1 ] ];
			}
			for (int i=1; i<max_rows; ++i)
				score_matrix[ pos_matrix[n] + i*max_cols ] = i * penalty;
			for (int j=1; j<max_cols; ++j)
				score_matrix[ pos_matrix[n] + j ] = j * penalty;
		}
	}
	/* Computation */
	nw_gpu(sequence_set1, sequence_set2, pos1, pos2, score_matrix, pos_matrix, dim_matrix, mem_type, kernel_type, dev_num);

	/* Validation */
#ifdef DEBUG
    int *score_matrix_cpu = (int *)malloc( pos_matrix[pair_num]*sizeof(int));
    needleman_cpu(sequence_set1, sequence_set2, pos1, pos2, score_matrix_cpu, pos_matrix, pair_num, penalty);
	if ( validation(score_matrix_cpu, score_matrix, pos_matrix[pair_num]) )
        printf("Validation: PASS\n");
    else
        printf("Validation: FAIL\n");
#endif

	/* Trace Back */
#ifdef TRACEBACK
	for (int i=0; i<pair_num; ++i) {
		//printf("For traceback, sequence langth: %d\n", seq_len);
		traceBack(	score_matrix+pos_matrix[i], sequence_set1+pos1[i], sequence_set2+pos2[i], 
					dim_matrix[i]+1, dim_matrix[i]+1, seq_len, seq_len, penalty	);
	}
	printf("Traceback finished\n");

#endif
}

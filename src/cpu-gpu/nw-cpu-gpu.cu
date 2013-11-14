/*
* File:  nw-multi-stream.cu
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

//#define TRACEBACK

extern char blosum62_cpu[24][24];
extern int tile_size;

void init_conf() {
	config.debug = false;
	config.device = 0;
	config.kernel = 0;
	config.cpu_pairs = 20;
	config.gpu_pairs = 80;
	config.num_streams = 1;
	config.num_blocks = 28;
	config.num_threads = 32;
	config.length = 1600;
	config.penalty = -10;
}

void init_device(int device) {
	cudaSetDevice(device);
	cudaFree(0);
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "\nUsage: %s [options]\n", argv[0]);
    fprintf(stderr, "\t[--length|-l <length> ] - x and y length\n");
    fprintf(stderr, "\t[--penalty|-p <penalty>] - penalty (negative integer)\n");
    fprintf(stderr, "\t[--cpu_pairs|-c <number>] - number of pairs on cpu\n");
    fprintf(stderr, "\t[--gpu_pairs|-g <number>] - number of pairs on gpu\n");
    fprintf(stderr, "\t[--device|-d <device> ]- device ID\n");
    fprintf(stderr, "\t[--kernel|-k <kernel type> ]- 0: diagonal 1: tile\n");
    fprintf(stderr, "\t[--num_blocks|-b <blocks> ]- blocks number per grid\n");
    fprintf(stderr, "\t[--num_threads|-t <threads> ]- threads number per block\n");
    fprintf(stderr, "\t[--debug]- 0: no validation (default) 1: validation\n");
    exit(1);
}

void print_config()
{
	fprintf(stderr, "=============== Configuration ================\n");
    fprintf(stderr, "device = %d\n", config.device);
	fprintf(stderr, "kernel = %d\n", config.kernel);
	if ( config.kernel == 1 )
		fprintf(stderr, "tile size = %d\n", tile_size);
    fprintf(stderr, "On CPU - sequence number = %d\n", config.cpu_pairs);
    fprintf(stderr, "On GPU - sequence number = %d\n", config.gpu_pairs);
    fprintf(stderr, "       - stream number = %d\n", config.num_streams);
	fprintf(stderr, "sequence length = %d\n", config.length);
    fprintf(stderr, "penalty = %d\n", config.penalty);
    fprintf(stderr, "block number = %d\n", config.num_blocks);
    fprintf(stderr, "thread number = %d\n", config.num_threads);
    fprintf(stderr, "debug = %d\n", config.debug);
	printf("==============================================\n");
}

void validate_config()
{
    if ( config.length % tile_size != 0 &&
         config.kernel == 1 ) {
        fprintf(stderr, "Tile kernel used.\nSequence length should be multiple times of tile size %d\n", tile_size);
        exit(1);
    }
}

int validation(int *score_matrix_cpu, int *score_matrix, unsigned int length)
{
    unsigned int i = 0;
	//printf("Length : %d\n", length);
    while (i!=length){
        if ( score_matrix_cpu[i]==score_matrix[i] ){
            ++i;
            continue;
        }
        else {
            printf("On GPU: score_matrix[%d] = %d\n", i, score_matrix[i]);
            printf("On CPU: score_matrix[%d] = %d\n", i, score_matrix_cpu[i]);
            return 0;
        }
    }
    return 1;
}

int parse_arguments(int argc, char **argv)
{
	int i = 1;
	if (argc<2) {
		usage(argc, argv);
		return 0;
	}
	while(i<argc) {
		if(strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"device number missing.\n");
				return 0 ;
			}
			config.device = atoi(argv[i]);
		}else if(strcmp(argv[i], "--debug") == 0){
			config.debug = 1;
		}else if(strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--kernel") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"device number missing.\n");
				return 0 ;
			}
			config.kernel = atoi(argv[i]);
		}else if(strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--num_threads") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"thread number missing.\n");
				return 0 ;
			}
			config.num_threads = atoi(argv[i]);
		}else if(strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--num_blocks") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"block number missing.\n");
				return 0 ;
			}
			config.num_blocks = atoi(argv[i]);
		}else if(strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--penalty") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"penalty score missing.\n");
				return 0 ;
			}
			config.penalty = atoi(argv[i]);
		}else if(strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--cpu_pairs") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"sequence length missing.\n");
				return 0 ;
			}
			config.cpu_pairs = atoi(argv[i]);
		}else if(strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--gpu_pairs") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"pair number on gpu missing.\n");
				return 0 ;
			}
			config.gpu_pairs = atoi(argv[i]);
		}else if(strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--lengths") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"sequence length missing.\n");
				return 0 ;
			}
			config.length = atoi(argv[i]);
			if ( config.length > MAX_SEQ_LEN ) {
				fprintf(stderr,"The maximum seqence length is %d\n", MAX_SEQ_LEN);
				return 0;
			}
		}
		else {
			fprintf(stderr, "Unrecognized option : %s", argv[i]);
			return 0;
		}
		i++;
	}
	return 1;
}

int main(int argc, char **argv)
{
	double s_time, e_time;
	init_conf();
	while(!parse_arguments(argc, argv)) usage(argc, argv);

	print_config();
	validate_config();
	int dev_num = config.device;
	int	penalty = config.penalty;
	int seq1_len, seq2_len;
	int seq_len = config.length;
	DEBUG = config.debug;

	cudaStream_t stream[config.num_streams];

	int *score_matrix[config.num_streams];
	srand ( 7 );
	for (int k=0; k<2;++k) {
    	pos_matrix[k][0] = pos1[k][0] = pos2[k][0] = 0;
		if ( k==0 )
			pair_num[k] = config.cpu_pairs;
		else
			pair_num[k] = config.gpu_pairs;
			
    	for (int i=0; i<pair_num[k]; ++i){
        	//please define your own sequence 1
        	seq1_len = seq_len; //64+rand() % 20;
        	//printf("Seq1 length: %d\n", seq1_len);    
        	for (int j=0; j<seq1_len; ++j)
            	sequence_set1[k][ pos1[k][i] + j ] = rand() % 20 + 1;
        	pos1[k][i+1] = pos1[k][i] + seq1_len;
        	//please define your own sequence 2.
        	seq2_len = seq_len;//64+rand() % 20;       
        	//printf("Seq2 length: %d\n\n", seq2_len);      
        	for (int j=0; j<seq2_len; ++j)
            	sequence_set2[k][ pos2[k][i] +j ] = rand() % 20 + 1;
        	pos2[k][i+1] = pos2[k][i] + seq2_len;
        	//printf("Matrix size increase: %d\n", (seq1_len+1) * (seq2_len+1));
        	pos_matrix[k][i+1] = pos_matrix[k][i] + (seq1_len+1) * (seq2_len+1);
			dim_matrix[k][i] = (unsigned int)ceil( (float)seq_len/TILE_SIZE )*TILE_SIZE;
    	}
		
		score_matrix[k] = (int *)malloc( pos_matrix[k][pair_num[k]]*sizeof(int) );
	}
	/* initialize the GPU */
	s_time = gettime();
	init_device( dev_num );
	e_time = gettime();
	printf("Initialize GPU : %fs\n", e_time - s_time);

	s_time = gettime();
	for (int i=0; i<config.num_streams; ++i) {
		nw_gpu_allocate(i+1);
		cudaStreamCreate( &(stream[i]) );
	}
	e_time = gettime();
	printf("Memory allocation and copy on GPU : %fs\n", e_time - s_time);
	
	s_time = gettime();
	#pragma omp parallel for
	for (int i=1; i<=config.num_streams; ++i ) {
		if ( pair_num[i]!=0 ) {
			nw_gpu(sequence_set1[i], sequence_set2[i], pos1[i], pos2[i], score_matrix[i], pos_matrix[i], pair_num[i], d_score_matrix[i], stream[i], i, config.kernel);
		}
	}
    needleman_cpu(sequence_set1[0], sequence_set2[0], pos1[0], pos2[0], score_matrix[0], pos_matrix[0], pair_num[0], penalty);
	
	for (int i=1; i<=config.num_streams; ++i ) {
		if ( pair_num[i]!=0 ) {
			nw_gpu_copyback(score_matrix[i], d_score_matrix[i], pos_matrix[i], pair_num[i], i);
		}
	}
	e_time = gettime();
	printf("Computation on CPU & GPU : %fs\n", e_time - s_time);
	
	if (DEBUG) {
		for ( int i=0; i<config.num_streams; ++i) {
    		int *score_matrix_cpu = (int *)malloc( pos_matrix[i][pair_num[i]]*sizeof(int));
    		needleman_cpu(sequence_set1[i], sequence_set2[i], pos1[i], pos2[i], score_matrix_cpu, pos_matrix[i], pair_num[i], penalty);
			if ( validation(score_matrix_cpu, score_matrix[i], pos_matrix[i][pair_num[i]]) )
        		printf("Stream %d - Validation: PASS\n", i);
    		else
        		printf("Stream %d - Validation: FAIL\n", i);
			free(score_matrix_cpu);
		}
	}
	printf("\n");
	for (int i=1; i<config.num_streams; ++i ) {
		nw_gpu_destroy(i);
		cudaStreamDestroy ( stream[i] );
	}
}

/*
* File:  global.h
 * Author: Da Li
 * Email:  da.li@mail.missouri.edu
 * Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
*
* Description: This file defines all the global variables.
*
*/

#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define MAX_SEQ_LEN 2000
#define MAX_SEQ_NUM 400
#define MAX_STREAM 20

struct _CONF_ {
	bool debug;
	int device;
	int kernel;
	int num_threads;
	int num_blocks;
	int num_streams;
	int num_pairs[MAX_STREAM];
	int length;
	int penalty;
	int repeat;
};

extern struct _CONF_ config;
extern int DEBUG;

extern char sequence_set1[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ];
extern char sequence_set2[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ];
extern unsigned int pos1[MAX_STREAM][MAX_SEQ_NUM];
extern unsigned int pos2[MAX_STREAM][MAX_SEQ_NUM];
extern unsigned int pos_matrix[MAX_STREAM][MAX_SEQ_NUM];
extern unsigned int dim_matrix[MAX_STREAM][MAX_SEQ_NUM];
extern char * d_sequence_set1[MAX_STREAM];
extern char * d_sequence_set2[MAX_STREAM];
extern unsigned int * d_pos1[MAX_STREAM];
extern unsigned int * d_pos2[MAX_STREAM];
extern int * d_score_matrix[MAX_STREAM];
extern unsigned int * d_pos_matrix[MAX_STREAM];
extern unsigned int * d_dim_matrix[MAX_STREAM];
extern int pair_num[MAX_STREAM];

double gettime();

#endif

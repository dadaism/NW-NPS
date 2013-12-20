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
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define MAX_SEQ_LEN 2000
#define MAX_SEQ_NUM 200
#define TILE_SIZE 16
#define MAX_STREAM 8

struct _CONF_ {
	bool debug;
	int device;
	int kernel;
	int num_streams;
	int num_pairs[MAX_STREAM];
	int cpu_threads;
	int phi_threads;
	int fraction;
	int length;
	int penalty;
};

extern struct _CONF_ config;
extern int DEBUG;

extern char sequence_set1[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ];
extern char sequence_set1_cpu[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ];
extern char sequence_set2[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ];
extern char sequence_set2_cpu[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ];
extern unsigned int pos1[MAX_STREAM][MAX_SEQ_NUM];
extern unsigned int pos1_cpu[MAX_STREAM][MAX_SEQ_NUM];
extern unsigned int pos2[MAX_STREAM][MAX_SEQ_NUM];
extern unsigned int pos2_cpu[MAX_STREAM][MAX_SEQ_NUM];
extern unsigned int pos_matrix[MAX_STREAM][MAX_SEQ_NUM];
extern unsigned int pos_matrix_cpu[MAX_STREAM][MAX_SEQ_NUM];
extern int pair_num[MAX_STREAM];
extern int pair_num_phi[MAX_STREAM];
extern int pair_num_cpu[MAX_STREAM];

double gettime();

#endif

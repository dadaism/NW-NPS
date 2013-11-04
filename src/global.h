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

#define MAX_SEQ_LEN 2000
#define MAX_SEQ_NUM 2800
#define TILE_SIZE 16

int * reference[MAX_SEQ_NUM];
int * reference_cuda[MAX_SEQ_NUM];
int * matrix_cuda[MAX_SEQ_NUM];
int max_cols, max_rows;

unsigned int pos1[MAX_SEQ_NUM] = {0};
unsigned int pos2[MAX_SEQ_NUM] = {0};
unsigned int pos_matrix[MAX_SEQ_NUM] = {0};
unsigned int dim_matrix[MAX_SEQ_NUM] = {0};

char * d_sequence_set1;
char * d_sequence_set2;
unsigned int * d_pos1;
unsigned int * d_pos2;
int * d_score_matrix;
unsigned int * d_pos_matrix;
unsigned int * d_dim_matrix;
int pair_num;
int penalty;
int maxLength;

#endif

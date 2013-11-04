#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>

#define MAX_SEQ_LEN 4800

int main(char **argv, int argc)
{

	unsigned int dist[MAX_SEQ_LEN] = {};
	char buffer[MAX_SEQ_LEN] = {};
	int i;
	int max_len = 0;
	int min_len = MAX_SEQ_LEN;
	int seq_len = 0;
	
	memset(dist, 0, MAX_SEQ_LEN*sizeof(unsigned int));
	memset(buffer, 0, MAX_SEQ_LEN*sizeof(char));


	while ( fgets(buffer, MAX_SEQ_LEN, stdin)!=NULL ) {
		if ( fgets(buffer, MAX_SEQ_LEN, stdin)==NULL ) {
			printf("Error input. Exit\n");
			exit(0);
		}
		seq_len = strlen(buffer);
		if ( seq_len<min_len )
			min_len = seq_len;
		if ( seq_len>max_len )
			max_len = seq_len;
		//printf("%d\n", seq_len);
		dist[seq_len]++;
	}

	for ( i=min_len; i<=max_len; ++i) {
		printf("%d %d\n", i, dist[i]);
	}
	return 0;
}

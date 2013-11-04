#!/bin/bash

for i in 0 1 2 3 4
	do
		../bin/nw-nps 1536 32 -10 1 $i
	done


#!/bin/bash
echo python3 --version
echo ${params}
echo ${BASH_ARGV[*]}

python object_tracker.py ${BASH_ARGV[*]}
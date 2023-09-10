#!/bin/bash
# get all filename in specified path

path=./matrix_case/benchmark
files=$(ls $path)
mkdir -p "result"
for filename in $files
do
    filepath=$path"/"$filename
    echo $filepath
    ./main "-m" $filepath "-r" ./perf_analysis/result
done
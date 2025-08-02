#!/bin/bash

divide_num=192
test_cutdown=9999999

for data_slice_index in $(seq 0 $(($divide_num - 1)))
do
    python ovd_resample.py --divide_num $divide_num --data_slice_index $data_slice_index --test_cutdown $test_cutdown &
done

wait
echo "All resampling processes completed."
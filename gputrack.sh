#!/bin/bash

all_time_high=0
total_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)

while true; do 
    used_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    percent=$(awk "BEGIN {printf \"%.2f\", ($used_mem / $total_mem) * 100}")

    if [ "$used_mem" -gt "$all_time_high" ]; then
        all_time_high=$used_mem
    fi

    high_percent=$(awk "BEGIN {printf \"%.2f\", ($all_time_high / $total_mem) * 100}")
    echo "Current Usage: $used_mem MiB (${percent}%) | All-Time High: $all_time_high MiB (${high_percent}%)"
    
    sleep 0.5
done

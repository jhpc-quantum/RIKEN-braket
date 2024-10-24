#!/bin/bash
chmod +x UnitTest*.sh

total_tests=16
success_count=0 

for i in $(seq -w 1 $total_tests); do
    ./UnitTest"$i".sh
    
    if [ $? -eq 0 ]; then
        ((success_count++))
    fi
done

echo "$success_count/$total_tests tests succeeded."

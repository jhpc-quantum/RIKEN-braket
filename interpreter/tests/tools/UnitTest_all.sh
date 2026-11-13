#!/bin/bash
chmod +x UnitTest*.sh
rm -fr ../result

total_tests=0
success_count=0 
export LD_LIBRARY_PATH=../../../../qe-qasm/build/lib:$LD_LIBRARY_PATH

for i in UnitTest0*.sh; do
    ./$i    
    if [ $? -eq 0 ]; then
        ((success_count++))
    fi
    ((total_tests++))
done

echo "$success_count/$total_tests tests succeeded."

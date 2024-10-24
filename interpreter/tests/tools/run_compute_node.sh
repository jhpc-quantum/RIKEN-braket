#!/bin/bash
#PJM -N "qasminterpreter_sample"            # job title
#PJM -S                                     # Instructions for outputting statistics file
#PJM -L "elapse=04:00:00"                   # Elapsed time limit for the job
#PJM -L "rscgrp=small"                      # Specify the number of nodes resource group
#PJM -L "node=1"                            # Number of nodes
#PJM --mpi "max-proc-per-node=4"            # Number of processes the program creates per node
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004 # VOLUME of data area used by the job
#PJM -g "xxxx"                              # Issue Group Designation

# Loading Spack environment
. ../../../../spack/share/spack/setup-env.sh

# gmp
spack load /rx544si
# mpfr
spack load /7whj32d
# mpc
#spack load mpc@1.2.1
spack load /5w5gp5k
# bison
spack load /7d5m4dq
# flex
spack load /o4lwh46
# mpi
spack load fujitsu-mpi%gcc@12.2.0
# boost
spack load /epjk46e

export LD_LIBRARY_PATH=../../../../qe-qasm/build/lib:$LD_LIBRARY_PATH

./UnitTest_all.sh
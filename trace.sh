#!/bin/bash

export EXTRAE_CONFIG_FILE=extrae.xml
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libomptrace.so    # OpenMP
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so   # MPI (C-based)
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so  # MPI (Fortran-based)
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libompitrace.so  # MPI+OpenMP (C-based)
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libompitracef.so # MPI+OpenMP (Fortran-based)
export LD_PRELOAD=${EXTRAE_HOME}/lib/libcudatrace.so # CUDA

$@

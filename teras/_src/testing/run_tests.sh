#!/bin/bash

FILE_PATH=${1:-../}
WARNING_OPTION=${2:---disable-warnings}
VERBOSE_OPTION=${3:--v}

run_tests(){
    echo "Testing with backend: $1"
    export KERAS_BACKEND=$1 && pytest "$FILE_PATH" "$WARNING_OPTION" "$VERBOSE_OPTION"
}

BACKENDS=(tensorflow jax torch)
for backend in "${BACKENDS[@]}"; do run_tests "$backend"; done

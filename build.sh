#!/bin/bash

mkdir -p build/$1
cmake --preset $1
cmake --build build/$1
ln -sf build/$1/compile_commands.json compile_commands.json

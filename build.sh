#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <preset>"
    exit 1
fi

mkdir -p "build/$1"
cmake --preset "$1"
cmake --build "build/$1"
ln -sf "build/$1/compile_commands.json" compile_commands.json

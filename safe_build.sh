#!/bin/bash
set -e

rm -Rf build
mkdir build
cd build

# Default prefix if not set
if [ -z "$PREFIX" ]; then
  PREFIX=$(pwd)/../dist
fi

echo "Installing to $PREFIX"

cmake ../src -DCMAKE_INSTALL_PREFIX=$PREFIX
make -j$(nproc)
make install
cd ..
# Do not remove ext or build

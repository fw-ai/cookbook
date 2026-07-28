#!/bin/bash
# Build the vendor implementation shared library

cd /app/impl
rm -rf build
mkdir -p build
cd build
cmake ../ 2>&1
make 2>&1
retcode=$?
cd /app

if [ $retcode -ne 0 ]; then
    echo "[ERROR] Failed to build implementation library"
    exit 1
fi

echo "[SUCCESS] Built implementation library"
exit 0

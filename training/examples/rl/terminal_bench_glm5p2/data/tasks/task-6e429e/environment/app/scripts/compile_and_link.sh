#!/bin/bash
# Find vendor library and build the test driver against it

root=$(pwd)
libroot=$root/lib

echo -n "Looking for implementation library in $libroot... "
numLibs=$(ls $libroot/libbvt_11_*_???.so 2>/dev/null | wc -l)
if [ $numLibs -eq 0 ]; then
    echo ""
    echo "[ERROR] Could not find implementation library in $libroot."
    echo "Library must be named libbvt_11_<vendor>_<three-digit-seq>.so"
    exit 1
elif [ $numLibs -gt 1 ]; then
    echo ""
    echo "[ERROR] Multiple matching libraries found in $libroot."
    exit 1
fi

libstring=$(ls $libroot/libbvt_11_*_???.so)
echo "[FOUND] $libstring"

export BVT_IMPL_LIB=$libstring

echo "Building test driver against $libstring..."
rm -rf build
mkdir -p build
cd build
cmake ../ > /dev/null 2>&1
make 2>&1
cd $root

if [ ! -f "bin/validate11" ]; then
    echo "[ERROR] Failed to build test driver executable"
    exit 1
fi
echo "[SUCCESS] Built test driver in $root/bin"
exit 0

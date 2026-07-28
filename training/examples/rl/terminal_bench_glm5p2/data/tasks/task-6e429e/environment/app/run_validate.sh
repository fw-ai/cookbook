#!/bin/bash
# BVT 1:1 Validation Pipeline
# Builds vendor library, links test driver, runs validation, checks results.

success=0
failure=1

# Build implementation library
echo "========================================="
echo " Building Implementation Library"
echo "========================================="
scripts/build_impl.sh
retcode=$?
if [[ $retcode != 0 ]]; then
    exit $failure
fi

# Compile and link test driver against vendor library
echo ""
echo "========================================="
echo " Compiling Test Driver"
echo "========================================="
scripts/compile_and_link.sh
retcode=$?
if [[ $retcode != 0 ]]; then
    exit $failure
fi

# Set dynamic library path
export LD_LIBRARY_PATH=$(pwd)/lib

# Run validation tests
echo ""
echo "========================================="
echo " Running Validation Tests"
echo "========================================="
scripts/run_testdriver.sh
retcode=$?
if [[ $retcode != 0 ]]; then
    exit $failure
fi

# Sanity checks on output logs
echo ""
echo "========================================="
echo " Sanity Checking Output Logs"
echo "========================================="
outputDir="validation"

for input in enroll verif match
do
    numInputLines=$(wc -l < input/$input.txt)
    numLogLines=$(sed '1d' $outputDir/$input.log | wc -l)

    if [ "$numInputLines" != "$numLogLines" ]; then
        echo "[ERROR] $outputDir/$input.log has wrong number of lines."
        echo "  Expected $numInputLines data lines, got $numLogLines."
        exit $failure
    fi

    # Check return codes and template sizes
    if [ "$input" == "enroll" ] || [ "$input" == "verif" ]; then
        numFail=$(sed '1d' $outputDir/$input.log | awk '{ if($3!=0) print }' | wc -l)
        if [ "$numFail" != "0" ]; then
            echo "[WARNING] Non-successful return codes in $input.log ($numFail entries)"
        fi

        # Validate template sizes are non-zero
        numZeroSize=$(sed '1d' $outputDir/$input.log | awk '{ if($2+0 == 0) print }' | wc -l)
        if [ "$numZeroSize" -gt "0" ]; then
            echo "[ERROR] $numZeroSize templates in $input.log have zero size."
            echo "  createTemplate must produce non-empty templates."
            exit $failure
        fi
    fi

    if [ "$input" == "match" ]; then
        # Check that at least 50% of match scores are unique
        minUniqScores=$(echo "$numInputLines * 0.5" | bc | awk '{printf("%d\n",$1 + 0.5)}')
        numUniqScores=$(sed '1d' $outputDir/$input.log | awk '{ print $4 }' | sort -u | wc -l)
        if [ "$numUniqScores" -lt "$minUniqScores" ]; then
            echo "[ERROR] Only $numUniqScores unique match scores found."
            echo "  Minimum $minUniqScores required (at least 50% unique)."
            exit $failure
        fi

        # Check for negative scores
        numNegScores=$(sed '1d' $outputDir/$input.log | awk '{ if($4==0 && ($3+0)<0) print }' | wc -l)
        if [ "$numNegScores" -gt "0" ]; then
            echo "[ERROR] $numNegScores negative match scores detected."
            echo "  Scores must be non-negative per the BVT specification."
            exit $failure
        fi
    fi
done

echo "[SUCCESS] All sanity checks passed"
echo ""
echo "Validation complete!"
exit $success

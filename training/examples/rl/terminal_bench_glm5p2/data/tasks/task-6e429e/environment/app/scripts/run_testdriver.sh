#!/bin/bash
# Run the BVT 1:1 validation tests

configDir=config
if [ ! -e "$configDir" ]; then
    echo "[ERROR] Missing $configDir directory"
    exit 1
fi

outputDir=validation
templatesDir=$outputDir/templates
rm -rf $outputDir
mkdir -p $templatesDir

export LD_LIBRARY_PATH=$(pwd)/lib

# Check for hard-coded config directory
echo -n "Checking for hard-coded config directory... "
tempConfigDir=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1)
chmod 775 $configDir 2>/dev/null
mv $configDir $tempConfigDir
chmod 550 $tempConfigDir 2>/dev/null

bin/validate11 createTemplate -x enroll -c $tempConfigDir -o $outputDir -h check -i input/short_enroll.txt -j $templatesDir 2>/dev/null
retCode=$?

chmod 775 $tempConfigDir 2>/dev/null
mv $tempConfigDir $configDir
chmod 550 $configDir 2>/dev/null

if [[ $retCode != 0 ]]; then
    echo "[SUCCESS]"
else
    echo "[ERROR] Hard-coded config directory detected in your software."
    exit 1
fi

rm -rf $outputDir
mkdir -p $templatesDir

# Create enrollment templates
echo -n "Creating enrollment templates... "
bin/validate11 createTemplate -x enroll -c $configDir -o $outputDir -h enroll -i input/enroll.txt -j $templatesDir
retEnroll=$?
if [[ $retEnroll != 0 ]]; then
    echo "[ERROR] Enrollment template creation failed"
    exit 1
fi
echo "[SUCCESS]"

# Create verification templates
echo -n "Creating verification templates... "
bin/validate11 createTemplate -x verif -c $configDir -o $outputDir -h verif -i input/verif.txt -j $templatesDir
retVerif=$?
if [[ $retVerif != 0 ]]; then
    echo "[ERROR] Verification template creation failed"
    exit 1
fi
echo "[SUCCESS]"

# Match templates
echo -n "Matching templates... "
bin/validate11 match -c $configDir -o $outputDir -h match -i input/match.txt -j $templatesDir
retMatch=$?
if [[ $retMatch != 0 ]]; then
    echo "[ERROR] Template matching failed"
    exit 1
fi
echo "[SUCCESS]"

exit 0

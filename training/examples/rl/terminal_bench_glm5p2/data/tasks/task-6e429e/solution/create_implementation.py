#!/usr/bin/env python3
"""
Solution: Design and build a conformant BVT 1:1 vendor library from scratch,
and fix harness bugs that prevent a correct implementation from passing.

Steps:
1. Fix bvt_structs.h version mismatch (header says 2, driver requires 3)
2. Create impl/CMakeLists.txt with correct library naming convention
3. Create impl/vendor_impl.cpp implementing all Interface pure virtuals
4. Fix run_validate.sh score uniqueness awk column ($4 -> $3)
5. Fix run_testdriver.sh inverted config-directory check
"""

import os

# --- Step 1: Fix bvt_structs.h version mismatch ---
# The test driver (validate11.cpp) requires structs version 3.1 via:
#   uint16_t reqStructsMajor{3}, reqStructsMinor{1};
# But the header currently declares version 2.1 via:
#   uint16_t BVT_STRUCTS_MAJOR_VERSION{2};
# The extern/define guard means the vendor library's definition is used at link time.
path = '/app/include/bvt_structs.h'
with open(path) as f:
    content = f.read()
content = content.replace(
    'BVT_STRUCTS_MAJOR_VERSION{2}',
    'BVT_STRUCTS_MAJOR_VERSION{3}'
)
with open(path, 'w') as f:
    f.write(content)
print("Step 1: Fixed bvt_structs.h version 2->3")

# --- Step 2: Create impl/CMakeLists.txt ---
# Discovered from scripts/compile_and_link.sh that library must match the glob
# pattern libbvt_11_*_???.so (3-digit sequence number).
# The include path must point to ../include to find bvt_structs.h and bvt11.h.
cmake_content = r"""cmake_minimum_required(VERSION 3.11)
project(vendor_impl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

add_library(bvt_11_vendor_001 SHARED vendor_impl.cpp)
"""
with open('/app/impl/CMakeLists.txt', 'w') as f:
    f.write(cmake_content)
print("Step 2: Created impl/CMakeLists.txt")

# --- Step 3: Create vendor_impl.cpp ---
# Design decisions:
#   - Template creation: extract raw pixel bytes from Image::data shared_ptr.
#     The test driver creates 4x4x8bit synthetic images (16 bytes each) with
#     seed-dependent content via createSyntheticImage(). Copying all pixel data
#     into the template ensures non-empty templates and deterministic content.
#   - Scoring: compute dot product of template byte vectors. This is always
#     non-negative (uint8 * uint8 >= 0) and produces unique scores across
#     different template pairs because different seeds produce different pixel
#     data, yielding distinct dot products.
impl_content = r'''/*
 * Vendor implementation of the BVT 1:1 interface.
 * Implements biometric template creation and 1:1 matching.
 */

#include <algorithm>
#include <cstring>
#include <cstdlib>

#include "bvt_structs.h"
#include "bvt11.h"

using namespace BVT;
using namespace BVT_11;

class VendorImpl : public BVT_11::Interface {
public:
    VendorImpl() {}
    ~VendorImpl() override {}

    ReturnStatus
    initialize(const std::string &configDir) override
    {
        this->configDir = configDir;
        return ReturnStatus(ReturnCode::Success);
    }

    ReturnStatus
    createTemplate(
        const std::vector<Image> &faces,
        TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<EyePair> &eyeCoordinates) override
    {
        templ.clear();
        for (const auto &face : faces) {
            if (face.data && face.size() > 0) {
                uint8_t *rawData = face.data.get();
                size_t sz = face.size();
                for (size_t i = 0; i < sz; i++) {
                    templ.push_back(rawData[i]);
                }
            }
            eyeCoordinates.push_back(EyePair(true, true, 0, 0, 1, 1));
        }
        while (templ.size() < 16) {
            templ.push_back(0);
        }
        return ReturnStatus(ReturnCode::Success);
    }

    ReturnStatus
    createTemplate(
        const Image &image,
        TemplateRole role,
        std::vector<std::vector<uint8_t>> &templs,
        std::vector<EyePair> &eyeCoordinates) override
    {
        std::vector<uint8_t> templ;
        if (image.data && image.size() > 0) {
            uint8_t *rawData = image.data.get();
            size_t sz = image.size();
            for (size_t i = 0; i < sz; i++) {
                templ.push_back(rawData[i]);
            }
        }
        while (templ.size() < 16) {
            templ.push_back(0);
        }
        templs.push_back(templ);
        eyeCoordinates.push_back(EyePair(true, true, 0, 0, 1, 1));
        return ReturnStatus(ReturnCode::Success);
    }

    ReturnStatus
    matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &score) override
    {
        double dotProduct = 0.0;
        size_t len = std::min(verifTemplate.size(), enrollTemplate.size());
        for (size_t i = 0; i < len; i++) {
            dotProduct += (double)verifTemplate[i] * (double)enrollTemplate[i];
        }
        score = dotProduct;
        return ReturnStatus(ReturnCode::Success);
    }

private:
    std::string configDir;
};

std::shared_ptr<Interface>
Interface::getImplementation()
{
    return std::make_shared<VendorImpl>();
}
'''
with open('/app/impl/vendor_impl.cpp', 'w') as f:
    f.write(impl_content)
print("Step 3: Created vendor_impl.cpp with template extraction and dot-product scoring")

# --- Step 4: Fix run_validate.sh score uniqueness check ---
# Bug: The awk command extracts column $4 (returnCode) instead of $3 (simScore)
# for the uniqueness check. The match.log format is:
#   enrollTemplate verifTemplate simScore returnCode
# Column $3 is simScore, $4 is returnCode. Since all return codes are 0,
# extracting $4 yields only one unique value, always failing the 50% threshold.
path = '/app/run_validate.sh'
with open(path) as f:
    content = f.read()
content = content.replace(
    "awk '{ print $4 }' | sort -u",
    "awk '{ print $3 }' | sort -u"
)
with open(path, 'w') as f:
    f.write(content)
print("Step 4: Fixed run_validate.sh score uniqueness column $4->$3")

# --- Step 5: Fix run_testdriver.sh config directory check ---
# Bug: Inverted conditional logic. The test renames config/ to a random name
# and passes the new name via -c. A correct implementation uses -c (returns 0);
# a hardcoded one fails (returns non-zero). The script currently treats
# retCode!=0 as SUCCESS (no hardcoding detected), which is backwards.
# Fix: retCode==0 means the impl correctly used the -c argument.
path = '/app/scripts/run_testdriver.sh'
with open(path) as f:
    content = f.read()
content = content.replace('$retCode != 0', '$retCode == 0')
with open(path, 'w') as f:
    f.write(content)
print("Step 5: Fixed run_testdriver.sh config-dir check != -> ==")

print("\nAll steps complete. Ready to run validation pipeline.")

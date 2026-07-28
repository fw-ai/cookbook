#!/usr/bin/env python3
"""
Fix all issues in the BVT 1:1 validation pipeline.

Issues:
1. bvt_structs.h: wrong major version (2 instead of 3)
2. vendor_impl.cpp: missing single-image createTemplate override (abstract class)
3. vendor_impl.cpp: multi-face createTemplate produces empty templates
4. vendor_impl.cpp: matchTemplates returns constant 0.0 (fails uniqueness)
5. impl/CMakeLists.txt: library name has 2-digit sequence (need 3-digit)
6. impl/CMakeLists.txt: wrong include path
7. run_validate.sh: uniqueness check extracts wrong column ($4 instead of $3)
8. scripts/run_testdriver.sh: inverted config directory check
"""

import os

# --- Fix 1: bvt_structs.h version ---
path = '/app/include/bvt_structs.h'
with open(path) as f:
    content = f.read()
content = content.replace(
    'BVT_STRUCTS_MAJOR_VERSION{2}',
    'BVT_STRUCTS_MAJOR_VERSION{3}'
)
with open(path, 'w') as f:
    f.write(content)
print("Fixed: bvt_structs.h version 2->3")

# --- Fix 2-4: Complete vendor_impl.cpp ---
# Design decisions:
#   - Template creation: extract raw pixel data from Image structs
#     so templates capture actual image content (non-empty, deterministic)
#   - Scoring: use dot product of template byte vectors as similarity measure
#     (always non-negative since bytes are unsigned, unique across different pairs)
path = '/app/impl/vendor_impl.cpp'
fixed_impl = '''/*
 * Vendor implementation of the BVT 1:1 interface.
 * Produces feature templates from image data for validation purposes.
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
with open(path, 'w') as f:
    f.write(fixed_impl)
print("Fixed: vendor_impl.cpp - designed template creation and scoring algorithms")

# --- Fix 5-6: impl/CMakeLists.txt ---
path = '/app/impl/CMakeLists.txt'
with open(path) as f:
    content = f.read()
# Fix 5: Library name (2-digit -> 3-digit)
content = content.replace('bvt_11_vendor_01 ', 'bvt_11_vendor_001 ')
# Fix 6: Include path
content = content.replace(
    '${CMAKE_CURRENT_SOURCE_DIR}/include',
    '${CMAKE_CURRENT_SOURCE_DIR}/../include'
)
with open(path, 'w') as f:
    f.write(content)
print("Fixed: impl/CMakeLists.txt - library name and include path")

# --- Fix 7: run_validate.sh score column ---
path = '/app/run_validate.sh'
with open(path) as f:
    content = f.read()
content = content.replace(
    "awk '{ print $4 }' | sort -u",
    "awk '{ print $3 }' | sort -u"
)
with open(path, 'w') as f:
    f.write(content)
print("Fixed: run_validate.sh - score uniqueness column $4->$3")

# --- Fix 8: Inverted config dir check ---
path = '/app/scripts/run_testdriver.sh'
with open(path) as f:
    content = f.read()
content = content.replace('$retCode != 0', '$retCode == 0')
with open(path, 'w') as f:
    f.write(content)
print("Fixed: run_testdriver.sh - config dir check != -> ==")

print("\nAll 8 issues fixed.")

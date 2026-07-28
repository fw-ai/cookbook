/*
 * Biometric Validation Test (BVT) - Common Structures
 * Inspired by NIST FRVT evaluation framework
 */

#ifndef BVT_STRUCTS_H_
#define BVT_STRUCTS_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace BVT {

typedef struct Image {
    uint16_t width;
    uint16_t height;
    uint8_t depth;
    std::shared_ptr<uint8_t> data;

    Image() :
        width{0},
        height{0},
        depth{8}
        {}

    size_t
    size() const { return (width * height * (depth / 8)); }
} Image;

enum class TemplateRole {
    Enrollment_11 = 0,
    Verification_11 = 1
};

enum class ReturnCode {
    Success = 0,
    UnknownError = 1,
    ConfigError = 2,
    RefuseInput = 3,
    ExtractError = 4,
    ParseError = 5,
    TemplateCreationError = 6,
    VerifTemplateError = 7,
    FaceDetectionError = 8,
    NumDataError = 9,
    TemplateFormatError = 10,
    EnrollDirError = 11,
    InputLocationError = 12,
    MemoryError = 13,
    MatchError = 14,
    NotImplemented = 16,
    VendorError = 17
};

inline std::ostream&
operator<<(
    std::ostream &s,
    const ReturnCode &rc)
{
    switch (rc) {
    case ReturnCode::Success:
        return (s << "Success");
    case ReturnCode::UnknownError:
        return (s << "Unknown Error");
    case ReturnCode::ConfigError:
        return (s << "Config Error");
    case ReturnCode::NotImplemented:
        return (s << "Not Implemented");
    default:
        return (s << "Error");
    }
}

typedef struct ReturnStatus {
    ReturnCode code;
    std::string info;

    ReturnStatus() :
        code{ReturnCode::UnknownError},
        info{""}
        {}

    ReturnStatus(
        const ReturnCode code,
        const std::string &info = ""
        ) :
        code{code},
        info{info}
        {}
} ReturnStatus;

typedef struct EyePair
{
    bool isLeftAssigned;
    bool isRightAssigned;
    uint16_t xleft;
    uint16_t yleft;
    uint16_t xright;
    uint16_t yright;

    EyePair() :
        isLeftAssigned{false},
        isRightAssigned{false},
        xleft{0},
        yleft{0},
        xright{0},
        yright{0}
        {}

    EyePair(
        bool isLeftAssigned,
        bool isRightAssigned,
        uint16_t xleft,
        uint16_t yleft,
        uint16_t xright,
        uint16_t yright
        ) :
        isLeftAssigned{isLeftAssigned},
        isRightAssigned{isRightAssigned},
        xleft{xleft},
        yleft{yleft},
        xright{xright},
        yright{yright}
        {}
} EyePair;

#ifdef NIST_EXTERN_BVT_STRUCTS_VERSION
extern uint16_t BVT_STRUCTS_MAJOR_VERSION;
extern uint16_t BVT_STRUCTS_MINOR_VERSION;
#else
uint16_t BVT_STRUCTS_MAJOR_VERSION{2};
uint16_t BVT_STRUCTS_MINOR_VERSION{1};
#endif

}

#endif /* BVT_STRUCTS_H_ */

/*
 * Biometric Validation Test (BVT) - 1:1 Matching Interface
 * Inspired by NIST FRVT evaluation framework
 */

#ifndef BVT11_H_
#define BVT11_H_

#include <cstdint>
#include <string>
#include <vector>

#include "bvt_structs.h"

namespace BVT_11 {

class Interface {
public:
    virtual ~Interface() {}

    /**
     * @brief Initialize the implementation. Called once before any
     * template creation or matching. The configDir is assigned by the
     * test harness and must not be hardcoded.
     */
    virtual BVT::ReturnStatus
    initialize(const std::string &configDir) = 0;

    /**
     * @brief Generate a template from one or more biometric samples
     * of exactly one subject. Must always produce a template (even on
     * failure) that can be passed to matchTemplates without error.
     */
    virtual BVT::ReturnStatus
    createTemplate(
        const std::vector<BVT::Image> &faces,
        BVT::TemplateRole role,
        std::vector<uint8_t> &templ,
        std::vector<BVT::EyePair> &eyeCoordinates) = 0;

    /**
     * @brief Generate one or more templates from a single image that
     * may contain multiple subjects. The output vectors must have
     * matching sizes.
     */
    virtual BVT::ReturnStatus
    createTemplate(
        const BVT::Image &image,
        BVT::TemplateRole role,
        std::vector<std::vector<uint8_t>> &templs,
        std::vector<BVT::EyePair> &eyeCoordinates) = 0;

    /**
     * @brief Compare two templates and produce a similarity score.
     * Score must be on range [0, DBL_MAX] (higher = more similar).
     * Return VerifTemplateError with score -1 if either template
     * is the result of a failed extraction.
     */
    virtual BVT::ReturnStatus
    matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &score) = 0;

    /**
     * @brief Factory method returning a managed pointer to the
     * implementation object.
     */
    static std::shared_ptr<Interface>
    getImplementation();
};

#ifdef NIST_EXTERN_API_VERSION
extern uint16_t API_MAJOR_VERSION;
extern uint16_t API_MINOR_VERSION;
#else
uint16_t API_MAJOR_VERSION{6};
uint16_t API_MINOR_VERSION{0};
#endif

}

#endif /* BVT11_H_ */

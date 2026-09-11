//
//  OrtApi.hpp
//  CVisionAX
//
//  WHAT: The ONE place that includes ONNX Runtime. Api table, shared env, error mapping.
//  IN:   Classifier.cpp, visionax.cpp
//  OUT:  <onnxruntime/onnxruntime_c_api.h>
//  PIN:  The include goes THROUGH THE FRAMEWORK. The pod archive ships an xcframework
//        with no module.modulemap and SwiftPM discards its top-level Headers/, so
//        <onnxruntime/...> resolving at all depends on the -F SwiftPM already passes
//        for the binary target — the same mechanism <opencv2/core.hpp> rides on.
//        The OrtEnv is a function-local static and is NEVER released: it must outlive
//        every session, and sessions are owned by Swift objects with no defined
//        teardown order. Process exit reclaims it.
//

#pragma once

#include <stdexcept>
#include <string>

#include <onnxruntime/onnxruntime_c_api.h>

namespace visionax {

/// The version-pinned API table. ORT_API_VERSION comes from the header we compiled
/// against, so a runtime older than that fails here rather than misbehaving later.
inline const OrtApi &ort() {
    static const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (api == nullptr) {
        throw std::runtime_error("ONNX Runtime is older than the headers this was built against");
    }
    return *api;
}

/// One process-wide environment. Creating one per session leaks thread pools.
inline OrtEnv *ortEnv() {
    static OrtEnv *env = [] {
        OrtEnv *created = nullptr;
        OrtStatus *status = ort().CreateEnv(ORT_LOGGING_LEVEL_WARNING, "visionax", &created);
        if (status != nullptr) {
            const std::string message = ort().GetErrorMessage(status);
            ort().ReleaseStatus(status);
            throw std::runtime_error("could not create the ONNX Runtime environment: " + message);
        }
        return created;
    }();
    return env;
}

/// Releases a status without raising — for cleanup paths, where the original failure
/// is the one worth reporting and a second one must not mask it.
inline void ortDiscard(OrtStatus *status) {
    if (status != nullptr) ort().ReleaseStatus(status);
}

/// Turns an OrtStatus into an exception and releases it. Every ORT call goes through
/// this — a leaked status is a leaked string, and an ignored one is a silent wrong answer.
inline void ortCheck(OrtStatus *status) {
    if (status == nullptr) return;
    const std::string message = ort().GetErrorMessage(status);
    ort().ReleaseStatus(status);
    throw std::runtime_error(message);
}

}  // namespace visionax

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "ShapedWeights.hpp"
#include "bfloat16.hpp"
#include "half.h"
#include <NvInfer.h>
#include <typeindex>
#include <unordered_map>

// Subset of helper functions that deal exclusively with weights to be shared across IParser and IParserRefitter classes.
// Define weightLog Macros here to ensure that an ImporterCtx class is not needed to log.

namespace onnx2trt
{

// Return the name of an ONNX data enum.
char const* getDtypeName(int32_t onnxDtype);

// Return the size in bits of an ONNX data type.
int32_t getDtypeSizeBits(int32_t onnxDtype);

// Return the size in bytes of an tensor/weights object, handle sub-byte padding.
size_t getTensorOrWeightsSizeBytes(int64_t count, int32_t onnxDtype);

// Find the corresponding ONNX data type of a built-in data type.
template <typename T>
ShapedWeights::DataType getShapedWeightsDataType()
{
    static std::unordered_map<std::type_index, ::onnx::TensorProto::DataType> const tMap({
        {std::type_index(typeid(bool)), ::onnx::TensorProto::BOOL},
        {std::type_index(typeid(int8_t)), ::onnx::TensorProto::INT8},
        {std::type_index(typeid(uint8_t)), ::onnx::TensorProto::UINT8},
        {std::type_index(typeid(int16_t)), ::onnx::TensorProto::INT16},
        {std::type_index(typeid(uint16_t)), ::onnx::TensorProto::UINT16},
        {std::type_index(typeid(int32_t)), ::onnx::TensorProto::INT32},
        {std::type_index(typeid(uint32_t)), ::onnx::TensorProto::UINT32},
        {std::type_index(typeid(int64_t)), ::onnx::TensorProto::INT64},
        {std::type_index(typeid(uint64_t)), ::onnx::TensorProto::UINT64},
        {std::type_index(typeid(float)), ::onnx::TensorProto::FLOAT},
        {std::type_index(typeid(double)), ::onnx::TensorProto::DOUBLE},
        {std::type_index(typeid(half_float::half)), ::onnx::TensorProto::FLOAT16},
        {std::type_index(typeid(BFloat16)), ::onnx::TensorProto::BFLOAT16},
        // TRT-22989: Add fp8 and int4 support
    });

    if (tMap.find(std::type_index(typeid(T))) != tMap.end())
    {
        return tMap.at(std::type_index(typeid(T)));
    }
    return ::onnx::TensorProto::UNDEFINED;
}

// Return the volume of a Dims object
int64_t volume(nvinfer1::Dims const& dims);

// Normalize the slashes in a string representing a filepath.
std::string normalizePath(std::string const& path);

// Generate a unique name for a given weight or tensor name (passed as the |basename|)
std::string const& generateUniqueName(
    std::set<std::string>& namesSet, int64_t& suffixCounter, std::string const& basename);

} // namespace onnx2trt

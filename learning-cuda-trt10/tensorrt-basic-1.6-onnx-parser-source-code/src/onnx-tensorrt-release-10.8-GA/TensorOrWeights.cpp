/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TensorOrWeights.hpp"
#include <cassert>

namespace onnx2trt
{

std::string TensorOrWeights::getType() const
{
    if (is_tensor())
    {
        switch (tensor().getType())
        {
        case nvinfer1::DataType::kFLOAT: return "FLOAT";
        case nvinfer1::DataType::kHALF: return "HALF";
        case nvinfer1::DataType::kBF16: return "BF16";
        case nvinfer1::DataType::kINT8: return "INT8";
        case nvinfer1::DataType::kUINT8: return "UINT8";
        case nvinfer1::DataType::kINT32: return "INT32";
        case nvinfer1::DataType::kINT64: return "INT64";
        case nvinfer1::DataType::kBOOL: return "BOOL";
        case nvinfer1::DataType::kFP8: return "FP8";
        case nvinfer1::DataType::kINT4: return "INT4";
        case nvinfer1::DataType::kFP4: return "FP4";
        }
    }
    else
    {
        switch (weights().type)
        {
        // Demote double to float.
        case ::onnx::TensorProto::DOUBLE: return "FLOAT";
        case ::onnx::TensorProto::FLOAT: return "FLOAT";
        case ::onnx::TensorProto::INT8: return "INT8";
        case ::onnx::TensorProto::UINT8: return "UINT8";
        case ::onnx::TensorProto::FLOAT16: return "HALF";
        case ::onnx::TensorProto::BFLOAT16: return "BF16";
        case ::onnx::TensorProto::BOOL: return "BOOL";
        case ::onnx::TensorProto::INT32: return "INT32";
        case ::onnx::TensorProto::INT64: return "INT64";
        case ::onnx::TensorProto::FLOAT8E4M3FN: return "FP8";
        case ::onnx::TensorProto::INT4: return "INT4";
        case ::onnx::TensorProto::FLOAT4E2M1: return "FP4";
        }
    }
    return "UNKNOWN TYPE";
}

nvinfer1::DataType TensorOrWeights::convertONNXDataType(ShapedWeights::DataType datatype) const
{
    switch (datatype)
    {
        case ::onnx::TensorProto::DOUBLE: return nvinfer1::DataType::kFLOAT;
        case ::onnx::TensorProto::FLOAT: return nvinfer1::DataType::kFLOAT;
        case ::onnx::TensorProto::INT8: return nvinfer1::DataType::kINT8;
        case ::onnx::TensorProto::UINT8: return nvinfer1::DataType::kUINT8;
        case ::onnx::TensorProto::FLOAT16: return nvinfer1::DataType::kHALF;
        case ::onnx::TensorProto::BFLOAT16: return nvinfer1::DataType::kBF16;
        case ::onnx::TensorProto::BOOL: return nvinfer1::DataType::kBOOL;
        case ::onnx::TensorProto::INT32: return nvinfer1::DataType::kINT32;
        case ::onnx::TensorProto::INT64: return nvinfer1::DataType::kINT64;
        case ::onnx::TensorProto::FLOAT8E4M3FN: return nvinfer1::DataType::kFP8;
        case ::onnx::TensorProto::INT4: return nvinfer1::DataType::kINT4;
        case ::onnx::TensorProto::FLOAT4E2M1: return nvinfer1::DataType::kFP4;
        }
        assert(false && "Unknown datatype");
        return nvinfer1::DataType::kFLOAT;
}

ShapedWeights::DataType TensorOrWeights::convertTRTDataType(nvinfer1::DataType datatype) const
{
    switch (datatype)
    {
        case nvinfer1::DataType::kFLOAT: return ::onnx::TensorProto::FLOAT;
        case nvinfer1::DataType::kINT8: return ::onnx::TensorProto::INT8;
        case nvinfer1::DataType::kUINT8: return ::onnx::TensorProto::UINT8;
        case nvinfer1::DataType::kHALF: return ::onnx::TensorProto::FLOAT16;
        case nvinfer1::DataType::kBF16: return ::onnx::TensorProto::BFLOAT16;
        case nvinfer1::DataType::kBOOL: return ::onnx::TensorProto::BOOL;
        case nvinfer1::DataType::kINT32: return ::onnx::TensorProto::INT32;
        case nvinfer1::DataType::kINT64: return ::onnx::TensorProto::INT64;
        case nvinfer1::DataType::kFP8: return ::onnx::TensorProto::FLOAT8E4M3FN;
        case nvinfer1::DataType::kINT4: return ::onnx::TensorProto::INT4;
        case nvinfer1::DataType::kFP4: return ::onnx::TensorProto::FLOAT4E2M1;
        }
        assert(false && "Unknown datatype");
        return ::onnx::TensorProto::FLOAT;
}

} // namespace onnx2trt

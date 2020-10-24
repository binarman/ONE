/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels/Gather.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

#include <stdexcept>
#include <thread>

namespace luci_interpreter
{
namespace kernels
{

Gather::Gather(const Tensor *params, const Tensor *indices, Tensor *output, int32_t axis)
    : Kernel({params, indices}, {output}), _axis(axis)
{
}

void Gather::configure()
{
  const Shape &p_shape = params()->shape();
  const Shape &i_shape = indices()->shape();
  int p_rank = p_shape.num_dims();
  int i_rank = i_shape.num_dims();

  int output_rank = p_rank + i_rank - 1;
  Shape output_shape(output_rank);
  for (int i = 0; i < _axis; ++i)
    output_shape.dim(i) = p_shape.dim(i);
  for (int i = 0; i < i_rank; ++i)
    output_shape.dim(_axis + i) = i_shape.dim(i);
  for (int i = _axis + 1; i < p_rank; ++i)
    output_shape.dim(i + i_rank - 1) = p_shape.dim(i);

  output()->resize(output_shape);
}

void Gather::execute() const
{

  if (indices()->element_type() == DataType::S32) {
    const int32_t *i_data = indices()->data<int32_t>();
    switch (params()->element_type()) {
      case DataType::FLOAT32:
        eval<float, int32_t>();
        break;
      case DataType::U8:
        eval<uint8_t, int32_t>();
        break;
      case DataType::S8:
        eval<int8_t, int32_t>();
        break;
      case DataType::S32:
        eval<int32_t, int32_t>();
        break;
      case DataType::S64:
        eval<int64_t, int32_t>();
        break;
      case DataType::BOOL:
        eval<bool, int32_t>();
        break;
      default:
        throw std::runtime_error("unsupported type by gather");
    }
  }
  if (indices()->element_type() == DataType::S64) {
    switch (params()->element_type()) {
      case DataType::FLOAT32:
        eval<float, int64_t>();
        break;
      case DataType::U8:
        eval<uint8_t, int64_t>();
        break;
      case DataType::S8:
        eval<int8_t, int64_t>();
        break;
      case DataType::S32:
        eval<int32_t, int64_t>();
        break;
      case DataType::S64:
        eval<int64_t, int64_t>();
        break;
      case DataType::BOOL:
        eval<bool, int64_t>();
        break;
      default:
        throw std::runtime_error("unsupported type by gather");
    }
  }
}

template <typename T, typename IndexT>
void Gather::eval() const
{
  tflite::GatherParams gather_params{static_cast<int16_t >(_axis)};

  tflite::RuntimeShape p_shape = getTensorShape(params());
  tflite::RuntimeShape i_shape = getTensorShape(indices());
  tflite::RuntimeShape o_shape = getTensorShape(output());

  const T *p_data = params()->data<T>();
  const IndexT *i_data = indices()->data<IndexT>();
  T *o_data = output()->data<T>();

  tflite::optimized_ops::Gather(gather_params, p_shape, p_data, i_shape, i_data, o_shape, o_data);
}

} // namespace kernels
} // namespace luci_interpreter

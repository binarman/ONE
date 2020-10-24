/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Range.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace luci_interpreter
{

namespace kernels
{

Range::Range(const Tensor *start,const Tensor *limit, const Tensor *delta, Tensor *output) : Kernel({start, limit, delta}, {output}) {}

template <typename T>
static inline Shape outputShape(const Tensor *start, const Tensor *limit, const Tensor *delta)
{
  T start_value = *start->data<T>();
  T limit_value = *limit->data<T>();
  T delta_value = *delta->data<T>();
  int32_t size =
      (std::is_integral<T>::value
       ? ((std::abs(limit_value - start_value) + std::abs(delta_value) - 1) / std::abs(delta_value))
       : std::ceil(std::abs((limit_value - start_value) / delta_value)));

  return Shape{size};
}

template <typename T>
static inline void eval(const Tensor* start, const Tensor* delta, Tensor* output) {
  const T start_value = *start->data<T>();
  const T delta_value = *delta->data<T>();
  T* output_data = output->data<T>();
  const int num_elements = output->shape().num_elements();
  T value = start_value;
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = value;
    value += delta_value;
  }
}

void Range::configure()
{
  if (start()->element_type() != limit()->element_type() || start()->element_type() != delta()->element_type())
    throw std::runtime_error("mismatch of input types");
  if (start()->shape().num_dims() != 0 || limit()->shape().num_dims() != 0 || delta()->shape().num_dims() != 0)
    throw std::runtime_error("non scalar input tensor shapes");
  switch (output()->element_type())
  {
    case DataType::S32:
      output()->resize(outputShape<int32_t>(start(), limit(), delta()));
      break;
    case DataType::FLOAT32:
      output()->resize(outputShape<float>(start(), limit(), delta()));
      break;
    default:
      throw std::runtime_error("unsupported element type");
  }
}

void Range::execute() const
{
  switch (output()->element_type())
  {
    case DataType::S32:
      eval<int32_t>(start(), delta(), output());
      break;
    case DataType::FLOAT32:
      eval<float>(start(), delta(), output());
      break;
    default:
      throw std::runtime_error("unsupported element type");
  }
}

} // namespace kernels
} // namespace luci_interpreter

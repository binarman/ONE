/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/ExpandDims.h"

#include <cassert>
#include <cstring>

namespace luci_interpreter
{

namespace kernels
{

ExpandDims::ExpandDims(const Tensor *input, const Tensor *axis, Tensor *output)
    : Kernel({input, axis}, {output})
{
}

void ExpandDims::configure()
{
  const Shape &input_shape = input()->shape();
  int axis_value;
  switch (axis()->element_type())
  {
    case DataType::S32:
      axis_value = *axis()->data<int32_t >();
      break;
    case DataType::S64:
      axis_value = *axis()->data<int64_t >();
      break;
    default:
      throw std::runtime_error("unsupported axis data type");
  }
  if (axis_value < 0)
    axis_value += input_shape.num_dims() + 1;
  if (axis_value < 0 || axis_value > input_shape.num_dims())
    throw std::runtime_error("axis is out of [-(D+1), D] range");
  Shape output_shape(input_shape.num_dims() + 1);
  for (int i = 0; i < axis_value; ++i)
    output_shape.dim(i) = input_shape.dim(i);
  output_shape.dim(axis_value) = 1;
  for (int i = axis_value; i < input_shape.num_dims(); ++i)
    output_shape.dim(i+1) = input_shape.dim(i);
  output()->resize(output_shape);
}

void ExpandDims::execute() const
{
  const auto *input_data = input()->data<void>();
  auto *output_data = output()->data<void>();

  const size_t element_size = getDataTypeSize(input()->element_type());
  const int32_t num_elements = input()->shape().num_elements();
  std::memcpy(output_data, input_data, num_elements * element_size);
}

} // namespace kernels
} // namespace luci_interpreter

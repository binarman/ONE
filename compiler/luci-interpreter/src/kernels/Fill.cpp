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

#include "kernels/Fill.h"

#include <stdexcept>
#include <algorithm>

namespace luci_interpreter
{

namespace kernels
{

Fill::Fill(const Tensor *dims, const Tensor *value, Tensor *output) : Kernel({dims, value}, {output}) {}

template <typename T>
Shape shapeFromData(int rank, const T *data)
{
  Shape s(rank);
  for (int i = 0; i < rank; ++i)
    s.dim(i) = static_cast<int>(data[i]);
  return s;
}

void Fill::configure()
{
  if (dims()->shape().num_dims() != 1)
    throw std::runtime_error("unexpected number of dimensions in dims tensor");
  if (value()->shape().num_dims() != 0)
    throw std::runtime_error("non scalar value tensor");
  if (dims()->element_type() != DataType::S32 && dims()->element_type() != DataType::S64)
    throw std::runtime_error("unsupported data type of dims tensor");
  if (value()->element_type() != output()->element_type())
    throw std::runtime_error("mismatch of value and output tensor element types");
  int output_rank = dims()->shape().dim(0);
  if (dims()->element_type() == DataType::S32)
  {
    output()->resize(shapeFromData(output_rank, dims()->data<int32_t >()));
  } else
  {
    output()->resize(shapeFromData(output_rank, dims()->data<int64_t >()));
  }
}

void Fill::execute() const
{
  size_t num_elements = output()->shape().num_elements();
  switch (value()->element_type())
  {
    case DataType::FLOAT32:
      std::fill_n(output()->data<float>(), num_elements, *value()->data<float>());
      break;
    case DataType::S32:
      std::fill_n(output()->data<int32_t >(), num_elements, *value()->data<int32_t >());
      break;
    case DataType::S64:
      std::fill_n(output()->data<int64_t >(), num_elements, *value()->data<int64_t >());
      break;
    case DataType::BOOL:
      std::fill_n(output()->data<bool>(), num_elements, *value()->data<bool>());
      break;
    default:
      throw std::runtime_error("unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter

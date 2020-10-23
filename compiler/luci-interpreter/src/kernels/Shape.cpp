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

#include "kernels/Shape.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

GetShape::GetShape(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void GetShape::configure()
{
  output()->resize({input()->shape().num_dims()});
}

template <typename T> static inline void writeShapeData(T *data, const Shape &s)
{
  for (int i = 0; i < s.num_dims(); ++i)
    data[i] = s.dim(i);
}

void GetShape::execute() const
{
  switch (output()->element_type())
  {
    case DataType::S32:
      writeShapeData(output()->data<int32_t>(), input()->shape());
      break;
    case DataType::S64:
      writeShapeData(output()->data<int64_t>(), input()->shape());
      break;
    default:
      throw std::runtime_error("unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter

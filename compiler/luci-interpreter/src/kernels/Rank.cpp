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

#include "kernels/Rank.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Rank::Rank(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Rank::configure()
{
  output()->resize({});
}

void Rank::execute() const
{
  if (output()->element_type() != DataType::S32)
    throw std::runtime_error("unsupported output type");
  *output()->data<int32_t >() = input()->shape().num_dims();
}

} // namespace kernels
} // namespace luci_interpreter

/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef NNCC_CODEGEN_KERNEL_BUILDER_H
#define NNCC_CODEGEN_KERNEL_BUILDER_H

#include "SubgraphContext.h"

namespace luci_codegen
{

class KernelBuilder
{
public:
  explicit KernelBuilder(SubgraphContext &subgraph);

  static bool is_supported(luci::CircleNode *node);

  void process();

private:
  SubgraphContext &_subgraph;
};

} // namespace luci_codegen

#endif // NNCC_CODEGEN_KERNEL_BUILDER_H
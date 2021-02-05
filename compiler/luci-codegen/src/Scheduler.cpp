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

#include "Scheduler.h"
//#include "loco/IR/Algorithm.h"

namespace luci_codegen
{

void Scheduler::process()
{
  for (auto &input: _subgraph.get_inputs())
  {
    luci::CircleNode *node = input.first;
    Halide::ImageParam input_param = input.second;
    Halide::Region estimates;
    uint32_t rank = node->rank();
    for (int i = 0; i < rank; ++i)
    {
      estimates.emplace_back(0, static_cast<int>(node->dim(rank - 1 - i).value()));
    }
    input_param.set_estimates(std::move(estimates));
  }
  std::vector<Halide::Func> halide_outputs;
  for (auto &output : _subgraph.get_outputs())
  {
    luci::CircleNode *node = output.first;
    Halide::Func output_func = output.second;
    halide_outputs.push_back(output_func);
    Halide::Region estimates;
    uint32_t rank = node->rank();
    for (int i = 0; i < rank; ++i)
    {
      estimates.emplace_back(0, static_cast<int>(node->dim(rank - 1 - i).value()));
    }
    output_func.set_estimates(std::move(estimates));
  }
  Halide::Pipeline pipeline(halide_outputs);
  Halide::MachineParams params(1, 32*1024, 40);
  // todo see GeneratorBase::build_module how to compile pipeline to module and use schedule
  Halide::AutoSchedulerResults schedule = pipeline.auto_schedule(Halide::get_host_target(), params);
}

} // namespace luci_codegen

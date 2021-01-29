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

#include "Codegen.h"
#include "CodegenKernelBuilder.h"
#include "SubgraphContext.h"
#include "Utilities.h"

#include "luci/IR/Nodes/CircleCustom.h"
#include "luci/IR/Nodes/CircleCustomOut.h"
#include "loco/IR/Algorithm.h"

#include "Halide.h"

#include "flatbuffers/flexbuffers.h"

#include <map>
#include <unordered_set>
#include <algorithm>

namespace
{

std::vector<uint8_t> create_custom_options(const std::string &name)
{
  flexbuffers::Builder fbb;
  fbb.Map([&]() {fbb.String("func_name", name);});
  fbb.Finish();
  return fbb.GetBuffer();
}

} // unnamed namespace

namespace luci_codegen
{

Codegen::Codegen(const Options &options) : _processed_graphs(0), _options(options) {}

Codegen::~Codegen() {}

bool Codegen::fits_constrains(luci::CircleNode *node) const
{
  if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    return const_node_size(node) <= _options.max_inline_buffer_threshold;
  return CodegenKernelBuilder::is_supported(node);
}

// todo move processed to class members
std::vector<luci::CircleNode *>
Codegen::gather_suitable_nodes(luci::CircleNode *node, std::unordered_set<luci::CircleNode *> &processed) const
{
  std::vector<luci::CircleNode *> subgraph_nodes;
  std::queue<luci::CircleNode *> queue;
  queue.push(node);
  processed.insert(node);
  while (!queue.empty())
  {
    luci::CircleNode *cur_node = queue.front();
    subgraph_nodes.push_back(cur_node);
    queue.pop();

    std::vector<luci::CircleNode *> adjacent;
    // gather adjacent nodes
    for (int i = 0; i < cur_node->arity(); ++i)
    {
      adjacent.push_back(static_cast<luci::CircleNode *>(cur_node->arg(i)));
    }
    auto succs = loco::succs(cur_node);
    for (auto succ: succs)
    {
      adjacent.push_back(static_cast<luci::CircleNode *>(succ));
    }
    // process adjacent nodes
    for (auto adj: adjacent)
    {
      if (processed.count(adj) || !fits_constrains(adj))
      {
        continue;
      }
      processed.insert(adj);
      queue.push(adj);
    }
  }
  return subgraph_nodes;
}

// check if we can compile found subgraph and remove redundant nodes
// Example of problematic subgraph:
// C - compilable node
// N - not compilable node
//
//   |
//   C
//  / \
// N  C
//  \ /
//   C
//   |
//
// this graph will be transformed into graph with cyclic dependency of generated node from itself
std::vector<luci::CircleNode *>
Codegen::filter_nodes(const std::vector<luci::CircleNode *> &nodes) const
{
  // TODO
  return nodes;
}

SubgraphContext *Codegen::create_subgraph(const std::vector<luci::CircleNode *> &nodes)
{
  std::string subgraph_name = "generated_subgraph_" + std::to_string(_processed_graphs);
  _compiled_subgraphs.emplace_back(subgraph_name, std::move(nodes));
  auto *subgraph = &_compiled_subgraphs.back();
  subgraph->finish_construction();
  return subgraph;
}

void Codegen::replace_subgraph_with_generated_node(SubgraphContext *subgraph) const
{
  auto &inputs = subgraph->get_inputs();
  const auto num_inputs = inputs.size();
  loco::Graph *graph = subgraph->get_graph();

  auto compiled_node = graph->nodes()->create<luci::CircleCustom>(num_inputs);
  compiled_node->custom_code("COMPILED_OP");

  auto options = create_custom_options(subgraph->get_name());
  compiled_node->custom_options(options);

  compiled_node->dtype(loco::DataType::FLOAT32);

  for (int i = 0; i < num_inputs; ++i)
  {
    compiled_node->inputs(i, subgraph->get_inputs()[i].first);
  }

  for (int i = 0; i < subgraph->get_outputs().size(); ++i)
  {
    auto output = subgraph->get_outputs()[i];
    auto custom_output = graph->nodes()->create<luci::CircleCustomOut>();
    custom_output->input(compiled_node);
    custom_output->index(i);
    custom_output->dtype(output.first->dtype());
    custom_output->shape_status(output.first->shape_status());

    // copy shape
    uint32_t rank = output.first->rank();
    custom_output->rank(rank);
    for (uint32_t i = 0; i < rank; ++i)
    {
      custom_output->dim(i) = output.first->dim(i);
    }

    loco::replace(output.first).with(custom_output);
  }
}

void Codegen::cleanup_graph(SubgraphContext *subgraph) const
{
  loco::Graph *graph = subgraph->get_graph();
  std::vector<loco::Node *> outputs;
  for (auto node: subgraph->get_outputs())
  {
    outputs.push_back(node.first);
  }
  auto ordered_nodes = loco::postorder_traversal(outputs);
  std::reverse(ordered_nodes.begin(), ordered_nodes.end());
  for (auto node: ordered_nodes)
  {
    if (subgraph->contains(static_cast<luci::CircleNode *>(node)))
      graph->nodes()->destroy(node);
  }
}

void Codegen::process_graph(loco::Graph &graph)
{
  std::unordered_set<luci::CircleNode *> processed;
  auto nodes = graph.nodes();

  // find and generate code
  for (int i = 0; i < nodes->size(); ++i)
  {
    auto node = static_cast<luci::CircleNode *>(nodes->at(i));

    // Check if we found node that belongs to subgraph we can compile
    if (processed.count(node) || !fits_constrains(node))
      continue;

    // Traverse graph to find all compilable adjacent nodes
    std::vector<luci::CircleNode *> suitable_nodes = gather_suitable_nodes(node, processed);

    std::vector<luci::CircleNode *> subgraph_nodes = filter_nodes(suitable_nodes);

    SubgraphContext *subgraph = create_subgraph(subgraph_nodes);

    // Create kernels for nodes
    CodegenKernelBuilder(*subgraph).process();

    // TODO add scheduler entity

    _processed_graphs++;
  }

  // replace circle graph with generated nodes
  for (SubgraphContext &subgraph: _compiled_subgraphs)
  {
    // Replace subgraph with custom operator
    replace_subgraph_with_generated_node(&subgraph);

    // Cleanup graph
    cleanup_graph(&subgraph);
  }
}

void Codegen::process_module(luci::Module &module)
{
  auto num_graphs = module.size();
  for (size_t i = 0; i < num_graphs; ++i)
    process_graph(*module.graph(i));
}

void Codegen::emit_code(std::string package_name)
{
  for (auto &subgraph: _compiled_subgraphs)
  {
    std::vector<Halide::Argument> arguments;
    for (auto input: subgraph.get_inputs())
    {
      arguments.push_back(input.second);
    }
    std::vector<Halide::Func> outputs;
    for (auto output: subgraph.get_outputs())
    {
      outputs.push_back(output.second);
    }
    Halide::Pipeline composite_output(outputs);
    composite_output.compile_to_lowered_stmt(subgraph.get_name() + ".html", arguments, Halide::StmtOutputFormat::HTML);
    composite_output.compile_to_object(subgraph.get_name() + ".o", arguments, subgraph.get_name());
  }
}

} // namespace luci_codegen

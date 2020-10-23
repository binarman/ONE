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

#include "kernels/While.h"
#include "kernels/Utils.h"
#include <iostream>

#include <cstring>

namespace luci_interpreter
{
namespace kernels
{

While::While(const std::vector<const Tensor *> &inputs, std::vector<Tensor *> outputs,
       RuntimeGraph *cond_graph, RuntimeGraph *body_graph)
    : Kernel(std::move(inputs), std::move(outputs)), _cond_graph(cond_graph),
      _body_graph(body_graph)
{
}

std::ostream &operator << (std::ostream &os, DataType dt)
{
  switch (dt)
  {
    case DataType::FLOAT32:
      os << "float";
      break;
    case DataType::S32:
      os << "int32";
      break;
    default:
      os << "other";
      break;
  }
  return os;
}

std::ostream &operator << (std::ostream &os, Shape s)
{
  os << "[";
  for (int i = 0; i < s.num_dims(); ++i)
    os << s.dim(i) << ", ";
  os << "]";
  return os;
}

template <typename T>
static inline void printTensors(const std::vector<T *> &v)
{
  for (int i = 0; i < v.size(); ++i)
  {
    const Tensor *t = v[i];
    std::cout << i << " " << t->name() << " " << t->element_type() << " " << t->shape() << "\n";
    if (t->shape().num_dims() == 0 && t->element_type() == DataType::S32)
      std::cout << *t->data<int32_t>() << "\n";
  }
}

void While::configure()
{
  LUCI_INTERPRETER_CHECK(_cond_graph->getOutputTensors().size() == 1);
  LUCI_INTERPRETER_CHECK(_cond_graph->getOutputTensors()[0]->element_type() == DataType::BOOL);
  LUCI_INTERPRETER_CHECK(_cond_graph->getOutputTensors()[0]->shape().num_elements() == 1);

  std::cout << "while op: " << getInputTensors().size() << " " << getOutputTensors().size() << "\n";
  printTensors(getInputTensors());
  std::cout << "\n";
  std::cout << "cond graph " << _cond_graph->getInputTensors().size() << " " << _cond_graph->getOutputTensors().size() << "\n";
  printTensors(_cond_graph->getInputTensors());
  std::cout << "\n";
  std::cout << "body graph " << _body_graph->getInputTensors().size() << " " << _body_graph->getOutputTensors().size() << "\n";
  printTensors(_body_graph->getInputTensors());
}

template <typename Tensor>
static inline void copyTensorsToGraph(const std::vector<Tensor *> &tensors, RuntimeGraph *g)
{
  const auto &graph_inputs = g->getInputTensors();
  for (size_t i = 0; i < tensors.size(); ++i)
  {
    LUCI_INTERPRETER_CHECK(graph_inputs[i]->element_type() == tensors[i]->element_type());
    graph_inputs[i]->resize(tensors[i]->shape());

    const int32_t num_elements = tensors[i]->shape().num_elements();
    const std::size_t element_size = getDataTypeSize(tensors[i]->element_type());
    std::memcpy(graph_inputs[i]->data<void>(), tensors[i]->template data<void>(),
        num_elements * element_size);
  }
}

static inline void copyTensorsNextIteration(RuntimeGraph *g)
{
  auto &graph_inputs = g->getInputTensors();
  const auto &graph_outputs = g->getOutputTensors();
  const std::size_t num_tensors = graph_inputs.size();
  // Need to store data in temporary place to avoid rewite of tensors which are both input and output of subgraph
  std::vector<std::vector<char>> tmp_storage(num_tensors);
  std::vector<Shape> output_shapes;

  for (size_t i = 0; i < num_tensors; ++i)
  {
    const Tensor *output = graph_outputs[i];
    const Shape &output_shape = output->shape();

    const int32_t num_elements = output_shape.num_elements();
    const std::size_t element_size = getDataTypeSize(output->element_type());
    const size_t data_size = num_elements * element_size;
    tmp_storage[i].resize(data_size);

    std::memcpy(tmp_storage[i].data(), output->data<void>(), data_size);
    output_shapes.push_back(output_shape);
  }

  for (size_t i = 0; i < num_tensors; ++i)
  {
    Tensor *input = graph_inputs[i];
    const Shape &output_shape = output_shapes[i];
    input->resize(output_shape);

    std::memcpy(input->data<void>(), tmp_storage[i].data(), tmp_storage[i].size());
  }
}

static inline void copyTensorsFromGraph(RuntimeGraph *g, const std::vector<Tensor *> &tensors)
{
  const auto &graph_outputs = g->getOutputTensors();
  for (size_t i = 0; i < graph_outputs.size(); ++i)
  {
    LUCI_INTERPRETER_CHECK(graph_outputs[i]->element_type() == tensors[i]->element_type());
    tensors[i]->resize(graph_outputs[i]->shape());

    const int32_t num_elements = graph_outputs[i]->shape().num_elements();
    const std::size_t element_size = getDataTypeSize(graph_outputs[i]->element_type());
    std::memcpy(tensors[i]->data<void>(), graph_outputs[i]->data<void>(),
        num_elements * element_size);
  }
}

void While::execute() const
{
  copyTensorsToGraph(getInputTensors(), _body_graph);

  int iter = 0;

  while (loopCondition())
  {
    std::cout << "  Iter: " << ++iter << "\n";
    std::cout << "INPUTS\n";
    std::cout << "\n";
    std::cout << "cond graph " << _cond_graph->getInputTensors().size() << "\n";
    printTensors(_cond_graph->getInputTensors());
    std::cout << "\n";
    std::cout << "body graph " << _body_graph->getInputTensors().size() << "\n";
    printTensors(_body_graph->getInputTensors());

    _body_graph->execute();

    std::cout << "\n";
    std::cout << "OUTPUTS\n";
    std::cout << "\n";
    std::cout << "cond graph " << " " << _cond_graph->getOutputTensors().size() << "\n";
    printTensors(_cond_graph->getOutputTensors());
    std::cout << "\n";
    std::cout << "body graph " << " " << _body_graph->getOutputTensors().size() << "\n";
    printTensors(_body_graph->getOutputTensors());
    std::cout << "\n";
    copyTensorsNextIteration(_body_graph);
  }
  copyTensorsFromGraph(_body_graph, getOutputTensors());
}

bool While::loopCondition() const
{
  copyTensorsToGraph(_body_graph->getInputTensors(), _cond_graph);
  _cond_graph->execute();
  return *_cond_graph->getOutputTensors()[0]->data<bool>();
}

} // namespace kernels
} // namespace luci_interpreter

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

#include <luci/Importer.h>
#include <luci_interpreter/Interpreter.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <random>

namespace
{

void fillInputWithOnes(char *data, size_t data_size)
{
  for (int i = 0; i < data_size/sizeof(float); ++i)
    reinterpret_cast<float*>(data)[i] = 1.f;
}

void printOutput(const char *data, size_t data_size)
{
  std::cout << "output: ";
  for (int i = 0; i < data_size/sizeof(float); ++i)
    std::cout << reinterpret_cast<const float*>(data)[i] << ", ";
  std::cout << "\n";
}

std::unique_ptr<luci::Module> importModel(const std::string &filename)
{
  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
  {
    throw std::runtime_error("Cannot open model file \"" + filename + "\".\n");
  }
  std::vector<char> model_data((std::istreambuf_iterator<char>(fs)),
                               std::istreambuf_iterator<char>());
  return luci::Importer().importModule(circle::GetModel(model_data.data()));
}

template <typename NodeT> size_t getTensorSize(const NodeT *node)
{
  uint32_t tensor_size = loco::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

} // namespace

/*
 * @brief EvalTester main
 *
 *        Driver for testing luci-inerpreter
 *
 */
int entry(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr
        << "Usage: " << argv[0]
        << " <path/to/circle/model> <num_inputs> <path/to/input/prefix> <path/to/output/file>\n";
    return EXIT_FAILURE;
  }

  const char *filename = argv[1];
/*  const int32_t num_inputs = atoi(argv[2]);
  const char *input_prefix = argv[3];
  const char *output_file = argv[4];
  const std::string intermediate_filename = std::string(filename) + ".inter.circle";*/

  // Import model again
  std::unique_ptr<luci::Module> module = importModel(filename);
  if (module == nullptr)
  {
    std::cerr << "ERROR: Failed to load '" << filename << "'" << std::endl;
    return EXIT_FAILURE;
  }

  // Create interpreter.
  luci_interpreter::Interpreter interpreter(module.get());

  // Set input.
  // Data for n'th input is read from ${input_prefix}n
  // (ex: Add.circle.input0, Add.circle.input1 ..)
  const auto input_nodes = loco::input_nodes(module->graph());
  for (int32_t i = 0; i < input_nodes.size(); i++)
  {
    const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[i]);
    std::vector<char> input_data(getTensorSize(input_node));
    fillInputWithOnes(input_data.data(), input_data.size());
    interpreter.writeInputTensor(input_node, input_data.data(), input_data.size());
  }

  // Do inference.
  interpreter.interpret();

  // Get output.
  const auto output_nodes = loco::output_nodes(module->graph());
  for (int i = 0; i < module->graph()->outputs()->size(); i++)
  {
    const auto *output_node = loco::must_cast<const luci::CircleOutput *>(output_nodes[i]);
    std::vector<char> output_data(getTensorSize(output_node));
    interpreter.readOutputTensor(output_node, output_data.data(), output_data.size());

    // Output data is written in ${output_file}
    // (ex: Add.circle.output0)
    // Output shape is written in ${output_file}.shape
    // (ex: Add.circle.output0.shape)
    printOutput(output_data.data(), output_data.size());
  }
  return EXIT_SUCCESS;
}

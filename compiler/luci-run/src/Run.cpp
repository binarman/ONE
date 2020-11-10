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

#include <arser/arser.h>
#include <vconone/vconone.h>

#include <luci/Importer.h>
#include <luci/IR/Module.h>
#include <luci_interpreter/Interpreter.h>
#include <loco/IR/DataTypeTraits.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

void fill_in_tensor(std::vector<char> &data, loco::DataType dtype)
{
  switch (dtype)
  {
    case loco::DataType::FLOAT32:
      for (int i = 0; i < data.size() / sizeof(float); ++i)
      {
        reinterpret_cast<float *>(data.data())[i] = 123.f;
      }
      break;
    default:
      assert(false);
  }
}

int main(const int argc, char **argv)
{
  if (argc != 2)
  {
    std::cout << "need path to circle model\n";
    return 1;
  }
  std::string model_path = argv[1];
  // Load model from the file
  std::ifstream fs(model_path, std::ifstream::binary);
  if (fs.fail())
  {
    throw std::runtime_error("Cannot open model file \"" + model_path + "\".\n");
  }
  std::vector<char> model_data((std::istreambuf_iterator<char>(fs)),
                               std::istreambuf_iterator<char>());

  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(model_data.data()),
                                 model_data.size()};
  if (!circle::VerifyModelBuffer(verifier))
  {
    throw std::runtime_error("ERROR: Failed to verify circle '" + model_path + "'");
  }

  auto module = luci::Importer().importModule(circle::GetModel(model_data.data()));

  if (module == nullptr)
  {
    throw std::runtime_error("ERROR: Failed to load '" + model_path + "'");
  }

  // Initialize interpreter
  auto interpreter = std::make_unique<luci_interpreter::Interpreter>(module.get());

  auto nodes = module->graph()->nodes();
  auto nodes_count = nodes->size();

  // Fill input tensors with some garbage
  for (int i = 0; i < nodes_count; ++i)
  {
    auto *node = dynamic_cast<luci::CircleNode *>(nodes->at(i));
    assert(node);
    if (node->opcode() == luci::CircleOpcode::CIRCLEINPUT)
    {
      auto *input_node = static_cast<luci::CircleInput *>(node);
      loco::GraphInput *g_input = module->graph()->inputs()->at(input_node->index());
      const loco::TensorShape *shape = g_input->shape();
      size_t data_size = 1;
      for (int d = 0; d < shape->rank(); ++d)
      {
        assert(shape->dim(d).known());
        data_size *= shape->dim(d).value();
      }
      data_size *= loco::size(g_input->dtype());
      std::vector<char> data(data_size);
      fill_in_tensor(data, g_input->dtype());

      interpreter->writeInputTensor(static_cast<luci::CircleInput *>(node), data.data(), data_size);
    }
  }

  int min = std::numeric_limits<int>::max();
  int max = std::numeric_limits<int>::min();
  int64_t avg = 0;
  int64_t square_sum = 0;
  constexpr int N = 50;
  std::cout << "[";
  for (int i = 0; i < N; ++i)
  {
    auto start = std::chrono::system_clock::now();
    interpreter->interpret();
    auto finish  = std::chrono::system_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    max = std::max(max, duration);
    min = std::min(min, duration);
    avg += duration;
    square_sum += duration * duration;
    std::cout << duration << ", ";
    std::cout.flush();
  }
  std::cout << "]\n";
  std::cout << "min: " << min << " ms\n";
  std::cout << "max: " << max << " ms\n";
  std::cout << "avg: " << avg / N << " ms\n";
  std::cout << "var: " << sqrt(square_sum/N - (avg*avg) / (N*N)) << " ms\n";
  return 0;
}

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

#include "LuciCodegen.h"
#include "luci/Importer.h"
#include "luci/CircleExporter.h"
#include "luci/CircleFileExpContract.h"

#include <iostream>
#include <string>

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    std::cout << "Usage: ./circle_codegen <input circle file> <output package name>\n";
    return 1;
  }
  std::string input_circle_name = argv[1];
  std::string output_package_name = argv[2];
  luci::Importer importer;
  const circle::Model *circle_module;
  std::unique_ptr<luci::Module> luci_module = importer.importModule(circle_module);

  luci_codegen::Options options;
  // set options if needed
  luci_codegen::LuciCodegen codegen(options);
  codegen.process_module(*luci_module);
  codegen.emit_code(output_package_name);

  luci::CircleExporter exporter;
  luci::CircleFileExpContract contract(luci_module.get(), output_package_name + ".circle");
  exporter.invoke(&contract);
  return 0;
}

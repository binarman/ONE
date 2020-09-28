/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "LinearExecutor.h"
#ifdef RUY_PROFILER
#include "ruy/profiler/instrumentation.h"
#endif

namespace onert
{
namespace exec
{

//#ifdef RUY_PROFILER
namespace
{

std::ostream &operator<<(std::ostream &os, const std::vector<int32_t> &v)
{
  os << "[";
  for (auto i : v)
    os << i << ", ";
  os << "]";
  return os;
}

std::vector<int32_t> simplify_shape(const std::vector<int32_t> &shape)
{
  size_t non_one_dim = 0;
  while (non_one_dim < shape.size() && shape[non_one_dim] == 1) ++non_one_dim;
  std::vector<int32_t> simple_shape;
  for (size_t i = non_one_dim; i < shape.size(); ++i)
    simple_shape.push_back(shape[i]);
  return simple_shape;
}

bool shapes_equal(const std::vector<int32_t> &shape1, const std::vector<int32_t> &shape2)
{
  if (shape1.size() != shape2.size())
    return false;
  for (size_t i = 0; i < shape1.size(); ++i)
    if (shape1[i] != shape2[i])
      return false;
  return true;
}

char *seq_to_label(const onert::ir::OpSequence *op_seq, const onert::ir::Operations &operations, const onert::ir::Operands &operands)
{
  bool is_sparse = false;
  bool need_broadcast = false;
  auto &operation = operations.at(*op_seq->begin());
  const auto &inputs = operation.getInputs();
  if (operation.opcode() == onert::ir::OpCode::BinaryArithmetic)
  {
    auto &lhs = operands.at(inputs.at(0));
    auto &rhs = operands.at(inputs.at(1));
    const onert::ir::Shape & lhs_shape = lhs.shape();
    const onert::ir::Shape & rhs_shape = rhs.shape();
    need_broadcast = !shapes_equal(lhs_shape.dims(), rhs_shape.dims());
  }
  for (const onert::ir::OperandIndex &idx : inputs)
  {
    if (operands.exist(idx) && operands.at(idx).typeInfo().sparsity())
    {
      is_sparse = true;
      break;
    }
  }
  auto node_name = operation.name();
  if (is_sparse)
  {
    node_name += "_sparse";
  }
  if (need_broadcast)
  {
    node_name += "_broadcast";
    auto &lhs = operands.at(inputs.at(0));
    auto &rhs = operands.at(inputs.at(1));
    std::vector<int32_t> simplified_lhs_shape = simplify_shape(lhs.shape().dims());
    std::vector<int32_t> simplified_rhs_shape = simplify_shape(rhs.shape().dims());
    // std::cerr << operation.name() << " " << lhs.shape().dims() << " " << rhs.shape().dims() << "\n";
    bool broadcast_simplification = shapes_equal(simplified_lhs_shape, simplified_rhs_shape);
    bool scalar_operand = (simplified_rhs_shape.size() == 0) || (simplified_lhs_shape.size() == 0);
    if (!(broadcast_simplification || scalar_operand))
      std::cerr << "FAILED TO OPTIMIZE " << operation.name() << " " << lhs.shape().dims() << " " << rhs.shape().dims() << "\n";
  }
  char *cstr = new char[node_name.length() + 1];
  std::strcpy(cstr, node_name.c_str());
  return cstr;
}
} // namespace
//#endif

void LinearExecutor::executeImpl()
{
  _subject.notifyModelBegin(this);
  for (auto &&code : _code)
  {
    const auto op_seq = code.op_seq;
    const auto backend = code.lower_info->backend();
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
    ruy::profiler::ScopeLabel label(seq_to_label(op_seq, _graph.operations(), _graph.operands()));
#endif
    _subject.notifyJobBegin(this, op_seq, backend);

    auto &fn_seq = code.fn_seq;
    bool handle_dynamic_tensor = op_seq->has_dynamic_tensor() || hasDynamicInput();

    fn_seq->enableDynamicShapeInferer(handle_dynamic_tensor);
    fn_seq->run();

    _subject.notifyJobEnd(this, op_seq, backend);
  }
  _subject.notifyModelEnd(this);
}

} // namespace exec
} // namespace onert

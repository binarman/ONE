/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

/*
 * Copyright (c) 2018-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef __ARM_COMPUTE_CLARGMINMAXLAYEREX_H__
#define __ARM_COMPUTE_CLARGMINMAXLAYEREX_H__

#include "arm_compute/core/CL/kernels/CLArgMinMaxLayerKernelEx.h"
#include "arm_compute/core/CL/kernels/CLReshapeLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
class ITensorInfo;
class ICLTensor;

/** Function to calculate the index of the minimum or maximum values in a
 *  tensor based on an axis.
 *
 * @note The default data type for an uninitialized output tensor is
 *       signed 32-bit integer (S32). It is the user's responsibility to check
 *       that the results do not overflow because the indices are computed
 *       in unsigned 32-bit (U32).
 */
class CLArgMinMaxLayerEx : public IFunction
{
public:
  /** Default Constructor.
   *
   * @param[in] memory_manager (Optional) Memory manager.
   */
  CLArgMinMaxLayerEx(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
  /** Set the input and output tensors.
   *
   * @param[in]  input  Input source tensor. Data types supported: QASYMM8/F16/F32.
   * @param[in]  axis   Axis to find max/min index.
   * @param[out] output Output source tensor. Data types supported: U32/S32.
   * @param[in]  op     Reduction operation to perform. Operations supported: ARG_IDX_MAX,
   * ARG_IDX_MIN
   */
  void configure(const ICLTensor *input, int axis, ICLTensor *output, const ReductionOperation &op);
  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLArgMinMaxLayerEx
   *
   * @param[in] input  Input source tensor info. Data types supported: QASYMM8/F16/F32.
   * @param[in] axis   Axis to find max/min index.
   * @param[in] output Output source tensor info. Data types supported: U32/S32.
   * @param[in] op     Reduction operation to perform. Operations supported: ARG_IDX_MAX,
   * ARG_IDX_MIN
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, int axis, const ITensorInfo *output,
                         const ReductionOperation &op);

  // Inherited methods overridden:
  void run() override;

private:
  MemoryGroup _memory_group;
  std::vector<CLTensor> _results_vector;
  CLTensor _not_reshaped_output;
  std::vector<CLArgMinMaxLayerKernelEx> _reduction_kernels_vector;
  CLReshapeLayerKernel _reshape_kernel;
  unsigned int _num_of_stages;
  unsigned int _reduction_axis;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLARGMINMAXLAYEREX_H__ */

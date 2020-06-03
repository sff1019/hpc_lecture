/******************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * \brief Utilities for device introspection
 */

namespace cutlass {


/******************************************************************************
 * arch_family_t
 ******************************************************************************/

/**
 * \brief Enumeration of NVIDIA GPU architectural families
 */
struct arch_family_t
{
    /// \brief Enumerants
    enum kind_t
    {
        Unsupported     = 0,
        Kepler      = 3,
        Maxwell     = 5,
        Volta       = 7,
    };

};


/**
 * Macro for architecture targeted by the current compiler pass
 */
#if defined(__CUDA_ARCH__)
    #define CUTLASS_ARCH __CUDA_ARCH__
#else
    #define CUTLASS_ARCH 0
#endif


/**
 * Macro for architecture family targeted by the current compiler pass
 */
#define CUTLASS_ARCH_FAMILY                         \
    (                                               \
        (CUTLASS_ARCH < 300) ?                      \
            arch_family_t::Unsupported :            \
            (CUTLASS_ARCH < 500) ?                  \
                arch_family_t::Kepler :             \
                (CUTLASS_ARCH < 700) ?              \
                    arch_family_t::Maxwell :        \
                    arch_family_t::Volta            \
    )




/******************************************************************************
 * Device introspection
 ******************************************************************************/

/**
 * \brief Retrieves the count for the current device
 */
cudaError_t get_sm_count(int &sm_count)
{
    cudaError_t error = cudaSuccess;

    // Get device ordinal
    int device_ordinal;
    if (CUDA_PERROR_DEBUG(error = cudaGetDevice(&device_ordinal)))
        return error;

    // Get SM count
    if (CUDA_PERROR_DEBUG(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal)))
        return error;

    return error;
}


} // namespace cutlass



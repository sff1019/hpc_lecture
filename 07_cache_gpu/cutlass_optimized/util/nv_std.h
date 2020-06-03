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
 * \brief C++ features that may be otherwise unimplemented for CUDA device functions.
 *
 * This file has three components:
 *
 *   (1) Macros:
 *       - Empty macro defines for C++ keywords not supported by the current
 *         version of C++. These simply allow compilation to proceed (but do
 *         not provide the added semantics).
 *           - \p noexcept
 *           - \p constexpr
 *           - \p nullptr
 *           - \p static_assert
 *
 *       - Macro functions that we need in constant expressions because the
 *         C++ equivalents require constexpr compiler support.  These are
 *         prefixed with \p __NV_STD_*
 *           - \p __NV_STD_MAX
 *           - \p __NV_STD_MIN
 *
 *   (2) Re-implementations of STL functions and types:
 *       - C++ features that need the \p __device__ annotation.  These are
 *         placed into the \p nv_std namespace.
 *           - \p plus
 *           - \p less
 *           - \p greater
 *           - \p min
 *           - \p max
 *           - \p methods on std::pair (==, !=, <, <=, >, >=, and make_pair())
 *
 *   (3) Stop-gap implementations of unsupported STL functions and types:
 *       - STL functions and types defined by C++ 11/14/17/etc. that are not
 *         provided by the current version of C++. These are placed into the
 *         \p nv_std namespace
 *           - \p integral_constant
 *           - \p nullptr_t
 *           - \p true_type
 *           - \p false_type
 *           - \p bool_constant
 *           - \p enable_if
 *           - \p conditional
 *           - \p is_same
 *           - \p is_base_of
 *           - \p remove_const
 *           - \p remove_volatile
 *           - \p remove_cv
 *           - \p is_volatile
 *           - \p is_pointer
 *           - \p is_void
 *           - \p is_integral
 *           - \p is_floating_point
 *           - \p is_arithmetic
 *           - \p is_fundamental
 *           - \p is_trivially_copyable
 *           - \p alignment_of
 *           - \p aligned_storage
 *
 *   (4) Functions and types that are STL-like (but aren't in the STL):
 *           - \p TODO: min and max functors?
 *
 * The idea is that, as we drop support for older compilers, we can simply #define
 * the \p __NV_STD_XYZ macros and \p nv_std namespace to alias their C++
 * counterparts (or trivially find-and-replace their occurrences in code text).
 */


//-----------------------------------------------------------------------------
// Include STL files that nv_std provides functionality for
//-----------------------------------------------------------------------------



/******************************************************************************
 * Macros
 ******************************************************************************/

//-----------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------

/// Select maximum(a, b)
#ifndef __NV_STD_MAX
    #define __NV_STD_MAX(a, b) (((b) > (a)) ? (b) : (a))
#endif

/// Select minimum(a, b)
#ifndef __NV_STD_MIN
    #define __NV_STD_MIN(a, b) (((b) < (a)) ? (b) : (a))
#endif





/******************************************************************************
 * Implementations of C++ 11/14/17/... STL features
 ******************************************************************************/

namespace nv_std {

//-----------------------------------------------------------------------------
// Integral constant helper types <type_traits>
//-----------------------------------------------------------------------------

#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1500))

    /// std::integral_constant
    template <typename value_t, value_t V>
    struct integral_constant;

    /// std::integral_constant
    template <typename value_t, value_t V>
    struct integral_constant
    {
        static const value_t value = V;

        typedef value_t                           value_type;
        typedef integral_constant<value_t, V>     type;

        inline __host__ __device__ operator value_type() const
        {
             return value;
        }

        inline __host__ __device__ const value_type operator()() const
        {
            return value;
        }
    };


#else

    using std::integral_constant;
    using std::pair;

#endif

    /// The type used as a compile-time boolean with true value.
    typedef integral_constant<bool, true>   true_type;

    /// The type used as a compile-time boolean with false value.
    typedef integral_constant<bool, false>  false_type;




    //-----------------------------------------------------------------------------
    // Conditional metaprogramming <type_traits>
    //-----------------------------------------------------------------------------

#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1600))

    /// std::enable_if (true specialization)
    template<bool C, typename T = void>
    struct enable_if {
      typedef T type;
    };

    /// std::enable_if (false specialization)
    template<typename T>
    struct enable_if<false, T> { };


    /// std::conditional (true specialization)
    template<bool B, class T, class F>
    struct conditional { typedef T type; };

    /// std::conditional (false specialization)
    template<class T, class F>
    struct conditional<false, T, F> { typedef F type; };

#else

    using std::enable_if;
    using std::conditional;

#endif



    //-----------------------------------------------------------------------------
    // Type relationships <type_traits>
    //-----------------------------------------------------------------------------

#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1500))

    /// std::is_same (false specialization)
    template <typename A, typename B>
    struct is_same : false_type
    {};

    /// std::is_same (true specialization)
    template <typename A>
    struct is_same<A, A> : true_type
    {};


    /// Helper for std::is_base_of
    template<typename BaseT, typename DerivedT>
    struct is_base_of_helper
    {
        typedef char (&yes)[1];
        typedef char (&no)[2];

        template<typename B, typename D>
        struct dummy
        {
            operator B*() const;
            operator D*();
        };

        template<typename T>
        static yes check(DerivedT*, T);

        static no check(BaseT*, int);

        static const bool value = sizeof(check(dummy<BaseT, DerivedT>(), int())) == sizeof(yes);
    };

    /// std::is_base_of
    template <typename BaseT, typename DerivedT>
    struct is_base_of : integral_constant<
        bool,
        (is_base_of_helper<typename remove_cv<BaseT>::type, typename remove_cv<DerivedT>::type>::value) ||
            (is_same<typename remove_cv<BaseT>::type, typename remove_cv<DerivedT>::type>::value)>
    {};


#else

    using std::is_same;
    using std::is_base_of;

#endif






}; // namespace nv_std


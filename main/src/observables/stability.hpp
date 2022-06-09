/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
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

/*! @file
 * @brief Global and local central value of density and radius
 *
 * @author Ruben Cabezon
 */

#pragma once

#include <vector>
#include <iostream>

#include "mpi.h"

#include "cstone/util/array.hpp"

namespace sphexa
{

  struct greater
  {
  template<class T> bool operator()(T const &a, T const &b) const { return a > b; }
  };

  /*! @brief local calculation of the maximum density (usually central) and radius
   *
   * @tparam        T            double or float
   * @param[in]     startIndex   first locally assigned particle index of buffers in @p d
   * @param[in]     endIndex     last locally assigned particle index of buffers in @p d
   * @param[in]     x            X coordinate array
   * @param[in]     y            Y coordinate array
   * @param[in]     z            Z coordinate array
   * @param[in]     rho          baryonic density
   *
   * Returns the 50 local particles with higher density and the 50 local particles
   * with higher radius. Sort function uses greater to sort in reverse order so
   * that we can benefit from resize to cut the vectors down to 50.
   */
  template<class T> util::tuple<std::vector<T>, std::vector<T>>
  localStabil(size_t startIndex, size_t endIndex, size_t n, const T* x, const T* y, const T* z,
                                    const T* kx, const T* m, const T* xm)
  {
      std::vector<T> radius(n);
      std::vector<T> localDensity(n);

  #pragma omp parallel for
      for (size_t i = startIndex; i < endIndex; i++)
      {
        localDensity[i-startIndex] = kx[i] * m[i] / xm[i];
        radius[i-startIndex]       = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
      }

      std::sort(localDensity.begin(), localDensity.end(), greater());
      std::sort(radius.begin(), radius.end(), greater());

      localDensity.resize(50);
      radius.resize(50);

      return{localDensity, radius};
  }

  /*! @brief global calculation of the central density and radius
   *
   * @tparam        T            double or float
   * @tparam        Dataset
   * @tparam        Box
   * @param[in]     startIndex   first locally assigned particle index of buffers in @p d
   * @param[in]     endIndex     last locally assigned particle index of buffers in @p d
   * @param[in]     d            particle data set
   * @param[in]     box          bounding box
   */
  template<typename T, class Dataset> util::tuple<T, T>
  computeStabil(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
  {
      auto [localDensity, localRadius] = localStabil(
          startIndex, endIndex, d.x.size(), d.x.data(), d.y.data(), d.z.data(), d.kx.data(), d.m.data(), d.xm.data());

      int rootRank = 0;
      int mpiranks;

      MPI_Comm_size(d.comm, &mpiranks);

      size_t rootsize = 50 * mpiranks;

      std::vector<T> globalDensity(rootsize);
      std::vector<T> globalRadius(rootsize);

      MPI_Gather(localDensity.data(), 50, MpiType<T>{}, globalDensity.data(), 50, MpiType<T>{}, rootRank, d.comm);
      MPI_Gather(localRadius.data(), 50, MpiType<T>{}, globalRadius.data(), 50, MpiType<T>{}, rootRank, d.comm);

      int rank;
      MPI_Comm_rank(d.comm, &rank);

      T centralDensity = 0.;
      T radius         = 0.;

      if (rank == 0)
      {
        std::sort(globalDensity.begin(), globalDensity.end(), greater());
        std::sort(globalRadius.begin(), globalRadius.end(), greater());


        for (size_t i = 0; i < 50; i++)
        {
          centralDensity += globalDensity[i];
          radius         += globalRadius[i];
        }
      }
      return {centralDensity/T(50.), radius/T(50.)};
  }

} // namespace sphexa

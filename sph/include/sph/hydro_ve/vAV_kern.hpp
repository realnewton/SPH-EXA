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
 * @brief Divergence of velocity vector field
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"

#include "sph/kernels.hpp"
#include "sph/math.hpp"
#include "sph/tables.hpp"

namespace sph
{

template<typename T>
HOST_DEVICE_FUN inline void vAVJLoop(int i, T sincIndex, T K, const cstone::Box<T>& box, const int* neighbors,
                                     int neighborsCount, const T* x, const T* y, const T* z, const T* h, const T* dvxdx,
                                     const T* dvxdy, const T* dvxdz, const T* dvydx, const T* dvydy, const T* dvydz,
                                     const T* dvzdx, const T* dvzdy, const T* dvzdz,T* vxAV, T* vyAV, T* vzAV)
{
    T xi  = x[i];
    T yi  = y[i];
    T zi  = z[i];
    T hi  = h[i];

    T dvxdxi  = dvxdx[i];
    T dvxdyi  = dvxdy[i];
    T dvxdzi  = dvxdz[i];
    T dvydxi  = dvydx[i];
    T dvydyi  = dvydy[i];
    T dvydzi  = dvydz[i];
    T dvzdxi  = dvzdx[i];
    T dvzdyi  = dvzdy[i];
    T dvzdzi  = dvzdz[i];

    for (int pj = 0; pj < neighborsCount; ++pj)
    {
        int j = neighbors[pj];

        T rx = xi - x[j];
        T ry = yi - y[j];
        T rz = zi - z[j];

        T hj  = h[j];

        applyPBC(box, T(2) * hi, rx, ry, rz);

        T r2   = rx * rx + ry * ry + rz * rz;
        T dist = std::sqrt(r2);

        T dvxdxj  = dvxdx[j];
        T dvxdyj  = dvxdy[j];
        T dvxdzj  = dvxdz[j];
        T dvydxj  = dvydx[j];
        T dvydyj  = dvydy[j];
        T dvydzj  = dvydz[j];
        T dvzdxj  = dvzdx[j];
        T dvzdyj  = dvzdy[j];
        T dvzdzj  = dvzdz[j];

        T dmy1 = dvxdxi * rx * rx + dvydxi * rx * ry + dvzdxi * rx * rz +
                 dvxdyi * ry * rx + dvydyi * ry * ry + dvzdyi * ry * rz +
                 dvxdzi * rz * rx + dvydzi * rz * ry + dvzdzi * rz * rz;
        T dmy2 = dvxdxj * rx * rx + dvydxj * rx * ry + dvzdxj * rx * rz +
                 dvxdyj * ry * rx + dvydyj * ry * ry + dvzdyj * ry * rz +
                 dvxdzj * rz * rx + dvydzj * rz * ry + dvzdzj * rz * rz;
        T A_ab = T(0);
        if (dmy2 != T(0)) { A_ab = dmy1 / dmy2 }
        T eta_ab = std::min(dist / hi, dist / hj);
        T dmy3 = T(1);
        if (eta_ab < eta_crit) {dmy3 = std::exp(-(eta_ab - eta_crit) / T(0.2))**2)}
        T phi_ab = std::max(T(0), std::min(T(1), T(4) * A_ab / (T(1) + A_ab)**2)) * dmy3


    }

    divv[i]  = K * hiInv3 * divvi / kxi;
    curlv[i] = K * hiInv3 * std::abs(std::sqrt(curlv_x * curlv_x + curlv_y * curlv_y + curlv_z * curlv_z)) / kxi;

    dvxdx[i] = dvxdxi / kxi;
    dvxdy[i] = dvxdyi / kxi;
    dvxdz[i] = dvxdzi / kxi;
    dvydx[i] = dvydxi / kxi;
    dvydy[i] = dvydyi / kxi;
    dvydz[i] = dvydzi / kxi;
    dvzdx[i] = dvzdxi / kxi;
    dvzdy[i] = dvzdyi / kxi;
    dvzdz[i] = dvzdzi / kxi;

}

} // namespace sph

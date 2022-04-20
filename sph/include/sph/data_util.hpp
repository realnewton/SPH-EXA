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
 * @brief Utility functions to resolve names of particle fields to pointers
 */

#pragma once

#include <vector>

#include "traits.hpp"

namespace sphexa
{

/*! @brief look up indices of field names
 *
 * @tparam     Array
 * @param[in]  allNames     array of strings with names of all fields
 * @param[in]  subsetNames  array of strings of field names to look up in @p allNames
 * @return                  the indices of @p subsetNames in @p allNames
 */
template<class Array>
std::vector<int> fieldStringsToInt(const Array& allNames, const std::vector<std::string>& subsetNames)
{
    std::vector<int> subsetIndices;
    subsetIndices.reserve(subsetNames.size());
    for (const auto& field : subsetNames)
    {
        auto it = std::find(allNames.begin(), allNames.end(), field);
        if (it == allNames.end()) { throw std::runtime_error("Field " + field + " does not exist\n"); }

        size_t fieldIndex = it - allNames.begin();
        subsetIndices.push_back(fieldIndex);
    }
    return subsetIndices;
}

//! @brief extract a vector of pointers to particle fields for file output
template<class Dataset>
auto getOutputArrays(Dataset& dataset)
{
    using T            = typename Dataset::RealType;
    auto fieldPointers = dataset.data();

    std::vector<const T*> outputFields(dataset.outputFields.size());
    std::transform(dataset.outputFields.begin(),
                   dataset.outputFields.end(),
                   outputFields.begin(),
                   [&fieldPointers](int i) { return fieldPointers[i]->data(); });

    return outputFields;
}

/*! @brief resizes all active particles fields of @p d to the specified size
 *
 * Important Note: this only resizes the fields that are listed either as conserved or dependent.
 * The conserved/dependent list may be set at runtime, depending on the need of the simulation!
 */
template<class Dataset>
void resize(Dataset& d, size_t size)
{
    double growthRate = 1.05;
    auto   data_      = d.data();

    for (int i : d.conservedFields)
    {
        reallocate(*data_[i], size, growthRate);
    }
    for (int i : d.dependentFields)
    {
        reallocate(*data_[i], size, growthRate);
    }

    reallocate(d.codes, size, growthRate);
    reallocate(d.neighborsCount, size, growthRate);

    d.devPtrs.resize(size);
}

//! resizes the neighbors list, only used in the CPU version
template<class Dataset>
void resizeNeighbors(Dataset& d, size_t size)
{
    double growthRate = 1.05;
    //! If we have a GPU, neighbors are calculated on-the-fly, so we don't need space to store them
    reallocate(d.neighbors, HaveGpu<typename Dataset::AcceleratorType>{} ? 0 : size, growthRate);
}

} // namespace sphexa
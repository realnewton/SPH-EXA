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
 * @brief Generation of locally essential global octrees in cornerstone format
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * A locally essential octree has a certain global resolution specified by a maximum
 * particle count per leaf node. In addition, it features a focus area defined as a
 * sub-range of the global space filling curve. In this focus sub-range, the resolution
 * can be higher, expressed through a smaller maximum particle count per leaf node.
 * Crucially, the resolution is also higher in the halo-areas of the focus sub-range.
 * These halo-areas can be defined as the overlap with the smoothing-length spheres around
 * the contained particles in the focus sub-range (SPH) or as the nodes whose opening angle
 * is too big to satisfy a multipole acceptance criterion from any perspective within the
 * focus sub-range (N-body).
 */

#pragma once

#include <vector>

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/morton.hpp"
#include "cstone/halos/boxoverlap.hpp"
#include "octree_internal.hpp"
#include "traversal.hpp"

namespace cstone
{

CUDA_HOST_DEVICE_FUN
template<int Period>
constexpr int rangeSeparation(int a, int b, int c, int d, bool pbc)
{
    assert(a < b && c < d);
    int cb = c - b;
    int ad = a - d;
    int cbPbc = (pbc) ? pbcDistance<Period>(cb) : cb;
    int adPbc = (pbc) ? pbcDistance<Period>(ad) : ad;
    return (cb >= 0 || ad >= 0) * stl::min(stl::abs(cbPbc), stl::abs(adPbc));
}

/*! @brief return the smallest distance squared between two points on the surface of the AABBs @p a and @p b
 *
 * @tparam T     float or double
 * @tparam I     32- or 64-bit unsigned integer
 * @param a      a box, specified with integer coordinates in [0:2^21]
 * @param b
 * @param box    floating point coordinate bounding box
 * @return       the square of the smallest distance between a and b
 */
template<class T, class I>
CUDA_HOST_DEVICE_FUN
T minDistanceSq(IBox a, IBox b, const Box<T>& box)
{
    constexpr size_t maxCoord = 1u<<maxTreeLevel<I>{};
    constexpr T unitLengthSq  = T(1.) / (maxCoord * maxCoord);

    size_t dx = rangeSeparation<maxCoord>(a.xmin(), a.xmax(), b.xmin(), b.xmax(), box.pbcX());
    size_t dy = rangeSeparation<maxCoord>(a.ymin(), a.ymax(), b.ymin(), b.ymax(), box.pbcY());
    size_t dz = rangeSeparation<maxCoord>(a.zmin(), a.zmax(), b.zmin(), b.zmax(), box.pbcZ());
    // the maximum for any integer is 2^21-1, so we can safely square each of them
    return ((dx*dx)*box.lx()*box.lx() + (dy*dy)*box.ly()*box.ly() +
            (dz*dz)*box.lz()*box.lz()) * unitLengthSq;
}

//! @brief return longest edge length of box @p b
template<class T, class I>
CUDA_HOST_DEVICE_FUN
T nodeLength(IBox b, const Box<T>& box)
{
    constexpr int maxCoord = 1u<<maxTreeLevel<I>{};
    constexpr T unitLength = T(1.) / maxCoord;

    // IBoxes for octree nodes are assumed cubic, only box can be rectangular
    return (b.xmax() - b.xmin()) * unitLength * box.maxExtent();
}

/*! @brief evaluate minimum distance MAC
 *
 * @param a            target cell
 * @param b            source cell
 * @param box          coordinate bounding box
 * @param invThetaSq   inverse theta squared
 * @return             true if MAC fulfilled, false otherwise
 *
 * Note: Mac is valid for any point in a w.r.t to box b, therefore only the
 * size of b is relevant.
 */
template<class T, class I>
CUDA_HOST_DEVICE_FUN
bool minDistanceMac(IBox a, IBox b, const Box<T>& box, float invThetaSq)
{
    T dsq = minDistanceSq<T, I>(a, b, box);
    // equivalent to "d > l / theta"
    T bLength = nodeLength<T, I>(b, box);
    return dsq > bLength * bLength * invThetaSq;
}

template<class T, class I>
CUDA_HOST_DEVICE_FUN
void markMacPerBox(IBox target, const Octree<I>& octree, const Box<T>& box,
                   float invThetaSq, I focusStart, I focusEnd, char* markings)
{
    auto checkAndMarkMac = [target, &octree, &box, invThetaSq, focusStart, focusEnd, markings](TreeNodeIndex idx)
    {
        I nodeStart = octree.codeStart(idx);
        I nodeEnd   = octree.codeEnd(idx);
        // if the tree node with index idx is fully contained in the focus, we stop traversal
        if (containedIn(nodeStart, nodeEnd, focusStart, focusEnd)) { return false; }

        IBox sourceBox = makeIBox(nodeStart, nodeEnd);

        bool violatesMac = !minDistanceMac<T, I>(target, sourceBox, box, invThetaSq);
        if (violatesMac) { markings[idx] = 1; }

        return violatesMac;
    };

    traverse(octree, checkAndMarkMac, [](TreeNodeIndex){});
}

/*! @brief Mark all leaf nodes that fail the MAC paired with leaf nodes from a given range
 *
 * @tparam T                float or double
 * @tparam I                32- or 64-bit unsigned integer
 * @param[in]  octree       octree, including internal part
 * @param[in]  box          global coordinate bounding box
 * @param[in]  firstLeaf    first leaf index of the cornerstone tree used to build @p octree
 *                          to check for nodes failing the minimum distance Mac
 * @param[in]  lastLeaf     last leaf index
 * @param[in]  invThetaSq   1./theta^2
 * @param[out] markings     array of length @p octree.nTreeNodes(), each position i
 *                          will be set to 1, if node with index i fails the MAC paired with any
 *                          of the leaf nodes with leaf index in [firstLeaf:lastLeaf]
 */
template<class T, class I>
void markMac(const Octree<I>& octree, const Box<T>& box, I focusStart, I focusEnd,
             float invThetaSq, char* markings)

{
    std::fill(markings, markings + octree.nTreeNodes(), 0);

    // find the minimum possible number of octree node boxes to cover the entire focus
    TreeNodeIndex numFocusBoxes = spanSfcRange(focusStart, focusEnd);
    std::vector<I> focusCodes(numFocusBoxes + 1);
    spanSfcRange(focusStart, focusEnd, focusCodes.data());
    focusCodes.back() = focusEnd;

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numFocusBoxes; ++i)
    {
        IBox target = makeIBox(focusCodes[i], focusCodes[i+1]);
        markMacPerBox(target, octree, box, invThetaSq, focusStart, focusEnd, markings);
    }
}

template<class I>
inline CUDA_HOST_DEVICE_FUN
int mergeCountAndMacOp(TreeNodeIndex leafIdx, const I* cstoneTree,
                       TreeNodeIndex numInternalNodes,
                       const TreeNodeIndex* leafParents,
                       const unsigned* leafCounts, const char* macs,
                       TreeNodeIndex firstFocusNode, TreeNodeIndex lastFocusNode,
                       unsigned bucketSize)
{
    auto p = siblingAndLevel(cstoneTree, leafIdx);
    unsigned siblingIdx = p[0];
    unsigned level      = p[1];

    if (siblingIdx > 0) // 8 siblings next to each other, node can potentially be merged
    {
        // pointer to first node in sibling group
        auto g = leafCounts + leafIdx - siblingIdx;

        bool countMerge = (g[0]+g[1]+g[2]+g[3]+g[4]+g[5]+g[6]+g[7]) <= bucketSize;
        bool macMerge   = macs[leafParents[leafIdx]] == 0;
        bool inFringe   = leafIdx - siblingIdx + 8 >= firstFocusNode && leafIdx - siblingIdx < lastFocusNode;

        if (countMerge || (macMerge && !inFringe)) { return 0; } // merge
    }

    bool inFocus  = (leafIdx >= firstFocusNode && leafIdx < lastFocusNode);
    if (level < maxTreeLevel<I>{} && leafCounts[leafIdx] > bucketSize
        && (macs[numInternalNodes + leafIdx] || inFocus))
    { return 8; } // split

    return 1; // default: do nothing
}

/*! @brief Compute locally essential split or fuse decision for each octree node in parallel
 *
 * @tparam I                   32- or 64-bit unsigned integer type
 * @param[in] cstoneTree       cornerstone octree leaves, length = @p numLeafNodes
 * @param[in] numInternalNodes number of internal octree nodes
 * @param[in] numLeafNodes     number of leaf octree nodes
 * @param[in] leafParents      stores the parent node index of each leaf, length = @p numLeafNodes
 * @param[in] leafCounts       output particle counts per leaf node, length = @p numLeafNodes
 * @param[in] macs             multipole pass or fail per node, length = @p numInternalNodes + numLeafNodes
 * @param[in] firstFocusNode   first focus node in @p cstoneTree, range = [0:numLeafNodes]
 * @param[in] lastFocusNode    last focus node in @p cstoneTree, range = [0:numLeafNodes]
 * @param[in] bucketSize       maximum particle count per (leaf) node and
 *                             minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps         stores rebalance decision result for each node, length = @p nLeafNodes()
 * @return                     true if converged, false
 *
 * For each node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class I, class LocalIndex>
bool rebalanceDecisionEssential(const I* cstoneTree, TreeNodeIndex numInternalNodes, TreeNodeIndex numLeafNodes,
                                const TreeNodeIndex* leafParents,
                                const unsigned* leafCounts, const char* macs,
                                TreeNodeIndex firstFocusNode, TreeNodeIndex lastFocusNode,
                                unsigned bucketSize, LocalIndex* nodeOps)
{
    bool converged = true;
    #pragma omp parallel
    {
        bool convergedThread = true;
        #pragma omp for
        for (TreeNodeIndex leafIdx = 0; leafIdx < numLeafNodes; ++leafIdx)
        {
            int opDecision = mergeCountAndMacOp(leafIdx, cstoneTree, numInternalNodes, leafParents, leafCounts,
                                                macs, firstFocusNode, lastFocusNode, bucketSize);
            if (opDecision != 1) { convergedThread = false; }

            nodeOps[leafIdx] = opDecision;
        }
        if (!convergedThread) { converged = false; }
    }
    return converged;
}

template<class I>
class FocusedOctree
{
public:
    FocusedOctree(unsigned bucketSize, float theta)
        : bucketSize_(bucketSize), theta_(theta), counts_{bucketSize+1}, macs_{1}
    {
        tree_.update(std::vector<I>{0, nodeRange<I>(0)});
    }

    template<class T>
    bool update(const Box<T>& box, const I* codesStart, const I* codesEnd, I focusStart, I focusEnd, const Octree<I>& globalTree)
    {
        const std::vector<I>& cstoneTree = tree_.cstoneTree();

        TreeNodeIndex firstFocusNode = std::upper_bound(begin(cstoneTree), end(cstoneTree), focusStart) - begin(cstoneTree) - 1;
        TreeNodeIndex lastFocusNode  = std::lower_bound(begin(cstoneTree), end(cstoneTree), focusEnd) - begin(cstoneTree);

        macs_.resize(tree_.nTreeNodes());
        markMac(tree_, box, focusStart, focusEnd, 1.0/(theta_*theta_), macs_.data());

        std::vector<TreeNodeIndex> nodeOps(tree_.nLeafNodes() + 1);
        bool converged = rebalanceDecisionEssential(cstoneTree.data(), tree_.nInternalNodes(), tree_.nLeafNodes(), tree_.leafParents(),
                                                    counts_.data(), macs_.data(), firstFocusNode, lastFocusNode,
                                                    bucketSize_, nodeOps.data());
        std::vector<I> newCstoneTree;
        rebalanceTree(cstoneTree, newCstoneTree, nodeOps.data());
        tree_.update(std::move(newCstoneTree));

        counts_.resize(tree_.nLeafNodes());
        // local node counts
        computeNodeCounts(cstoneTree.data(), counts_.data(), nNodes(cstoneTree), codesStart, codesEnd,
                          std::numeric_limits<unsigned>::max(), true);
        // global node count sums when using distributed builds
        //if constexpr (!std::is_same_v<void, Reduce>) Reduce{}(counts);

        return converged;
    }

    const std::vector<I>& leafTree() const { return tree_.cstoneTree(); }

private:
    //! @brief max number of particles per node in focus
    unsigned bucketSize_;
    //! @brief opening angle refinement criterion
    float theta_;

    //! @brief the focused tree
    Octree<I> tree_;
    //! @brief particle counts of the focused tree
    std::vector<unsigned> counts_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    std::vector<char> macs_;
};

} // namespace cstone
#pragma once
/**
 * Grid Cell Encoding via Fractional Power Encoding (FPE)
 *
 * From the Hippocampal Design Paper (Section 2.1):
 *   "Grid Cells provide a universal metric for conceptual spaces...
 *    positions close in physical space have high cosine similarity
 *    in vector space, while distant positions are orthogonal."
 *
 * Implementation:
 *   A position (x, y) is encoded as: p(x,y) = X^x ⊛ Y^y
 *   where X and Y are basis vectors and ^ denotes repeated binding.
 *
 * Key Property:
 *   The "difference vector" between two positions encodes the transformation:
 *   diff = unbind(p(x2,y2), p(x1,y1)) ≈ X^(x2-x1) ⊛ Y^(y2-y1)
 *
 * This allows learning transformations (like "move right") independently
 * of specific positions - crucial for ARC structural generalization.
 *
 * Reference: Section 3.1.2 of "Designing Hippocampal AI System.md"
 */

#include "vsa.hpp"
#include <unordered_map>
#include <cmath>

namespace bpagi {

class GridCells {
public:
    // Maximum grid size we support
    static constexpr int MAX_COORD = 30;  // ARC grids are max 30x30

    explicit GridCells(VSA& vsa, uint32_t seed = 12345)
        : vsa_(vsa)
        , rng_(seed)
    {
        initializeBasisVectors();
        precomputePowers();
    }

    // ========================================
    // Position Encoding
    // ========================================

    /**
     * Encode a 2D position as a hypervector.
     *
     * Formula: p(x,y) = X^x ⊛ Y^y
     *
     * Properties:
     *   - Nearby positions have similar vectors
     *   - Distant positions are orthogonal
     *   - Differences encode transformations
     */
    VSA::HyperVector encodePosition(int x, int y) const {
        // Clamp to valid range
        x = std::max(-MAX_COORD, std::min(MAX_COORD, x));
        y = std::max(-MAX_COORD, std::min(MAX_COORD, y));

        // Get precomputed powers (with offset for negative indices)
        const auto& xPower = xPowers_[x + MAX_COORD];
        const auto& yPower = yPowers_[y + MAX_COORD];

        // Combine: X^x ⊛ Y^y
        return vsa_.bind(xPower, yPower);
    }

    /**
     * Compute the transformation vector between two positions.
     *
     * This is the key to structural generalization:
     *   transform = unbind(p(x2,y2), p(x1,y1))
     *             ≈ X^(x2-x1) ⊛ Y^(y2-y1)
     *
     * The same transform vector can be applied to ANY starting position
     * to get the corresponding ending position.
     */
    VSA::HyperVector computeTransform(int x1, int y1, int x2, int y2) const {
        auto p1 = encodePosition(x1, y1);
        auto p2 = encodePosition(x2, y2);
        return vsa_.unbind(p2, p1);
    }

    /**
     * Apply a transformation to a position encoding.
     *
     * newPos = bind(oldPos, transform)
     */
    VSA::HyperVector applyTransform(const VSA::HyperVector& position,
                                     const VSA::HyperVector& transform) const {
        return vsa_.bind(position, transform);
    }

    // ========================================
    // Canonical Transform Vectors
    // ========================================

    /**
     * Get the "move right" transformation vector.
     * Applying this to any position shifts it +1 in X.
     */
    const VSA::HyperVector& moveRight() const { return moveRight_; }

    /**
     * Get the "move left" transformation vector.
     */
    const VSA::HyperVector& moveLeft() const { return moveLeft_; }

    /**
     * Get the "move down" transformation vector.
     */
    const VSA::HyperVector& moveDown() const { return moveDown_; }

    /**
     * Get the "move up" transformation vector.
     */
    const VSA::HyperVector& moveUp() const { return moveUp_; }

    /**
     * Get transformation for arbitrary delta.
     */
    VSA::HyperVector getDeltaTransform(int dx, int dy) const {
        return encodePosition(dx, dy);
    }

    // ========================================
    // Scene Encoding
    // ========================================

    /**
     * Encode an entire grid scene as a single hypervector.
     *
     * From the paper: "A concept like 'Red Square' is the binding of
     * the vector for 'Red' and 'Square'."
     *
     * For a grid, we bind each cell's (position, value) pair and
     * then bundle all pairs together:
     *
     *   scene = Σ (position_i ⊛ value_i)
     *
     * @param grid Flattened grid values (row-major)
     * @param width Grid width
     * @param height Grid height
     * @param valueVectors Pre-assigned vectors for each value (0-9 for ARC)
     */
    VSA::HyperVector encodeScene(
        const std::vector<uint8_t>& grid,
        int width, int height,
        const std::vector<VSA::HyperVector>& valueVectors) const
    {
        std::vector<VSA::HyperVector> bindings;
        bindings.reserve(width * height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                if (idx >= static_cast<int>(grid.size())) break;

                uint8_t value = grid[idx];
                if (value < valueVectors.size()) {
                    // Bind position with value: (x,y) ⊛ color
                    auto posVec = encodePosition(x, y);
                    auto binding = vsa_.bind(posVec, valueVectors[value]);
                    bindings.push_back(binding);
                }
            }
        }

        return vsa_.bundle(bindings);
    }

    /**
     * Query what value is at a position in an encoded scene.
     *
     *   query = unbind(scene, position)
     *
     * Then find the most similar value vector.
     */
    int queryPosition(
        const VSA::HyperVector& scene,
        int x, int y,
        const std::vector<VSA::HyperVector>& valueVectors) const
    {
        auto posVec = encodePosition(x, y);
        auto query = vsa_.unbind(scene, posVec);

        // Find most similar value vector
        int bestValue = -1;
        float bestSim = -1.0f;

        for (size_t i = 0; i < valueVectors.size(); i++) {
            float sim = vsa_.similarity(query, valueVectors[i]);
            if (sim > bestSim) {
                bestSim = sim;
                bestValue = static_cast<int>(i);
            }
        }

        return bestValue;
    }

    // ========================================
    // Accessors
    // ========================================

    const VSA::HyperVector& getBasisX() const { return basisX_; }
    const VSA::HyperVector& getBasisY() const { return basisY_; }

private:
    VSA& vsa_;
    std::mt19937 rng_;

    // Basis vectors for X and Y dimensions
    VSA::HyperVector basisX_;
    VSA::HyperVector basisY_;

    // Precomputed powers: X^n for n in [-MAX_COORD, MAX_COORD]
    std::vector<VSA::HyperVector> xPowers_;
    std::vector<VSA::HyperVector> yPowers_;

    // Canonical movement transforms
    VSA::HyperVector moveRight_;
    VSA::HyperVector moveLeft_;
    VSA::HyperVector moveDown_;
    VSA::HyperVector moveUp_;

    void initializeBasisVectors() {
        // Generate random orthogonal basis vectors
        basisX_ = vsa_.random();
        basisY_ = vsa_.random();

        // Precompute canonical movements
        // move_right = X^1 ⊛ Y^0 = X
        // move_left = X^(-1) ⊛ Y^0 (inverse of X)
        moveRight_ = basisX_;
        moveLeft_ = basisX_;  // For XOR, inverse is same (self-inverse)
        moveDown_ = basisY_;
        moveUp_ = basisY_;
    }

    void precomputePowers() {
        // Precompute X^n and Y^n for n in [-MAX_COORD, MAX_COORD]
        size_t size = 2 * MAX_COORD + 1;
        xPowers_.resize(size);
        yPowers_.resize(size);

        // Identity vector for power 0
        VSA::HyperVector identity = vsa_.zero();
        // For XOR binding, identity should be all zeros
        // bind(A, 0) = A

        // X^0 = identity, Y^0 = identity
        xPowers_[MAX_COORD] = identity;
        yPowers_[MAX_COORD] = identity;

        // Compute positive powers: X^n = X^(n-1) ⊛ X
        VSA::HyperVector xAccum = identity;
        VSA::HyperVector yAccum = identity;

        for (int n = 1; n <= MAX_COORD; n++) {
            xAccum = vsa_.bind(xAccum, basisX_);
            yAccum = vsa_.bind(yAccum, basisY_);
            xPowers_[MAX_COORD + n] = xAccum;
            yPowers_[MAX_COORD + n] = yAccum;
        }

        // Compute negative powers: X^(-n) = inverse of X^n
        // For XOR, inverse is same as original
        xAccum = identity;
        yAccum = identity;

        for (int n = 1; n <= MAX_COORD; n++) {
            xAccum = vsa_.bind(xAccum, basisX_);
            yAccum = vsa_.bind(yAccum, basisY_);
            xPowers_[MAX_COORD - n] = xAccum;
            yPowers_[MAX_COORD - n] = yAccum;
        }
    }
};

}  // namespace bpagi

#pragma once
/**
 * Vector Symbolic Architecture (VSA) / Hyperdimensional Computing
 *
 * From the Hippocampal Design Paper:
 *   "VSA provides a mathematically rigorous framework for distributed
 *    representations that aligns with biological plausibility."
 *
 * Key Operations:
 *   - Binding (⊛): Associates two concepts → new orthogonal vector
 *   - Bundling (+): Superimposes concepts → similar to all inputs
 *   - Permutation (ρ): Encodes sequence/position
 *
 * This implementation uses Binary Sparse Block Codes (BSBC) for efficiency:
 *   - Vectors are binary (0/1) stored as uint64_t blocks
 *   - XOR for binding (fast, invertible)
 *   - Majority vote for bundling
 *   - Circular shift for permutation
 *
 * Reference: Section 3.1 of "Designing Hippocampal AI System.md"
 */

#include <vector>
#include <cstdint>
#include <random>
#include <cmath>
#include <algorithm>

namespace bpagi {

class VSA {
public:
    // Default dimension: 4096 bits = 64 uint64_t blocks
    // Paper suggests 10,000 but 4096 is more efficient (power of 2)
    static constexpr size_t DEFAULT_DIM = 4096;
    static constexpr size_t BITS_PER_BLOCK = 64;

    using HyperVector = std::vector<uint64_t>;

    explicit VSA(size_t dimension = DEFAULT_DIM, uint32_t seed = 42)
        : dimension_(dimension)
        , numBlocks_((dimension + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK)
        , rng_(seed)
    {}

    // ========================================
    // Vector Generation
    // ========================================

    /**
     * Generate a random hypervector.
     * Each bit has 50% probability of being 1.
     * Random vectors are quasi-orthogonal in high dimensions.
     */
    HyperVector random() {
        HyperVector v(numBlocks_);
        std::uniform_int_distribution<uint64_t> dist;
        for (size_t i = 0; i < numBlocks_; i++) {
            v[i] = dist(rng_);
        }
        return v;
    }

    /**
     * Generate a zero vector.
     */
    HyperVector zero() const {
        return HyperVector(numBlocks_, 0);
    }

    /**
     * Generate a vector with all ones.
     */
    HyperVector ones() const {
        return HyperVector(numBlocks_, ~0ULL);
    }

    // ========================================
    // Core VSA Operations
    // ========================================

    /**
     * BINDING (⊛): XOR operation
     *
     * Properties:
     *   - bind(A, B) is orthogonal to both A and B
     *   - bind(bind(A, B), B) ≈ A (self-inverse)
     *   - Commutative: bind(A, B) = bind(B, A)
     *
     * Use: Associate two concepts (e.g., "Red" ⊛ "Square" = "RedSquare")
     */
    HyperVector bind(const HyperVector& a, const HyperVector& b) const {
        HyperVector result(numBlocks_);
        for (size_t i = 0; i < numBlocks_; i++) {
            result[i] = a[i] ^ b[i];
        }
        return result;
    }

    /**
     * UNBIND: Same as bind (XOR is self-inverse)
     * unbind(bind(A, B), B) ≈ A
     */
    HyperVector unbind(const HyperVector& bound, const HyperVector& key) const {
        return bind(bound, key);  // XOR is its own inverse
    }

    /**
     * BUNDLING (+): Majority vote (thresholded sum)
     *
     * Properties:
     *   - bundle({A, B, C}) is similar to A, B, and C
     *   - Supports superposition of multiple items
     *
     * Use: Create a set/bag of concepts
     */
    HyperVector bundle(const std::vector<HyperVector>& vectors) const {
        if (vectors.empty()) return zero();
        if (vectors.size() == 1) return vectors[0];

        HyperVector result(numBlocks_, 0);
        size_t threshold = vectors.size() / 2;

        // Count bits at each position
        for (size_t block = 0; block < numBlocks_; block++) {
            for (size_t bit = 0; bit < BITS_PER_BLOCK; bit++) {
                size_t count = 0;
                for (const auto& v : vectors) {
                    if (v[block] & (1ULL << bit)) count++;
                }
                // Majority vote
                if (count > threshold) {
                    result[block] |= (1ULL << bit);
                } else if (count == threshold && (rng_() & 1)) {
                    // Tie-breaker: random
                    result[block] |= (1ULL << bit);
                }
            }
        }
        return result;
    }

    /**
     * PERMUTATION (ρ): Circular bit shift
     *
     * Properties:
     *   - permute(A, n) is orthogonal to A for n > 0
     *   - Used for encoding sequence position
     *
     * Use: Encode order (e.g., "first item", "second item")
     */
    HyperVector permute(const HyperVector& v, int shift) const {
        if (shift == 0) return v;

        // Normalize shift to positive
        shift = ((shift % static_cast<int>(dimension_)) + dimension_) % dimension_;

        HyperVector result(numBlocks_, 0);

        for (size_t i = 0; i < dimension_; i++) {
            size_t srcBlock = i / BITS_PER_BLOCK;
            size_t srcBit = i % BITS_PER_BLOCK;

            size_t dstPos = (i + shift) % dimension_;
            size_t dstBlock = dstPos / BITS_PER_BLOCK;
            size_t dstBit = dstPos % BITS_PER_BLOCK;

            if (v[srcBlock] & (1ULL << srcBit)) {
                result[dstBlock] |= (1ULL << dstBit);
            }
        }
        return result;
    }

    // ========================================
    // Similarity & Distance
    // ========================================

    /**
     * Hamming distance: Number of differing bits
     */
    size_t hammingDistance(const HyperVector& a, const HyperVector& b) const {
        size_t dist = 0;
        for (size_t i = 0; i < numBlocks_; i++) {
            dist += __builtin_popcountll(a[i] ^ b[i]);
        }
        return dist;
    }

    /**
     * Cosine similarity approximation for binary vectors.
     * Returns value in [-1, 1] range.
     * 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
     */
    float similarity(const HyperVector& a, const HyperVector& b) const {
        size_t dist = hammingDistance(a, b);
        // Convert Hamming distance to similarity
        // similarity = 1 - 2*(dist/dimension)
        return 1.0f - 2.0f * static_cast<float>(dist) / dimension_;
    }

    /**
     * Check if two vectors are similar (above threshold).
     */
    bool isSimilar(const HyperVector& a, const HyperVector& b,
                   float threshold = 0.5f) const {
        return similarity(a, b) >= threshold;
    }

    // ========================================
    // Utility
    // ========================================

    size_t getDimension() const { return dimension_; }
    size_t getNumBlocks() const { return numBlocks_; }

    /**
     * Count number of 1 bits in vector.
     */
    size_t popcount(const HyperVector& v) const {
        size_t count = 0;
        for (uint64_t block : v) {
            count += __builtin_popcountll(block);
        }
        return count;
    }

    /**
     * Compute density (fraction of 1 bits).
     */
    float density(const HyperVector& v) const {
        return static_cast<float>(popcount(v)) / dimension_;
    }

private:
    size_t dimension_;
    size_t numBlocks_;
    mutable std::mt19937 rng_;
};

}  // namespace bpagi

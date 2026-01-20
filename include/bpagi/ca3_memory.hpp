#pragma once
/**
 * CA3 Auto-Associative Memory (Fast Weight Programmer)
 *
 * From the Hippocampal Design Paper (Section 3.2):
 *   "The CA3 region creates an Auto-Associative Memory network...
 *    When presented with a partial or noisy cue, the recurrent dynamics
 *    drive the network state back to the stored attractor basin."
 *
 * Implementation uses Fast Weight Matrix with Hebbian outer-product learning:
 *   W_t = W_(t-1) + β_t * (v_t - W_(t-1) * k_t) ⊗ k_t^T
 *
 * Where:
 *   - k_t (Key): The cue/query pattern (e.g., partial input)
 *   - v_t (Value): The complete pattern to retrieve
 *   - β_t: Learning rate (dopamine-modulated)
 *   - ⊗: Outer product
 *
 * Retrieval: y = W * q (query → retrieved pattern)
 *
 * Reference: Section 3.2 of "Designing Hippocampal AI System.md"
 */

#include "vsa.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace bpagi {

class CA3Memory {
public:
    /**
     * Create CA3 memory with specified dimension.
     *
     * The fast weight matrix W is (dim x dim) but we use a sparse
     * representation since VSA vectors are binary.
     */
    explicit CA3Memory(size_t dimension = VSA::DEFAULT_DIM)
        : dimension_(dimension)
        , numBlocks_((dimension + 63) / 64)
    {
        clear();
    }

    // ========================================
    // Fast Weight Learning (Hebbian)
    // ========================================

    /**
     * Store an association: key → value
     *
     * Uses the Delta Rule (Widrow-Hoff):
     *   W += β * (v - W*k) ⊗ k^T
     *
     * Simplified for binary vectors:
     *   For each bit position, update association strength.
     *
     * @param key The cue pattern (what triggers retrieval)
     * @param value The target pattern (what to retrieve)
     * @param learningRate β - typically modulated by dopamine (0.0 to 1.0)
     */
    void store(const VSA::HyperVector& key,
               const VSA::HyperVector& value,
               float learningRate = 0.1f) {

        // For efficiency with binary vectors, we store associations
        // as a list of (key, value, strength) tuples rather than
        // a dense matrix.

        // Check if this key already exists
        for (auto& assoc : associations_) {
            if (hammingDistance(assoc.key, key) < dimension_ / 10) {
                // Similar key exists - update it (reconsolidation)
                assoc.value = value;
                assoc.strength = std::min(1.0f, assoc.strength + learningRate);
                return;
            }
        }

        // New association
        Association assoc;
        assoc.key = key;
        assoc.value = value;
        assoc.strength = learningRate;
        associations_.push_back(std::move(assoc));

        // Limit total associations (memory capacity)
        if (associations_.size() > maxAssociations_) {
            evictWeakest();
        }
    }

    /**
     * One-shot store with maximum strength.
     * Used for critical memories (high surprise/dopamine).
     */
    void storeOneShot(const VSA::HyperVector& key,
                      const VSA::HyperVector& value) {
        store(key, value, 1.0f);
    }

    // ========================================
    // Pattern Completion (Retrieval)
    // ========================================

    /**
     * Retrieve the value associated with a query.
     *
     * Implements pattern completion:
     *   - Partial/noisy query → complete stored pattern
     *
     * Returns the best matching value, or empty vector if no match.
     */
    VSA::HyperVector recall(const VSA::HyperVector& query,
                            float similarityThreshold = 0.3f) const {

        if (associations_.empty()) {
            return VSA::HyperVector(numBlocks_, 0);
        }

        // Find best matching key
        float bestSim = -1.0f;
        const Association* bestAssoc = nullptr;

        for (const auto& assoc : associations_) {
            float sim = binarySimilarity(query, assoc.key);
            // Weight by association strength
            sim *= assoc.strength;

            if (sim > bestSim) {
                bestSim = sim;
                bestAssoc = &assoc;
            }
        }

        if (bestAssoc && bestSim >= similarityThreshold) {
            return bestAssoc->value;
        }

        return VSA::HyperVector(numBlocks_, 0);  // No match
    }

    /**
     * Recall with iterative cleanup (attractor dynamics).
     *
     * Simulates CA3 recurrent dynamics by iteratively
     * retrieving and re-querying to "snap" to attractor.
     */
    VSA::HyperVector recallIterative(const VSA::HyperVector& query,
                                      int iterations = 3,
                                      float threshold = 0.3f) const {
        VSA::HyperVector current = query;

        for (int i = 0; i < iterations; i++) {
            VSA::HyperVector retrieved = recall(current, threshold);
            if (isZero(retrieved)) break;

            // Check for convergence
            if (binarySimilarity(current, retrieved) > 0.95f) {
                return retrieved;
            }
            current = retrieved;
        }

        return current;
    }

    /**
     * Query with confidence score.
     * Returns both the retrieved pattern and confidence (0-1).
     */
    std::pair<VSA::HyperVector, float> recallWithConfidence(
        const VSA::HyperVector& query) const {

        if (associations_.empty()) {
            return {VSA::HyperVector(numBlocks_, 0), 0.0f};
        }

        float bestSim = -1.0f;
        const Association* bestAssoc = nullptr;

        for (const auto& assoc : associations_) {
            float sim = binarySimilarity(query, assoc.key) * assoc.strength;
            if (sim > bestSim) {
                bestSim = sim;
                bestAssoc = &assoc;
            }
        }

        if (bestAssoc) {
            return {bestAssoc->value, bestSim};
        }

        return {VSA::HyperVector(numBlocks_, 0), 0.0f};
    }

    // ========================================
    // Memory Management
    // ========================================

    /**
     * Clear all stored associations.
     */
    void clear() {
        associations_.clear();
    }

    /**
     * Decay all association strengths (forgetting).
     */
    void decay(float amount = 0.01f) {
        for (auto& assoc : associations_) {
            assoc.strength = std::max(0.0f, assoc.strength - amount);
        }

        // Remove fully decayed associations
        associations_.erase(
            std::remove_if(associations_.begin(), associations_.end(),
                [](const Association& a) { return a.strength <= 0.0f; }),
            associations_.end()
        );
    }

    /**
     * Get number of stored associations.
     */
    size_t size() const { return associations_.size(); }

    /**
     * Check if memory is empty.
     */
    bool empty() const { return associations_.empty(); }

    /**
     * Set maximum number of associations.
     */
    void setCapacity(size_t maxAssoc) { maxAssociations_ = maxAssoc; }

private:
    struct Association {
        VSA::HyperVector key;
        VSA::HyperVector value;
        float strength;
    };

    size_t dimension_;
    size_t numBlocks_;
    size_t maxAssociations_ = 1000;
    std::vector<Association> associations_;

    /**
     * Binary cosine similarity.
     */
    float binarySimilarity(const VSA::HyperVector& a,
                           const VSA::HyperVector& b) const {
        size_t dist = hammingDistance(a, b);
        return 1.0f - 2.0f * static_cast<float>(dist) / dimension_;
    }

    /**
     * Hamming distance between binary vectors.
     */
    size_t hammingDistance(const VSA::HyperVector& a,
                           const VSA::HyperVector& b) const {
        size_t dist = 0;
        size_t blocks = std::min(a.size(), b.size());
        for (size_t i = 0; i < blocks; i++) {
            dist += __builtin_popcountll(a[i] ^ b[i]);
        }
        return dist;
    }

    /**
     * Check if vector is all zeros.
     */
    bool isZero(const VSA::HyperVector& v) const {
        for (uint64_t block : v) {
            if (block != 0) return false;
        }
        return true;
    }

    /**
     * Remove weakest association to make room.
     */
    void evictWeakest() {
        if (associations_.empty()) return;

        auto weakest = std::min_element(
            associations_.begin(), associations_.end(),
            [](const Association& a, const Association& b) {
                return a.strength < b.strength;
            }
        );

        associations_.erase(weakest);
    }
};

}  // namespace bpagi

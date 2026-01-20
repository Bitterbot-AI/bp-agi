#pragma once
/**
 * Hippocampus - Episodic Memory & Memory Consolidation
 *
 * Full implementation per "Designing Hippocampal AI System.md":
 *
 *   1. Entorhinal Cortex (EC) → Grid Cell encoding via VSA
 *   2. Dentate Gyrus (DG) → Pattern separation via hypervector hashing
 *   3. CA3 → Auto-associative memory with fast weights
 *   4. CA1 → Comparator (prediction vs reality → novelty detection)
 *
 * Key Insight from the paper:
 *   "The brain factorizes knowledge into Structure (Entorhinal) and
 *    Content (Sensory). By learning transitions between Grid Codes
 *    independently of Content, the system can generalize."
 *
 * This enables:
 *   - One-shot learning of input→output associations
 *   - Structural generalization (apply learned rule to new objects)
 *   - Pattern completion from partial cues
 *   - Prioritized replay based on surprise (dream consolidation)
 */

#include "vsa.hpp"
#include "grid_cells.hpp"
#include "ca3_memory.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <random>
#include <cmath>

namespace bpagi {

// ============================================
// Episode: A Single Memory
// ============================================
struct Episode {
    // Raw data (for replay to cortex)
    std::vector<uint8_t> inputRetina;   // The input pattern
    std::vector<uint8_t> targetRetina;  // The expected output
    int width;
    int height;

    // VSA encodings (for fast retrieval & generalization)
    VSA::HyperVector inputEncoding;   // Grid cell encoded scene
    VSA::HyperVector outputEncoding;  // Expected output encoding
    VSA::HyperVector transformRule;   // Learned transformation (unbind(out, in))

    // Metadata
    int surpriseLevel;       // Prediction error magnitude (0-100)
    int64_t timestamp;       // When this was experienced
    float confidence;        // How well the rule worked (0.0-1.0)
    uint64_t patternHash;    // Fast hash for deduplication

    Episode()
        : width(0), height(0), surpriseLevel(0), timestamp(0)
        , confidence(0.0f), patternHash(0) {}

    Episode(const std::vector<uint8_t>& input,
            const std::vector<uint8_t>& target,
            int surprise, int64_t ts)
        : inputRetina(input)
        , targetRetina(target)
        , width(0), height(0)
        , surpriseLevel(surprise)
        , timestamp(ts)
        , confidence(1.0f - surprise / 100.0f)
        , patternHash(0) {}
};

// ============================================
// CA1 Comparator - Novelty Detection
// ============================================
class CA1Comparator {
public:
    explicit CA1Comparator(VSA& vsa) : vsa_(vsa) {}

    /**
     * Compare prediction with reality.
     * Returns novelty score (0 = perfect match, 1 = completely different)
     */
    float compare(const VSA::HyperVector& prediction,
                  const VSA::HyperVector& reality) const {
        float sim = vsa_.similarity(prediction, reality);
        return (1.0f - sim) / 2.0f;
    }

    /**
     * Detect if input is novel (low similarity to any stored pattern).
     */
    bool isNovel(const VSA::HyperVector& input,
                 const std::vector<VSA::HyperVector>& knownPatterns,
                 float noveltyThreshold = 0.7f) const {
        for (const auto& known : knownPatterns) {
            if (vsa_.similarity(input, known) > (1.0f - noveltyThreshold)) {
                return false;
            }
        }
        return true;
    }

private:
    VSA& vsa_;
};

// ============================================
// Hippocampus: The Complete Memory System
// ============================================
class Hippocampus {
public:
    // Configuration
    static constexpr size_t MAX_EPISODES = 1000;
    static constexpr int MIN_SURPRISE_TO_STORE = 5;
    static constexpr int SIMILARITY_THRESHOLD = 90;  // % for dedup
    static constexpr size_t VSA_DIM = 4096;

    Hippocampus(uint32_t seed = 42)
        : vsa_(VSA_DIM, seed)
        , gridCells_(vsa_, seed + 1)
        , ca3_(VSA_DIM)
        , ca1_(vsa_)
        , rng_(seed + 2)
    {
        initializeValueVectors();
    }

    // ========================================
    // DENTATE GYRUS: Pattern Separation
    // ========================================

    uint64_t generateHash(const std::vector<uint8_t>& input);
    int hammingDistance(uint64_t a, uint64_t b);
    bool isSimilar(uint64_t hash1, uint64_t hash2);

    // ========================================
    // Main Interface
    // ========================================

    /**
     * Store a raw experience (basic API - no grid dimensions).
     * Used by Brain::captureEpisode().
     */
    void store(const std::vector<uint8_t>& input,
               const std::vector<uint8_t>& target,
               int surprise, int64_t timestamp) {

        if (surprise < MIN_SURPRISE_TO_STORE) return;

        // Infer dimensions (assume square if possible)
        int side = static_cast<int>(std::sqrt(input.size()));
        int width = (side * side == static_cast<int>(input.size())) ? side : input.size();
        int height = (side * side == static_cast<int>(input.size())) ? side : 1;

        experience(input, target, width, height, surprise, timestamp);
    }

    /**
     * Process a full experience with dimensions (preferred API).
     */
    void experience(const std::vector<uint8_t>& input,
                    const std::vector<uint8_t>& output,
                    int width, int height,
                    int surprise, int64_t timestamp);

    // ========================================
    // Retrieval & Prediction
    // ========================================

    const Episode* fetchForReplay();

    const Episode* fetchByIndex(size_t idx) const {
        if (idx >= episodes_.size()) return nullptr;
        return &episodes_[idx];
    }

    std::pair<VSA::HyperVector, float> predict(
        const std::vector<uint8_t>& input,
        int width, int height);

    VSA::HyperVector applyLearnedTransform(
        const std::vector<uint8_t>& input,
        int width, int height);

    VSA::HyperVector computeTransformation(
        const std::vector<uint8_t>& input1,
        const std::vector<uint8_t>& input2,
        int width, int height);

    std::vector<const Episode*> getReverseReplaySequence(size_t maxLen = 10);

    // ========================================
    // Novelty Assessment
    // ========================================

    float assessNovelty(const std::vector<uint8_t>& input,
                        int width, int height);

    // ========================================
    // Memory Management
    // ========================================

    void decay(int amount = 1);
    void reinforce(const Episode* ep, float amount = 0.1f);

    // ========================================
    // Accessors (inline)
    // ========================================

    size_t size() const { return episodes_.size(); }
    bool empty() const { return episodes_.empty(); }

    void clear() {
        episodes_.clear();
        ca3_.clear();
    }

    int getTotalSurprise() const {
        int total = 0;
        for (const auto& ep : episodes_) {
            total += ep.surpriseLevel;
        }
        return total;
    }

    const Episode* getMostSurprising() const {
        if (episodes_.empty()) return nullptr;
        return &*std::max_element(episodes_.begin(), episodes_.end(),
            [](const Episode& a, const Episode& b) {
                return a.surpriseLevel < b.surpriseLevel;
            });
    }

    // Component access for testing
    VSA& getVSA() { return vsa_; }
    GridCells& getGridCells() { return gridCells_; }
    CA3Memory& getCA3() { return ca3_; }

private:
    VSA vsa_;
    GridCells gridCells_;
    CA3Memory ca3_;
    CA1Comparator ca1_;
    std::mt19937 rng_;

    std::vector<Episode> episodes_;
    std::vector<VSA::HyperVector> valueVectors_;  // One per ARC color (0-9)

    void initializeValueVectors();
    VSA::HyperVector encodeScene(const std::vector<uint8_t>& grid, int width, int height);
    void evictOne();
};

}  // namespace bpagi

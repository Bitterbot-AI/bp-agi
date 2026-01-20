/**
 * Hippocampus Implementation
 *
 * Heavy methods moved from header for faster compilation.
 * See hippocampus.hpp for class documentation.
 */

#include "bpagi/hippocampus.hpp"
#include <algorithm>
#include <numeric>

namespace bpagi {

// ============================================
// Dentate Gyrus: Pattern Separation
// ============================================

uint64_t Hippocampus::generateHash(const std::vector<uint8_t>& input) {
    if (input.empty()) return 0;

    uint64_t sum = 0;
    for (uint8_t v : input) sum += v;
    uint8_t mean = static_cast<uint8_t>(sum / input.size());

    uint64_t hash = 0;
    size_t step = std::max(size_t(1), input.size() / 64);

    for (int i = 0; i < 64; i++) {
        size_t idx = (i * step) % input.size();
        if (input[idx] > mean) {
            hash |= (1ULL << i);
        }
    }

    // Non-linear mixing for orthogonalization
    hash ^= (hash >> 33);
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= (hash >> 33);
    hash *= 0xc4ceb9fe1a85ec53ULL;
    hash ^= (hash >> 33);

    return hash;
}

int Hippocampus::hammingDistance(uint64_t a, uint64_t b) {
    return __builtin_popcountll(a ^ b);
}

bool Hippocampus::isSimilar(uint64_t hash1, uint64_t hash2) {
    int dist = hammingDistance(hash1, hash2);
    int similarity = 100 - (dist * 100 / 64);
    return similarity >= SIMILARITY_THRESHOLD;
}

// ============================================
// Main Learning Interface
// ============================================

void Hippocampus::experience(const std::vector<uint8_t>& input,
                              const std::vector<uint8_t>& output,
                              int width, int height,
                              int surprise, int64_t timestamp) {

    if (surprise < MIN_SURPRISE_TO_STORE) return;

    // Pattern separation hash (Dentate Gyrus)
    uint64_t hash = generateHash(input);

    // Check for similar existing episode (dedup)
    for (auto& ep : episodes_) {
        if (isSimilar(ep.patternHash, hash)) {
            // Memory reconsolidation
            ep.surpriseLevel = std::max(ep.surpriseLevel, surprise);
            ep.timestamp = timestamp;
            return;
        }
    }

    // Encode scenes as hypervectors (Entorhinal Cortex)
    auto inputVec = encodeScene(input, width, height);
    auto outputVec = encodeScene(output, width, height);

    // Compute transformation rule
    auto transform = vsa_.bind(outputVec, inputVec);

    // Store in CA3 (fast associative memory)
    float learningRate = std::min(1.0f, surprise / 100.0f);
    ca3_.store(inputVec, outputVec, learningRate);
    ca3_.store(inputVec, transform, learningRate * 0.5f);

    // Create episode
    Episode ep;
    ep.inputRetina = input;
    ep.targetRetina = output;
    ep.width = width;
    ep.height = height;
    ep.inputEncoding = inputVec;
    ep.outputEncoding = outputVec;
    ep.transformRule = transform;
    ep.surpriseLevel = surprise;
    ep.timestamp = timestamp;
    ep.confidence = 1.0f - (surprise / 100.0f);
    ep.patternHash = hash;

    // Evict if at capacity
    if (episodes_.size() >= MAX_EPISODES) {
        evictOne();
    }

    episodes_.push_back(std::move(ep));
}

// ============================================
// Retrieval & Prediction
// ============================================

const Episode* Hippocampus::fetchForReplay() {
    if (episodes_.empty()) return nullptr;

    std::vector<double> weights;
    for (const auto& ep : episodes_) {
        double w = ep.surpriseLevel * ep.surpriseLevel + 1.0;
        weights.push_back(w);
    }

    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return &episodes_[dist(rng_)];
}

std::pair<VSA::HyperVector, float> Hippocampus::predict(
    const std::vector<uint8_t>& input,
    int width, int height) {

    auto inputVec = encodeScene(input, width, height);
    return ca3_.recallWithConfidence(inputVec);
}

VSA::HyperVector Hippocampus::applyLearnedTransform(
    const std::vector<uint8_t>& input,
    int width, int height) {

    auto inputVec = encodeScene(input, width, height);

    for (const auto& ep : episodes_) {
        float sim = vsa_.similarity(inputVec, ep.inputEncoding);
        if (sim > 0.5f) {
            return vsa_.bind(inputVec, ep.transformRule);
        }
    }
    return vsa_.zero();
}

VSA::HyperVector Hippocampus::computeTransformation(
    const std::vector<uint8_t>& input1,
    const std::vector<uint8_t>& input2,
    int width, int height) {

    auto vec1 = encodeScene(input1, width, height);
    auto vec2 = encodeScene(input2, width, height);
    return vsa_.unbind(vec2, vec1);
}

std::vector<const Episode*> Hippocampus::getReverseReplaySequence(size_t maxLen) {
    std::vector<const Episode*> sequence;

    std::vector<size_t> indices(episodes_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [this](size_t a, size_t b) {
            return episodes_[a].timestamp > episodes_[b].timestamp;
        });

    for (size_t i = 0; i < std::min(maxLen, indices.size()); i++) {
        sequence.push_back(&episodes_[indices[i]]);
    }
    return sequence;
}

// ============================================
// Novelty Assessment
// ============================================

float Hippocampus::assessNovelty(const std::vector<uint8_t>& input,
                                  int width, int height) {
    auto inputVec = encodeScene(input, width, height);

    float minDist = 1.0f;
    for (const auto& ep : episodes_) {
        float sim = vsa_.similarity(inputVec, ep.inputEncoding);
        float dist = (1.0f - sim) / 2.0f;
        minDist = std::min(minDist, dist);
    }
    return minDist;
}

// ============================================
// Memory Management
// ============================================

void Hippocampus::decay(int amount) {
    for (auto& ep : episodes_) {
        ep.surpriseLevel = std::max(0, ep.surpriseLevel - amount);
    }

    episodes_.erase(
        std::remove_if(episodes_.begin(), episodes_.end(),
            [](const Episode& ep) { return ep.surpriseLevel <= 0; }),
        episodes_.end()
    );

    ca3_.decay(0.01f);
}

void Hippocampus::reinforce(const Episode* ep, float amount) {
    for (auto& stored : episodes_) {
        if (&stored == ep) {
            stored.confidence = std::min(1.0f, stored.confidence + amount);
            stored.surpriseLevel = std::max(0, stored.surpriseLevel - 5);
            break;
        }
    }
}

void Hippocampus::evictOne() {
    if (episodes_.empty()) return;

    auto worst = std::min_element(episodes_.begin(), episodes_.end(),
        [](const Episode& a, const Episode& b) {
            float scoreA = a.surpriseLevel + a.confidence * 50;
            float scoreB = b.surpriseLevel + b.confidence * 50;
            return scoreA < scoreB;
        });

    episodes_.erase(worst);
}

VSA::HyperVector Hippocampus::encodeScene(const std::vector<uint8_t>& grid,
                                           int width, int height) {
    return gridCells_.encodeScene(grid, width, height, valueVectors_);
}

void Hippocampus::initializeValueVectors() {
    valueVectors_.resize(10);
    for (int i = 0; i < 10; i++) {
        valueVectors_[i] = vsa_.random();
    }
}

}  // namespace bpagi

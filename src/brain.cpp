#include "bpagi/brain.hpp"
#include <algorithm>
#include <functional>
#include <iostream>

namespace bpagi {

static UKS::Config makeUKSConfig(const Brain::Config& config) {
    UKS::Config uks;
    uks.numColumns = config.numColumns;
    uks.busWidth = config.busWidth;
    uks.recognitionThreshold = 12;
    uks.enableLearning = config.enableLearning;
    return uks;
}

Brain::Brain(const Config& config)
    : config_(config)
    , network_(200000, 2000000)  // Large capacity for all systems
    , vision_(network_)
    , uks_(network_, makeUKSConfig(config))
    , hasImage_(false)
    , prevAllocatedCount_(0)
    , lastAllocatedColumn_(std::nullopt)
    , didAllocate_(false)
    , ticksSincePresent_(0)
    , patternPresentedToUKS_(false)
{
    currentBusPattern_.reserve(config.busWidth);
    currentImage_.resize(RETINA_SIZE, 0);
}

// ========================================
// Main Interface
// ========================================

void Brain::present(const std::vector<uint8_t>& image) {
    // Store image for continuous presentation
    if (image.size() == RETINA_SIZE) {
        currentImage_ = image;
        hasImage_ = true;
        // Clear accumulated pattern for new image
        accumulatedBusPattern_.clear();
        // Reset stabilization counter - wait for features to accumulate
        ticksSincePresent_ = 0;
        // Allow new presentation to UKS
        patternPresentedToUKS_ = false;
    }
    vision_.present(image);
}

std::optional<uint32_t> Brain::step() {
    // Track allocation state before step
    prevAllocatedCount_ = uks_.getAllocatedCount();
    didAllocate_ = false;

    // Continuously re-present the image (like sustained visual input)
    // This keeps retina neurons firing as long as the image is shown
    if (hasImage_) {
        vision_.present(currentImage_);
    }

    // Step 1: Vision processing (before network step so features are fresh)
    vision_.step();

    // Step 2: Update the Axon Bundle (read boundary activity)
    updateBusPattern();

    // Step 3: Present the visual pattern to UKS (inject charge into bus)
    // IMPORTANT: Wait for pattern stabilization before presenting to UKS!
    // Charge must be injected BEFORE network_.step() so bus neurons can fire.
    //
    // CONTINUOUS PRESENTATION: Inject charge every tick after stabilization.
    // This is critical for the WTA timing fix:
    // - Request neuron needs sustained input to accumulate charge
    // - Column recognition inhibits Request before it can fire
    // - Without continuous input, Request either fires too fast or never
    //
    ticksSincePresent_++;
    if (ticksSincePresent_ >= STABILIZATION_TICKS && !currentBusPattern_.empty()) {
        // PHASE 20: ACETYLCHOLINE INPUT GATING
        // ACh modulates the strength of sensory (external) input:
        //   High ACh (100): 1.5x input strength - strong encoding mode
        //   Normal ACh (50): 1.0x input strength - baseline
        //   Low ACh (0): 0.5x input strength - weak sensory, internal mode
        //
        // Blueprint: "ACh promotes feedforward processing (sensory input → cortex)
        //             while suppressing feedback (internal predictions)"
        //
        // This allows dream() to work properly: low ACh means internal
        // recurrent patterns dominate over weak sensory injection.

        int8_t achLevel = network_.chemicals().acetylcholine;
        // Map ACh 0-100 to charge multiplier 5-15 (0.5x to 1.5x of baseline 10)
        Charge sensoryCharge = 5 + (achLevel / 10);  // 5-15 based on ACh

        // First time: call present() to set up currentInput_ for learning
        if (!patternPresentedToUKS_) {
            uks_.present(currentBusPattern_);
            patternPresentedToUKS_ = true;
        } else {
            // Subsequent ticks: inject ACh-modulated charge
            // This provides sustained input scaled by attention level
            const auto& busNeurons = uks_.getBusNeurons();
            for (NeuronId idx : currentBusPattern_) {
                if (idx < busNeurons.size()) {
                    network_.injectCharge(busNeurons[idx], sensoryCharge);
                }
            }
        }
    }

    // Step 4: Advance the network (propagate spikes)
    // This fires bus neurons that got charge from uks_.present()
    network_.step();

    // Step 5: Step UKS (WTA selection, learning)
    uks_.step();

    // Track new allocations
    size_t newAllocatedCount = uks_.getAllocatedCount();
    if (newAllocatedCount > prevAllocatedCount_) {
        didAllocate_ = true;
        // Find the newly allocated column by counting
        const auto& columns = uks_.getColumns();
        size_t count = 0;
        for (uint32_t i = 0; i < columns.size(); i++) {
            if (columns[i].isAllocated) {
                count++;
                // The Nth allocated column is at index i when count == N
                if (count == newAllocatedCount) {
                    lastAllocatedColumn_ = i;
                    break;
                }
            }
        }
    }

    return uks_.getActiveColumn();
}

std::optional<uint32_t> Brain::run(size_t ticks) {
    std::optional<uint32_t> result = std::nullopt;

    for (size_t t = 0; t < ticks; t++) {
        auto active = step();
        if (active.has_value()) {
            result = active;
        }
    }

    return result;
}

void Brain::reset() {
    std::cerr << "    [Brain::reset] network_.reset()..." << std::flush;
    network_.reset();
    std::cerr << " done" << std::endl;
    std::cerr << "    [Brain::reset] vision_.reset()..." << std::flush;
    vision_.reset();
    std::cerr << " done" << std::endl;
    // Note: Don't reset UKS - we want to keep learned columns!
    std::cerr << "    [Brain::reset] clear patterns..." << std::flush;
    currentBusPattern_.clear();
    accumulatedBusPattern_.clear();
    std::cerr << " done" << std::endl;
    // Clear the current image
    std::cerr << "    [Brain::reset] fill currentImage_ (size=" << currentImage_.size() << ")..." << std::flush;
    std::fill(currentImage_.begin(), currentImage_.end(), 0);
    std::cerr << " done" << std::endl;
    hasImage_ = false;
    prevAllocatedCount_ = 0;
    lastAllocatedColumn_ = std::nullopt;
    didAllocate_ = false;
    ticksSincePresent_ = 0;
    patternPresentedToUKS_ = false;
}

void Brain::resetShortTermMemory() {
    // Reset neural activations but KEEP learned synaptic weights
    // This is used between test cases to preserve learned patterns

    // Reset network activations (charges) but not weights
    network_.reset();

    // Clear vision working memory
    vision_.reset();

    // Clear bus patterns
    currentBusPattern_.clear();
    accumulatedBusPattern_.clear();

    // Clear current image
    std::fill(currentImage_.begin(), currentImage_.end(), 0);
    hasImage_ = false;

    // Reset tracking
    prevAllocatedCount_ = uks_.getAllocatedCount();  // Keep allocated count!
    lastAllocatedColumn_ = std::nullopt;
    didAllocate_ = false;
    ticksSincePresent_ = 0;
    patternPresentedToUKS_ = false;
}

void Brain::injectDopamine(int amount) {
    // The "Save Button" - boost dopamine to enable/strengthen STDP learning
    network_.chemicals().dopamine = static_cast<int8_t>(std::min(std::max(amount, 0), 100));
}

void Brain::injectNoise(int amplitude) {
    // Stochastic Resonance: Inject random noise to shake system out of local minima
    // This is used for "creative" second guesses in the submission generator

    // Use simple PRNG (deterministic for reproducibility within a run)
    static uint32_t seed = 12345;

    const size_t numNeurons = network_.getNeuronCount();
    for (size_t i = 0; i < numNeurons; i++) {
        // Linear congruential generator
        seed = seed * 1103515245 + 12345;
        int noise = static_cast<int>((seed >> 16) & 0x7FFF) % (2 * amplitude + 1) - amplitude;
        network_.injectCharge(static_cast<NeuronId>(i), static_cast<Charge>(noise));
    }
}

void Brain::injectNoiseToHidden(int amplitude) {
    // Tonic Norepinephrine: Inject noise to hidden layers only
    // Skips retina neurons so input signal isn't corrupted
    // Based on Aston-Jones & Cohen's Adaptive Gain Theory:
    // - Sustained (tonic) NE promotes exploration mode
    // - Noise in hidden layers is more powerful than input noise (PMC2771718)

    static uint32_t seed = 54321;  // Different seed from injectNoise

    // Build set of retina neuron IDs to skip
    const auto& retinaNeurons = vision_.getRetinaNeurons();
    std::unordered_set<NeuronId> retinaSet(retinaNeurons.begin(), retinaNeurons.end());

    const size_t numNeurons = network_.getNeuronCount();
    for (size_t i = 0; i < numNeurons; i++) {
        // Skip retina neurons - don't corrupt the input signal
        if (retinaSet.count(static_cast<NeuronId>(i))) {
            continue;
        }

        // Linear congruential generator
        seed = seed * 1103515245 + 12345;
        int noise = static_cast<int>((seed >> 16) & 0x7FFF) % (2 * amplitude + 1) - amplitude;
        network_.injectCharge(static_cast<NeuronId>(i), static_cast<Charge>(noise));
    }
}

// ========================================
// Query Interface
// ========================================

std::optional<uint32_t> Brain::getActiveColumn() const {
    return uks_.getActiveColumn();
}

bool Brain::didRequestFire() const {
    return uks_.didRequestFire();
}

bool Brain::didAllocate() const {
    return didAllocate_;
}

std::optional<uint32_t> Brain::getLastAllocatedColumn() const {
    return lastAllocatedColumn_;
}

size_t Brain::getAllocatedCount() const {
    return uks_.getAllocatedCount();
}

size_t Brain::getActiveBoundaryCount() const {
    return vision_.getActiveBoundaryCount();
}

// ========================================
// Axon Bundle Implementation (Relational Hashing)
// ========================================

void Brain::updateBusPattern() {
    // RELATIONAL HASHING: Count features by type, not position
    // This achieves TRANSLATION INVARIANCE - a square is recognized
    // regardless of where it appears in the visual field.
    //
    // Bus layout (64 indices):
    //   0-7:   Orthogonal Corners (4 types × 2 bins) - SQUARE signature
    //   8-15:  Acute Vertices (2 types × 4 bins) - TRIANGLE signature
    //   16-63: Boundary counts (4 types × 12 bins)
    //
    // KEY INSIGHT for Shape Discrimination:
    //   Square:   HIGH corners (4), ZERO acute vertices
    //   Triangle: LOW corners (~2), HIGH acute vertices (1+ at apex)
    //
    // This creates near-zero overlap between Square and Triangle!

    // NOTE: Do NOT clear accumulatedBusPattern_ here!
    // Features fire on different ticks due to the neural cascade:
    //   Tick 1: Boundaries fire
    //   Tick 2: Corners & Acute Vertices fire (need boundaries from tick 1)
    // The clearing happens in present() when a NEW image arrives.
    // This allows features to accumulate across the cascade delay.

    // ========================================
    // Orthogonal Corner Encoding (Bus 0-7)
    // ========================================
    // Squares have 4 orthogonal corners (one of each type)
    // Triangles have 0-2 orthogonal corners (at base only)

    for (size_t t = 0; t < NUM_CORNER_TYPES; t++) {
        CornerType type = static_cast<CornerType>(t);
        size_t count = vision_.countCornersByType(type);

        // 2 bins per corner type
        size_t baseIdx = t * 2;  // 0, 2, 4, 6

        if (count > 0) accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx));
        if (count > 5) accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 1));
    }

    // ========================================
    // Acute Vertex Encoding (Bus 8-15) - THE KEY DISCRIMINATOR!
    // ========================================
    // Triangles have acute vertices (where / meets \)
    // Squares have ZERO acute vertices (only 90-degree corners)
    //
    // CRITICAL: Use PRESENCE encoding, not count encoding!
    // If ANY acute vertex is detected, activate ALL 4 bins for that type.
    // This gives Triangle 8 unique indices vs Square's 0.
    //
    // This is the "new word" that distinguishes:
    //   Square = "Thing with 90-degree corners" (indices 8-15: NONE)
    //   Triangle = "Thing with acute vertices" (indices 8-15: ALL)

    for (size_t t = 0; t < NUM_ACUTE_VERTEX_TYPES; t++) {
        AcuteVertexType type = static_cast<AcuteVertexType>(t);
        size_t count = vision_.countAcuteVerticesByType(type);

        // 4 bins per acute vertex type - PRESENCE ENCODING
        size_t baseIdx = 8 + t * 4;  // 8, 12

        // If ANY acute vertex detected, activate ALL bins for this type
        // This maximizes discrimination signal
        if (count > 0) {
            accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx));
            accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 1));
            accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 2));
            accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 3));
        }
    }

    // ========================================
    // Boundary Feature Encoding (Bus 16-63)
    // ========================================
    // Edge orientations provide additional discrimination

    for (size_t t = 0; t < NUM_BOUNDARY_TYPES; t++) {
        BoundaryType type = static_cast<BoundaryType>(t);
        size_t count = vision_.countBoundariesByType(type);

        // 12 bins per boundary type
        size_t baseIdx = 16 + t * 12;  // 16, 28, 40, 52

        if (count > 0)   accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx));
        if (count > 15)  accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 1));
        if (count > 30)  accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 2));
        if (count > 40)  accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 3));
        if (count > 50)  accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 4));
        if (count > 60)  accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 5));
        if (count > 75)  accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 6));
        if (count > 100) accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 7));
        if (count > 130) accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 8));
        if (count > 170) accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 9));
        if (count > 220) accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 10));
        if (count > 280) accumulatedBusPattern_.insert(static_cast<NeuronId>(baseIdx + 11));
    }

    // Convert accumulated set to sorted vector for presentation
    currentBusPattern_.assign(accumulatedBusPattern_.begin(), accumulatedBusPattern_.end());
    std::sort(currentBusPattern_.begin(), currentBusPattern_.end());
}

NeuronId Brain::hashToBusIndex(size_t x, size_t y, BoundaryType type) const {
    // DEPRECATED: This spatial hash is no longer used.
    // Kept for API compatibility but relational hashing in updateBusPattern()
    // now provides translation-invariant encoding.
    //
    // Old behavior preserved for reference:
    size_t superX = x / 8;
    size_t superY = y / 8;
    NeuronId baseIdx = static_cast<NeuronId>(superY * 8 + superX);
    size_t orientationOffset = static_cast<size_t>(type) * 16;
    return (baseIdx + orientationOffset) % config_.busWidth;
}

// ========================================
// Hippocampus Interface (Phase 18)
// ========================================

void Brain::captureEpisode(const std::vector<uint8_t>& input,
                           const std::vector<uint8_t>& target,
                           int surprise) {
    // Store input/output pair in episodic memory
    // Called after prediction error (failed task)
    int64_t timestamp = network_.getCurrentTick();
    hippocampus_.store(input, target, surprise, timestamp);
}

void Brain::dream(int episodes, int ticksPerEpisode, int dopamineLevel) {
    /**
     * Dream: The Secret Weapon for Memory Consolidation
     *
     * Neuroscience: During REM sleep, the hippocampus replays experiences
     * to the cortex, strengthening important memories. High-dopamine events
     * (surprising/rewarding) are replayed more frequently.
     *
     * NEUROMODULATION (Phase 20):
     *   - LOW ACh: Suppresses external input, boosts internal recurrent loops
     *     This allows the cortex to "imagine" without sensory interference
     *   - HIGH DA: Strengthens associations being replayed
     *   - LOW NE: Calm state, no panic interrupts during consolidation
     *
     * Implementation:
     *   1. Set ACh LOW (internal consolidation mode)
     *   2. Fetch high-surprise episode from hippocampus
     *   3. Inject input pattern into vision system
     *   4. Wait for propagation through cortex
     *   5. Force target pattern onto output (supervised signal)
     *   6. Massive dopamine (STDP: "REMEMBER THIS!")
     *   7. Repeat for consolidation
     *   8. Restore ACh to baseline
     *
     * Result: 1 failed task → 1000s of training iterations → mastery
     */

    if (hippocampus_.empty()) {
        return;  // Nothing to dream about
    }

    // Save current neuromodulator state
    bool wasLearning = network_.isPlasticityEnabled();
    int8_t savedACh = network_.chemicals().acetylcholine;
    int8_t savedNE = network_.chemicals().norepinephrine;

    // Enter dream state:
    // - Low ACh = internal/recurrent mode (not attending to external input)
    // - Low NE = calm, no panic resets
    // - Plasticity enabled for learning
    network_.setPlasticityEnabled(true);
    network_.chemicals().acetylcholine = 20;   // Low = retrieval/consolidation
    network_.chemicals().norepinephrine = 20;  // Low = calm, stable

    for (int ep = 0; ep < episodes; ep++) {
        // Fetch a high-surprise memory (biased toward failures)
        const Episode* memory = hippocampus_.fetchForReplay();
        if (!memory) continue;

        // ========================================
        // Phase 1: Present Input (5 ticks)
        // ========================================
        // "Remember seeing this..."
        resetShortTermMemory();
        present(memory->inputRetina);

        for (int t = 0; t < ticksPerEpisode / 2; t++) {
            step();
        }

        // ========================================
        // Phase 2: Force Target + Dopamine (5 ticks)
        // ========================================
        // "...and it looked like THIS!"
        // Inject massive dopamine: "THIS IS IMPORTANT, REMEMBER IT!"
        injectDopamine(dopamineLevel);

        // Force target pattern onto the visual system
        // This creates the A→B association in the cortex
        present(memory->targetRetina);

        for (int t = 0; t < ticksPerEpisode / 2; t++) {
            step();
        }

        // ========================================
        // Phase 3: Consolidation pause
        // ========================================
        // Let STDP stabilize the new weights
        injectDopamine(dopamineLevel / 2);
        for (int t = 0; t < 3; t++) {
            step();
        }
    }

    // Decay old memories (forgetting curve)
    // This makes room for new important memories
    hippocampus_.decay(1);

    // Restore neuromodulator state (wake up from dream)
    network_.setPlasticityEnabled(wasLearning);
    network_.chemicals().acetylcholine = savedACh;
    network_.chemicals().norepinephrine = savedNE;
}

}  // namespace bpagi

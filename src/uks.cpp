#include "bpagi/uks.hpp"
#include <algorithm>
#include <limits>

namespace bpagi {

UKS::UKS(Network& network, const Config& config)
    : network_(network)
    , config_(config)
    , requestNeuron_(INVALID_NEURON)
    , globalInhibitor_(INVALID_NEURON)
    , activeColumn_(std::nullopt)
    , requestFired_(false)
    , totalAllocations_(0)
    , totalRecognitions_(0)
{
    // Build the neural infrastructure
    buildBus();
    buildColumns();
    buildWTACircuit();
}

void UKS::buildBus() {
    // Create Recognition Bus neurons
    // These receive external input and broadcast to all columns
    busNeurons_.reserve(config_.busWidth);

    for (size_t i = 0; i < config_.busWidth; i++) {
        // Bus neurons have low threshold (easy to activate from external input)
        // and short refractory (can fire rapidly)
        NeuronId busNeuron = network_.addNeuron(2, 0, 1);  // threshold=2, no leak, refractory=1
        busNeurons_.push_back(busNeuron);
    }
}

void UKS::buildColumns() {
    columns_.resize(config_.numColumns);

    for (size_t col = 0; col < config_.numColumns; col++) {
        std::vector<NeuronId> inputs;
        std::vector<NeuronId> pyramidals;

        // Create input neurons for this column
        // These receive from the Recognition Bus
        for (size_t i = 0; i < COLUMN_INPUT_NEURONS; i++) {
            NeuronId n = network_.addNeuron(COLUMN_INPUT_THRESHOLD, 1, 2);
            inputs.push_back(n);
        }

        // Create pyramidal neurons (main processing)
        for (size_t i = 0; i < COLUMN_PYRAMIDAL_NEURONS; i++) {
            NeuronId n = network_.addNeuron(COLUMN_PYRAMIDAL_THRESHOLD, 1, 3);
            pyramidals.push_back(n);
        }

        // Create output neuron (column "fires" when this activates)
        // High threshold ensures it needs many pyramidals to fire
        // Short refractory (2) allows frequent firing to keep suppressing Request
        NeuronId output = network_.addNeuron(COLUMN_OUTPUT_THRESHOLD, 0, 2);

        // Create inhibitory interneuron (local inhibition)
        NeuronId inhibitory = network_.addNeuron(COLUMN_INHIBITORY_THRESHOLD, 0, 2);

        // Initialize column struct
        columns_[col].initialize(static_cast<uint32_t>(col), inputs, pyramidals, output, inhibitory);

        // Wire internal column circuitry:
        // Input -> Pyramidal (feedforward) - STRONG connections
        for (NeuronId inp : inputs) {
            for (NeuronId pyr : pyramidals) {
                // Dense internal connectivity (~40%) with strong weights
                if ((inp * 7 + pyr * 13) % 5 < 2) {
                    network_.connectNeurons(inp, pyr, 5, false);  // Strong excitation
                }
            }
        }

        // Pyramidal -> Output (convergence) - each pyramidal contributes
        for (NeuronId pyr : pyramidals) {
            network_.connectNeurons(pyr, output, 1, false);  // Many weak = strong together
        }

        // Pyramidal -> Inhibitory (feedback)
        for (NeuronId pyr : pyramidals) {
            network_.connectNeurons(pyr, inhibitory, 1, false);
        }

        // Inhibitory -> Pyramidal (lateral inhibition within column)
        // Keeps activity sparse within the column
        for (NeuronId pyr : pyramidals) {
            network_.connectNeurons(inhibitory, pyr, -2, false);
        }

        // ========================================
        // RECURRENT SYNAPSES: The Reverberation Loop
        // ========================================
        // Phase 6: Object Permanence (Short-Term Memory)
        //
        // Physics: When the column is active, pyramidal neurons feed charge
        // back into each other. When input stops:
        //   - Leak drains charge (-1 per tick)
        //   - Recurrent synapses add charge (+1 when neighbor fires)
        //   - Net effect: Charge decays slowly, keeping column "primed"
        //
        // This creates a "memory trace" - the column remembers recent activity
        // and can respond faster when the same input returns.
        //
        // TUNING: With 50 pyramidals and 10% connectivity:
        //   - Each neuron receives from ~5 others
        //   - When those 5 fire: 5 × 1 = 5 charge
        //   - Threshold is 5, leak is 1
        //   - Net: 5 - 1 = 4 (below threshold without external input)
        //   - This creates DECAYING reverberation, not sustained
        //
        // Connectivity: Sparse (~10%) to control excitation
        // Weight: +1 (balance with leak to create decay)
        //
        for (size_t i = 0; i < pyramidals.size(); i++) {
            for (size_t j = 0; j < pyramidals.size(); j++) {
                if (i != j) {  // No self-connections
                    // Sparse connectivity (~10%) using deterministic hash
                    if ((pyramidals[i] * 11 + pyramidals[j] * 17) % 10 == 0) {
                        network_.connectNeurons(pyramidals[i], pyramidals[j], 1, false);
                    }
                }
            }
        }
    }
}

void UKS::buildWTACircuit() {
    // Request Neuron: Fires when NO column recognizes the input
    //
    // TIMING FIX: Request must NOT fire before columns have time to respond!
    // Column cascade: bus -> input -> pyramidal -> output takes ~3-4 ticks
    //
    // Strategy:
    // - High threshold (130): Single bus burst (56×1=56) won't trigger
    // - Low weight (+1): Slow accumulation requires sustained input
    // - Strong column inhibition (4×-16=-64): Recognition suppresses Request
    //
    // Math for novel pattern (no column responds):
    //   Tick 8: charge=56 < 130 (doesn't fire)
    //   Tick 9: 56+56-3=109 < 130 (doesn't fire)
    //   Tick 10: 109+56-3=162 > 130 (fires! → allocate)
    //
    // Math for known pattern (column responds at tick 10):
    //   Tick 10: 109+56-3-64=98 < 130 (doesn't fire → no double allocation)
    //
    requestNeuron_ = network_.addNeuron(130, 3, 25);  // threshold=130, leak=3, long refractory=25

    // Global Inhibitor: Enforces WTA across all columns
    globalInhibitor_ = network_.addNeuron(3, 0, 2);

    // Wire Request Neuron:
    // - Receives WEAK excitation from bus (+1 each, slow accumulation)
    // - Receives STRONG inhibition from column outputs (4 synapses × -16 = -64)
    for (NeuronId bus : busNeurons_) {
        network_.connectNeurons(bus, requestNeuron_, 1, false);  // Weak excitation
    }

    for (const auto& col : columns_) {
        // Column output inhibits Request Neuron with MULTIPLE synapses
        // Total inhibition per column: 4 × -16 = -64
        // This ensures recognition definitively suppresses Request
        network_.connectNeurons(col.outputNeuron, requestNeuron_, WEIGHT_MIN, false);
        network_.connectNeurons(col.outputNeuron, requestNeuron_, WEIGHT_MIN, false);
        network_.connectNeurons(col.outputNeuron, requestNeuron_, WEIGHT_MIN, false);
        network_.connectNeurons(col.outputNeuron, requestNeuron_, WEIGHT_MIN, false);
    }

    // ========================================
    // SOFT WTA: Lateral Inhibition Between Columns
    // ========================================
    // The "Best Fit" column must suppress "Good Fit" columns
    // This ensures only ONE column wins even with similar patterns
    //
    // Two-tier inhibition:
    // 1. Global Inhibitor: Fast, broad suppression
    // 2. Direct Lateral: Column-to-column suppression (stronger)

    // Global Inhibitor: Fast broadcast inhibition
    for (const auto& col : columns_) {
        network_.connectNeurons(col.outputNeuron, globalInhibitor_, 4, false);  // Stronger excitation
    }
    for (const auto& col : columns_) {
        network_.connectNeurons(globalInhibitor_, col.outputNeuron, -10, false);  // Stronger inhibition
    }

    // DIRECT LATERAL INHIBITION: Each column output inhibits OTHER columns' outputs
    // This creates true competition where the strongest signal wins
    // The column that fires first (strongest match) suppresses all others
    for (size_t i = 0; i < columns_.size(); i++) {
        for (size_t j = 0; j < columns_.size(); j++) {
            if (i != j) {
                // Column i's output inhibits Column j's output
                // Strong inhibition ensures winner suppresses runners-up
                network_.connectNeurons(columns_[i].outputNeuron,
                                       columns_[j].outputNeuron,
                                       -6, false);  // Direct lateral inhibition
            }
        }
    }
}

std::optional<uint32_t> UKS::present(const std::vector<NeuronId>& inputPattern) {
    currentInput_ = inputPattern;
    activeColumn_ = std::nullopt;
    requestFired_ = false;

    // Activate bus neurons corresponding to input pattern
    // Only inject charge - let them fire naturally via threshold crossing
    // This avoids the refractory period issue with injectSpike
    for (NeuronId inputId : inputPattern) {
        if (inputId < busNeurons_.size()) {
            NeuronId busNeuron = busNeurons_[inputId];
            network_.injectCharge(busNeuron, 10);  // Well above threshold (2)
        }
    }

    return activeColumn_;  // Will be set after step()
}

void UKS::step() {
    // Check which columns fired
    std::vector<uint32_t> responding = getRespondingColumns();

    // ========================================
    // HOMEOSTATIC LOOP: The Chemical Driver
    // ========================================
    // The UKS regulates neuromodulator levels based on events.
    // This creates dynamic brain states (alert, calm, learning).

    bool anyActivity = !responding.empty();
    bool surpriseEvent = false;

    if (!responding.empty()) {
        // Recognition occurred!
        // WTA ensures only one should fire, but take first if multiple
        activeColumn_ = responding[0];
        columns_[*activeColumn_].isActive = true;
        columns_[*activeColumn_].activationCount++;
        totalRecognitions_++;

        // Recognition suppresses learning
        requestFired_ = false;

        // Recognition is mildly rewarding (predictable world)
        network_.chemicals().spikeDopamine(10);
    } else {
        // No recognition yet - check if Request Neuron fired
        requestFired_ = network_.didFire(requestNeuron_);

        if (requestFired_) {
            // ========================================
            // SURPRISE EVENT: Novel input detected!
            // ========================================
            // The Request Neuron fired = something new was seen.
            // Spike Norepinephrine (+50): "Wake up! Pay attention!"
            // This lowers thresholds, making the system more responsive.
            network_.surpriseSignal(50);
            surpriseEvent = true;

            // Also boost Acetylcholine: "Focus on external input!"
            network_.chemicals().spikeAcetylcholine(30);

            if (config_.enableLearning && !currentInput_.empty()) {
                // Novel input! Allocate a new column
                auto freeColumn = findFreeColumn();
                if (freeColumn.has_value()) {
                    allocateColumn(*freeColumn, currentInput_);
                    activeColumn_ = freeColumn;

                    // Learning moment: Spike Dopamine (+30)
                    // "This is worth remembering!"
                    network_.chemicals().spikeDopamine(30);

                    // Clear input to prevent re-allocation on subsequent ticks
                    currentInput_.clear();
                }
            }
        }
    }

    // ========================================
    // TIMEOUT: No activity = calm down
    // ========================================
    // If nothing is happening, increase Serotonin.
    // This makes the system more stable and patient.
    if (!anyActivity && !surpriseEvent) {
        // Spike Serotonin (+5): "Nothing happening, chill out"
        network_.calmSignal(5);

        // Decrease Acetylcholine: "Okay to focus on internal thoughts"
        if (network_.chemicals().acetylcholine > 30) {
            network_.chemicals().acetylcholine -= 2;
        }
    }

    // Update active flags
    for (auto& col : columns_) {
        col.isActive = col.checkActive(network_);
    }
}

void UKS::reset() {
    for (auto& col : columns_) {
        col.reset();
    }
    currentInput_.clear();
    activeColumn_ = std::nullopt;
    requestFired_ = false;
    totalAllocations_ = 0;
    totalRecognitions_ = 0;
}

std::optional<uint32_t> UKS::getActiveColumn() const {
    return activeColumn_;
}

bool UKS::didRequestFire() const {
    return requestFired_;
}

size_t UKS::getAllocatedCount() const {
    size_t count = 0;
    for (const auto& col : columns_) {
        if (col.isAllocated) count++;
    }
    return count;
}

size_t UKS::getFreeCount() const {
    return columns_.size() - getAllocatedCount();
}

const CorticalColumn& UKS::getColumn(uint32_t id) const {
    return columns_.at(id);
}

std::vector<uint32_t> UKS::getRespondingColumns() const {
    std::vector<uint32_t> responding;
    for (size_t i = 0; i < columns_.size(); i++) {
        if (columns_[i].isAllocated && network_.didFire(columns_[i].outputNeuron)) {
            responding.push_back(static_cast<uint32_t>(i));
        }
    }
    return responding;
}

std::optional<uint32_t> UKS::findFreeColumn() const {
    for (size_t i = 0; i < columns_.size(); i++) {
        if (!columns_[i].isAllocated) {
            return static_cast<uint32_t>(i);
        }
    }
    return std::nullopt;  // All columns allocated
}

void UKS::allocateColumn(uint32_t columnId, const std::vector<NeuronId>& pattern) {
    CorticalColumn& col = columns_[columnId];

    // Mark as allocated
    col.isAllocated = true;
    col.allocatedAtTick = network_.getCurrentTick();
    totalAllocations_++;

    // ONE-SHOT LEARNING: Wire the input pattern to this column's input neurons
    // This is the critical "instant learning" step - no gradual weight updates!
    //
    // Strategy:
    // 1. Bus neurons IN the pattern → EXCITE input neurons (strong positive weight)
    // 2. Bus neurons NOT in pattern → INHIBIT input neurons (mismatch penalty)
    //
    // This creates a "template match" where the column only fires strongly
    // when the input pattern closely matches the learned pattern.

    // Create a set for fast lookup of pattern membership
    std::unordered_set<NeuronId> patternSet(pattern.begin(), pattern.end());

    for (size_t busIdx = 0; busIdx < busNeurons_.size(); busIdx++) {
        NeuronId busNeuron = busNeurons_[busIdx];
        bool isInPattern = (patternSet.find(static_cast<NeuronId>(busIdx)) != patternSet.end());

        if (isInPattern) {
            // Pattern member: weak excitation (+1)
            // With ~56 indices in pattern, full match gives 56*1 = 56 charge
            // Must overcome mismatches: 4 mismatches * 16 = 64 inhibition
            // Net for mismatch: 48*1 - 4*16 = -16 → below threshold!
            // Net for match: 56*1 = 56 → above threshold!
            for (NeuronId inputNeuron : col.inputNeurons) {
                network_.connectNeurons(busNeuron, inputNeuron, 1, false);
            }
        } else {
            // Non-pattern member: MAXIMUM inhibition (-16)
            // This is the key to discrimination!
            // With 8 mismatches: 8*16 = 128 inhibition
            // With 48 matches: 48*2 = 96 excitation
            // Net: -32 → below threshold → column doesn't fire!
            //
            // For correct match (56 matches, 0 mismatches): 112 → fires!
            for (NeuronId inputNeuron : col.inputNeurons) {
                network_.connectNeurons(busNeuron, inputNeuron, WEIGHT_MIN, false);
            }
        }
    }

    // Also suppress other free columns to prevent double-allocation
    // (This is the WTA "suppressive gate")
    suppressOthers(columnId);
}

void UKS::suppressOthers(uint32_t winnerId) {
    // Inject inhibition to all other free columns
    // This prevents them from being allocated on the same tick
    for (size_t i = 0; i < columns_.size(); i++) {
        if (i != winnerId && !columns_[i].isAllocated) {
            // Inject negative charge (inhibition) to their output neurons
            network_.injectCharge(columns_[i].outputNeuron, -10);
        }
    }
}

}  // namespace bpagi

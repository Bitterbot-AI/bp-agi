#pragma once
#include <cstdint>
#include <vector>
#include <limits>

namespace bpagi {

// Integer-only types per Blueprint specification
// All arithmetic must use integer operations - no floating point

using NeuronId = uint32_t;
using Tick = int64_t;
using Charge = int32_t;
using Weight = int8_t;  // Range: -16 to +16 (4-bit effective)

// Weight bounds as specified in Blueprint
constexpr Weight WEIGHT_MIN = -16;
constexpr Weight WEIGHT_MAX = +16;

// Charge bounds
constexpr Charge CHARGE_MIN = 0;
constexpr Charge CHARGE_MAX = std::numeric_limits<Charge>::max();

// Invalid neuron ID sentinel
constexpr NeuronId INVALID_NEURON = std::numeric_limits<NeuronId>::max();

// Default neuron parameters
constexpr Charge DEFAULT_THRESHOLD = 10;
constexpr Charge DEFAULT_LEAK = 1;
constexpr int32_t DEFAULT_REFRACTORY = 5;

// STDP timing window (in ticks)
constexpr Tick STDP_WINDOW = 20;

// ========================================
// Neuromodulation System (The Chemical Layer)
// ========================================
// These are the "Quad-Core" control signals that regulate brain state:
//   DA  - Dopamine:       Learning gate ("Save Button")
//   NE  - Norepinephrine: Threshold gain ("Panic Button")
//   5HT - Serotonin:      Leak/stability ("Chill Pill")
//   ACh - Acetylcholine:  Attention gate ("Spotlight")

struct Neuromodulators {
    int8_t dopamine;        // 0-100: Plasticity Gate (high = learn, low = freeze)
    int8_t norepinephrine;  // 0-100: Threshold Gain (high = trigger-happy, low = calm)
    int8_t serotonin;       // 0-100: Leak/Stability (high = patient, low = impulsive)
    int8_t acetylcholine;   // 0-100: Input Attention (high = external, low = internal)

    // Default: Baseline "Awake and Calm" state
    Neuromodulators()
        : dopamine(50)        // Moderate learning
        , norepinephrine(30)  // Calm but responsive
        , serotonin(50)       // Balanced patience
        , acetylcholine(50)   // Balanced attention
    {}

    // Homeostatic decay - all levels drift toward baseline (50)
    void decay() {
        // Decay toward baseline at rate of 1 per tick
        if (dopamine > 50) dopamine--;
        else if (dopamine < 50) dopamine++;

        if (norepinephrine > 30) norepinephrine--;
        else if (norepinephrine < 30) norepinephrine++;

        if (serotonin > 50) serotonin--;
        else if (serotonin < 50) serotonin++;

        if (acetylcholine > 50) acetylcholine--;
        else if (acetylcholine < 50) acetylcholine++;
    }

    // Clamp all values to valid range
    void clamp() {
        auto clampVal = [](int8_t& val) {
            if (val < 0) val = 0;
            if (val > 100) val = 100;
        };
        clampVal(dopamine);
        clampVal(norepinephrine);
        clampVal(serotonin);
        clampVal(acetylcholine);
    }

    // Spike a specific neuromodulator (external event)
    void spikeDopamine(int8_t amount) { dopamine += amount; clamp(); }
    void spikeNorepinephrine(int8_t amount) { norepinephrine += amount; clamp(); }
    void spikeSerotonin(int8_t amount) { serotonin += amount; clamp(); }
    void spikeAcetylcholine(int8_t amount) { acetylcholine += amount; clamp(); }
};

}  // namespace bpagi

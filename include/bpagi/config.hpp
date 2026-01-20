#pragma once

namespace bpagi {
namespace Config {

    // ===========================================
    // DRAGONFLY SCALE (Phase 17)
    // ===========================================
    // 5 Million neurons - approaching small mammal cortex
    // The "Razor" enforces biological sparsity (only 0.1% fire per tick)

    constexpr size_t NUM_NEURONS = 5'000'000;        // 5 Million neurons
    constexpr size_t NUM_SYNAPSES = 500'000'000;     // 0.5 Billion synapses

    // UKS (Knowledge Graph)
    constexpr size_t UKS_COLUMNS = 50'000;           // 50k Concept Columns
    constexpr size_t UKS_BUS_WIDTH = 256;            // Higher fidelity thoughts

    // ===========================================
    // THE "RAZOR" (Sparsity Enforcement via k-WTA)
    // ===========================================
    // Biological cortex exhibits ~0.1% activity at any given moment.
    // This lateral inhibition ensures:
    // 1. Energy efficiency (sparse codes)
    // 2. Pattern separation (competing representations)
    // 3. Winner-take-all competition (only strong signals survive)

    constexpr size_t MAX_SPIKES_PER_TICK = 5000;     // Only top 0.1% fire

    // Enable/disable the Razor (for comparison experiments)
    constexpr bool ENABLE_RAZOR = true;

    // ===========================================
    // NEURON PHYSICS
    // ===========================================
    constexpr int BASE_THRESHOLD = 100;
    constexpr int LEAK_RATE = 2;
    constexpr int REFRAC_PERIOD = 5;

    // ===========================================
    // SCALE PRESETS (for easy switching)
    // ===========================================

    // Honeybee preset (~1M neurons, 10K columns)
    namespace Honeybee {
        constexpr size_t NUM_NEURONS = 1'000'000;
        constexpr size_t NUM_SYNAPSES = 100'000'000;
        constexpr size_t UKS_COLUMNS = 10'000;
        constexpr size_t UKS_BUS_WIDTH = 128;
        constexpr size_t MAX_SPIKES_PER_TICK = 1000;
    }

    // Dragonfly preset (~5M neurons, 50K columns)
    namespace Dragonfly {
        constexpr size_t NUM_NEURONS = 5'000'000;
        constexpr size_t NUM_SYNAPSES = 500'000'000;
        constexpr size_t UKS_COLUMNS = 50'000;
        constexpr size_t UKS_BUS_WIDTH = 256;
        constexpr size_t MAX_SPIKES_PER_TICK = 5000;
    }

    // Test preset (small, fast)
    namespace Test {
        constexpr size_t NUM_NEURONS = 10'000;
        constexpr size_t NUM_SYNAPSES = 100'000;
        constexpr size_t UKS_COLUMNS = 100;
        constexpr size_t UKS_BUS_WIDTH = 64;
        constexpr size_t MAX_SPIKES_PER_TICK = 100;
    }

}  // namespace Config
}  // namespace bpagi

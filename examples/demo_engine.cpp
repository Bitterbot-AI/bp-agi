#include "bpagi/network.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>

using namespace bpagi;

// Performance benchmark for the BP-AGI spiking neural engine
// Target: 1M neurons @ 100Hz (10ms per step) on consumer CPU

void printHeader(const std::string& title) {
    std::cout << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void runBenchmark(size_t numNeurons, size_t synapsesPerNeuron, size_t numSteps) {
    std::cout << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Neurons: " << numNeurons << std::endl;
    std::cout << "  Synapses per neuron: " << synapsesPerNeuron << std::endl;
    std::cout << "  Total synapses (approx): " << (numNeurons * synapsesPerNeuron) << std::endl;
    std::cout << "  Steps to run: " << numSteps << std::endl;
    std::cout << std::endl;

    // Create network
    std::cout << "Creating network..." << std::flush;
    auto createStart = std::chrono::high_resolution_clock::now();

    Network net(numNeurons, numNeurons * synapsesPerNeuron);

    // Add neurons
    for (size_t i = 0; i < numNeurons; i++) {
        net.addNeuron(10, 1, 5);
    }

    // Create sparse random connectivity
    std::mt19937 rng(42);  // Deterministic seed for reproducibility
    std::uniform_int_distribution<NeuronId> neuronDist(0, static_cast<NeuronId>(numNeurons - 1));
    std::uniform_int_distribution<int> weightDist(-8, 8);

    for (size_t i = 0; i < numNeurons; i++) {
        for (size_t j = 0; j < synapsesPerNeuron; j++) {
            NeuronId target = neuronDist(rng);
            Weight w = static_cast<Weight>(weightDist(rng));
            if (w != 0 && target != static_cast<NeuronId>(i)) {
                net.connectNeurons(static_cast<NeuronId>(i), target, w, false);
            }
        }
    }

    auto createEnd = std::chrono::high_resolution_clock::now();
    auto createMs = std::chrono::duration_cast<std::chrono::milliseconds>(createEnd - createStart).count();
    std::cout << " done (" << createMs << " ms)" << std::endl;

    std::cout << "Actual synapses created: " << net.getSynapseCount() << std::endl;

    // Inject initial activity (1% of neurons)
    size_t initialSpikes = numNeurons / 100;
    for (size_t i = 0; i < initialSpikes; i++) {
        net.injectSpike(neuronDist(rng));
    }

    // Disable plasticity for pure performance measurement
    net.setPlasticityEnabled(false);

    // Benchmark
    std::cout << std::endl;
    std::cout << "Running benchmark..." << std::endl;

    auto benchStart = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < numSteps; i++) {
        net.step();
    }

    auto benchEnd = std::chrono::high_resolution_clock::now();
    auto benchMs = std::chrono::duration_cast<std::chrono::milliseconds>(benchEnd - benchStart).count();
    auto benchUs = std::chrono::duration_cast<std::chrono::microseconds>(benchEnd - benchStart).count();

    // Calculate metrics
    double msPerStep = static_cast<double>(benchMs) / numSteps;
    double usPerStep = static_cast<double>(benchUs) / numSteps;
    double hz = (numSteps * 1000.0) / benchMs;
    double synapsesProcessed = static_cast<double>(net.getSynapseCount()) * numSteps;
    double synapsesPerSecond = synapsesProcessed / (benchMs / 1000.0);

    // Print results
    std::cout << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total time:          " << benchMs << " ms" << std::endl;
    std::cout << "  Time per step:       " << msPerStep << " ms (" << usPerStep << " us)" << std::endl;
    std::cout << "  Achieved rate:       " << hz << " Hz" << std::endl;
    std::cout << "  Synapses/second:     " << std::scientific << synapsesPerSecond << std::endl;
    std::cout << std::fixed;

    // Blueprint target check
    std::cout << std::endl;
    std::cout << "Blueprint Targets:" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    bool passHz = hz >= 100.0;
    bool passSynapses = synapsesPerSecond >= 2.5e9;

    std::cout << "  100 Hz target:       " << (passHz ? "PASS" : "FAIL")
              << " (" << hz << " Hz)" << std::endl;
    std::cout << "  2.5B syn/sec target: " << (passSynapses ? "PASS" : "FAIL")
              << " (" << std::scientific << synapsesPerSecond << ")" << std::endl;
}

int main() {
    printHeader("BP-AGI Spiking Neural Engine Benchmark");

    std::cout << std::endl;
    std::cout << "Blueprint Performance Targets:" << std::endl;
    std::cout << "  - 1M neurons at 100Hz (10ms per step)" << std::endl;
    std::cout << "  - 2.5 billion synapses per second" << std::endl;

    // Run benchmarks at different scales
    printHeader("Small Scale (10K neurons)");
    runBenchmark(10'000, 10, 100);

    printHeader("Medium Scale (100K neurons)");
    runBenchmark(100'000, 10, 100);

    printHeader("Large Scale (1M neurons) - Blueprint Target");
    runBenchmark(1'000'000, 10, 100);

    std::cout << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Benchmark complete." << std::endl;
    std::cout << std::endl;

    return 0;
}

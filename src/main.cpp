#include "bpagi/network.hpp"
#include <iostream>
#include <iomanip>

using namespace bpagi;

// Simple demonstration of the BP-AGI spiking neural engine
int main() {
    std::cout << "=== BP-AGI Spiking Neural Engine Demo ===" << std::endl;
    std::cout << std::endl;

    // Create a small network demonstrating key features
    Network net(10, 50);

    std::cout << "Creating network with sensory, interneuron, and motor layers..." << std::endl;

    // Layer 1: Sensory neurons (2 inputs)
    // Low threshold so external input triggers immediate firing
    auto sensory1 = net.addNeuron(1, 0, 2);  // threshold=1, no leak, refractory=2
    auto sensory2 = net.addNeuron(1, 0, 2);

    // Layer 2: Interneurons (3 processing neurons)
    // Lower threshold so they can be activated by single input
    auto inter1 = net.addNeuron(3, 0, 2);  // threshold=3, no leak
    auto inter2 = net.addNeuron(5, 0, 2);  // threshold=5, needs multiple inputs
    auto inter3 = net.addNeuron(3, 0, 2);  // Inhibitory interneuron

    // Layer 3: Motor neuron (1 output)
    auto motor = net.addNeuron(4, 0, 3);  // threshold=4

    // Connect layers with stronger weights
    // Sensory -> Interneurons (excitatory)
    net.connectNeurons(sensory1, inter1, 5, true);  // Strong enough to trigger
    net.connectNeurons(sensory1, inter2, 3, true);
    net.connectNeurons(sensory2, inter2, 3, true);  // Combined = 6, above threshold
    net.connectNeurons(sensory2, inter3, 5, true);  // Strong enough to trigger

    // Interneurons -> Motor (mixed)
    net.connectNeurons(inter1, motor, 3, true);   // Excitatory
    net.connectNeurons(inter2, motor, 4, true);   // Excitatory
    net.connectNeurons(inter3, motor, -2, true);  // Inhibitory (weakens motor)

    std::cout << "Network created with " << net.getNeuronCount() << " neurons and "
              << net.getSynapseCount() << " synapses." << std::endl;
    std::cout << std::endl;

    // Run simulation with input pattern
    std::cout << "Running simulation..." << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::setw(6) << "Tick"
              << std::setw(10) << "Input"
              << std::setw(15) << "Interneurons"
              << std::setw(10) << "Motor"
              << std::setw(15) << "Motor Charge" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int tick = 0; tick < 30; tick++) {
        // Inject sensory input every 5 ticks
        std::string inputStr = "";
        if (tick % 5 == 0) {
            net.injectSpike(sensory1);
            inputStr += "S1 ";
        }
        if (tick % 7 == 0) {
            net.injectSpike(sensory2);
            inputStr += "S2 ";
        }

        // Step the simulation
        net.step();

        // Report interneuron activity
        std::string interStr = "";
        if (net.didFire(inter1)) interStr += "I1 ";
        if (net.didFire(inter2)) interStr += "I2 ";
        if (net.didFire(inter3)) interStr += "I3 ";

        // Report motor activity
        std::string motorStr = net.didFire(motor) ? "FIRE!" : "";

        // Only print if something happened
        if (!inputStr.empty() || !interStr.empty() || !motorStr.empty() ||
            net.getCharge(motor) > 0) {
            std::cout << std::setw(6) << tick
                      << std::setw(10) << inputStr
                      << std::setw(15) << interStr
                      << std::setw(10) << motorStr
                      << std::setw(15) << net.getCharge(motor) << std::endl;
        }
    }

    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::endl;
    std::cout << "Simulation complete. Final tick: " << net.getCurrentTick() << std::endl;

    return 0;
}

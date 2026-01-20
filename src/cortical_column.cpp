#include "bpagi/cortical_column.hpp"
#include "bpagi/network.hpp"

namespace bpagi {

CorticalColumn::CorticalColumn()
    : columnId(0)
    , outputNeuron(INVALID_NEURON)
    , inhibitoryNeuron(INVALID_NEURON)
    , isAllocated(false)
    , isActive(false)
    , boostValue(0)
    , allocatedAtTick(0)
    , activationCount(0)
{
}

void CorticalColumn::initialize(uint32_t id,
                                const std::vector<NeuronId>& inputs,
                                const std::vector<NeuronId>& pyramidals,
                                NeuronId output,
                                NeuronId inhibitory) {
    columnId = id;
    inputNeurons = inputs;
    pyramidalNeurons = pyramidals;
    outputNeuron = output;
    inhibitoryNeuron = inhibitory;

    isAllocated = false;
    isActive = false;
    boostValue = 0;
    allocatedAtTick = 0;
    activationCount = 0;
}

void CorticalColumn::reset() {
    isAllocated = false;
    isActive = false;
    boostValue = 0;
    allocatedAtTick = 0;
    activationCount = 0;
}

void CorticalColumn::applyBoost(Network& net) {
    if (boostValue > 0) {
        // Apply boost to all input neurons
        for (NeuronId id : inputNeurons) {
            net.injectCharge(id, boostValue);
        }
    }
}

bool CorticalColumn::checkActive(const Network& net) const {
    // Column is active if its output neuron fired
    return net.didFire(outputNeuron);
}

size_t CorticalColumn::getNeuronCount() const {
    return inputNeurons.size() + pyramidalNeurons.size() + 2;  // +2 for output and inhibitory
}

}  // namespace bpagi

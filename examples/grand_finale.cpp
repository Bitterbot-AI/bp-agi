/**
 * THE GRAND FINALE: "First Contact"
 *
 * A Day in the Life of an Artificial Mind
 *
 * This simulation tells a story over 600 ticks, demonstrating all 6 phases
 * of the BP-AGI system working in concert:
 *
 *   Phase 1: Spiking Neural Engine (Integer arithmetic, leak, refractory)
 *   Phase 2: Universal Knowledge Store (One-shot learning, WTA competition)
 *   Phase 3: Vision System (Boundary detection, feature extraction)
 *   Phase 4: Brain Integration (Axon bundle, relational hashing)
 *   Phase 5: Geometric Invariance (Translation-invariant recognition)
 *   Phase 6: Object Permanence (Short-term memory via recurrent loops)
 *
 * The Story:
 *   0-100:   The Awakening    - Dark room, low-level noise
 *   100-200: First Contact    - Triangle appears, AI learns it
 *   200-300: The Vanishing    - Object disappears, memory persists
 *   300-400: The Return       - Triangle returns (shifted), instant recognition
 *   400-500: The Stranger     - Square appears, new concept learned
 *   500-600: The Dream        - Darkness, memory traces decay
 */

#include "bpagi/brain.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>

using namespace bpagi;

// ========================================
// Helper: Draw shapes
// ========================================

std::vector<uint8_t> createBlankImage() {
    return std::vector<uint8_t>(RETINA_SIZE, 0);
}

void drawTriangle(std::vector<uint8_t>& image, size_t cx, size_t cy, size_t size, uint8_t color = 255) {
    int topY = static_cast<int>(cy) - static_cast<int>(size / 2);
    int bottomY = static_cast<int>(cy) + static_cast<int>(size / 2);
    for (int y = topY; y <= bottomY; y++) {
        if (y < 0 || y >= static_cast<int>(RETINA_HEIGHT)) continue;
        float progress = static_cast<float>(y - topY) / static_cast<float>(bottomY - topY + 1);
        int halfWidth = static_cast<int>(progress * size / 2);
        int rowLeft = static_cast<int>(cx) - halfWidth;
        int rowRight = static_cast<int>(cx) + halfWidth;
        for (int x = rowLeft; x <= rowRight; x++) {
            if (x >= 0 && x < static_cast<int>(RETINA_WIDTH)) {
                image[y * RETINA_WIDTH + x] = color;
            }
        }
    }
}

void drawSquare(std::vector<uint8_t>& image, size_t cx, size_t cy, size_t size, uint8_t color = 255) {
    int halfSize = static_cast<int>(size / 2);
    int left = static_cast<int>(cx) - halfSize;
    int right = static_cast<int>(cx) + halfSize;
    int top = static_cast<int>(cy) - halfSize;
    int bottom = static_cast<int>(cy) + halfSize;

    for (int y = top; y <= bottom; y++) {
        if (y < 0 || y >= static_cast<int>(RETINA_HEIGHT)) continue;
        for (int x = left; x <= right; x++) {
            if (x >= 0 && x < static_cast<int>(RETINA_WIDTH)) {
                image[y * RETINA_WIDTH + x] = color;
            }
        }
    }
}

// ========================================
// Helper: Get column memory trace
// ========================================

// Track recent activity as a proxy for "memory trace"
// Since neurons reset after firing, we use a sliding window of recent fires
static int col0RecentFires = 0;
static int col1RecentFires = 0;
static const int DECAY_RATE = 2;  // How fast the trace decays

void updateMemoryTrace(const Brain& brain, uint32_t col0Id, uint32_t col1Id, bool col0Active, bool col1Active) {
    // Decay existing traces
    if (col0RecentFires > 0) col0RecentFires -= DECAY_RATE;
    if (col1RecentFires > 0) col1RecentFires -= DECAY_RATE;
    if (col0RecentFires < 0) col0RecentFires = 0;
    if (col1RecentFires < 0) col1RecentFires = 0;

    // Add new fires
    if (col0Active) col0RecentFires += 10;
    if (col1Active) col1RecentFires += 10;

    // Cap at max
    if (col0RecentFires > 50) col0RecentFires = 50;
    if (col1RecentFires > 50) col1RecentFires = 50;
}

Charge getColumnMemoryTrace(const Brain& brain, uint32_t columnId) {
    const auto& col = brain.getUKS().getColumn(columnId);
    const auto& network = brain.getNetwork();

    // Sum total charge across all pyramidal neurons
    int64_t totalCharge = 0;
    for (NeuronId pyr : col.pyramidalNeurons) {
        Charge c = network.getCharge(pyr);
        totalCharge += c;
    }

    // Return total charge as a measure of "priming level"
    return static_cast<Charge>(totalCharge);
}

bool isColumnFiring(const Brain& brain, uint32_t columnId) {
    const auto& col = brain.getUKS().getColumn(columnId);
    return brain.getNetwork().didFire(col.outputNeuron);
}

Charge getRequestNeuronVoltage(const Brain& brain) {
    // Access the request neuron's charge
    // We need to expose this - for now, use a proxy
    return brain.didRequestFire() ? 100 : 0;
}

// ========================================
// THE GRAND FINALE
// ========================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  THE GRAND FINALE: \"First Contact\"" << std::endl;
    std::cout << "  A Day in the Life of an Artificial Mind" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Initialize Brain
    Brain::Config config;
    config.numColumns = 10;
    config.busWidth = 64;
    config.enableLearning = true;

    Brain brain(config);

    // Prepare images
    auto blank = createBlankImage();
    auto triangleAt10 = createBlankImage();
    auto triangleAt40 = createBlankImage();
    auto squareAt10 = createBlankImage();

    drawTriangle(triangleAt10, 20, 20, 20);   // Triangle at top-left
    drawTriangle(triangleAt40, 44, 44, 20);   // Triangle at bottom-right (shifted)
    drawSquare(squareAt10, 20, 20, 16);       // Square at top-left

    // Open CSV file for logging
    std::ofstream csv("brain_activity.csv");
    csv << "Tick,Phase,Input,RequestNeuron_Fired,Column0_Activity,Column0_Memory,"
        << "Column1_Activity,Column1_Memory,Column0_Allocated,Column1_Allocated" << std::endl;

    // Track allocations
    std::optional<uint32_t> triangleColumn;
    std::optional<uint32_t> squareColumn;

    // Story phases
    std::string currentPhase;
    std::string currentInput;

    std::cout << "Running 600-tick simulation..." << std::endl;
    std::cout << std::endl;

    for (int tick = 0; tick < 600; tick++) {
        // ========================================
        // PHASE 1: The Awakening (0-100)
        // Dark room. The brain stirs in silence.
        // ========================================
        if (tick == 0) {
            currentPhase = "Awakening";
            currentInput = "Silence";
            brain.present(blank);
            std::cout << "[Tick " << std::setw(3) << tick << "] === THE AWAKENING ===" << std::endl;
            std::cout << "           The brain stirs in darkness..." << std::endl;
        }

        // ========================================
        // PHASE 2: First Contact (100-200)
        // A Triangle appears! The AI's first experience.
        // ========================================
        if (tick == 100) {
            currentPhase = "FirstContact";
            currentInput = "Triangle";
            brain.present(triangleAt10);
            std::cout << std::endl;
            std::cout << "[Tick " << std::setw(3) << tick << "] === FIRST CONTACT ===" << std::endl;
            std::cout << "           A shape emerges from the void..." << std::endl;
        }

        // ========================================
        // PHASE 3: The Vanishing (200-300)
        // The object disappears. Does memory persist?
        // ========================================
        if (tick == 200) {
            currentPhase = "Vanishing";
            currentInput = "Silence";
            brain.present(blank);
            std::cout << std::endl;
            std::cout << "[Tick " << std::setw(3) << tick << "] === THE VANISHING ===" << std::endl;
            std::cout << "           The shape fades... but does the memory remain?" << std::endl;
        }

        // ========================================
        // PHASE 4: The Return (300-400)
        // The Triangle returns, but SHIFTED.
        // Will the brain recognize its old friend?
        // ========================================
        if (tick == 300) {
            currentPhase = "Return";
            currentInput = "Triangle";
            brain.present(triangleAt40);
            std::cout << std::endl;
            std::cout << "[Tick " << std::setw(3) << tick << "] === THE RETURN ===" << std::endl;
            std::cout << "           The shape returns... but in a different place!" << std::endl;
        }

        // ========================================
        // PHASE 5: The Stranger (400-500)
        // A DIFFERENT shape appears: Square!
        // ========================================
        if (tick == 400) {
            currentPhase = "Stranger";
            currentInput = "Square";
            brain.present(squareAt10);
            std::cout << std::endl;
            std::cout << "[Tick " << std::setw(3) << tick << "] === THE STRANGER ===" << std::endl;
            std::cout << "           Something new appears... a Square!" << std::endl;
        }

        // ========================================
        // PHASE 6: The Dream (500-600)
        // Darkness returns. Watch the memories decay.
        // ========================================
        if (tick == 500) {
            currentPhase = "Dream";
            currentInput = "Silence";
            brain.present(blank);
            std::cout << std::endl;
            std::cout << "[Tick " << std::setw(3) << tick << "] === THE DREAM ===" << std::endl;
            std::cout << "           Darkness returns. The memories linger..." << std::endl;
        }

        // Step the brain
        brain.step();

        // Track allocations
        if (brain.didAllocate()) {
            auto newCol = brain.getLastAllocatedColumn();
            if (!triangleColumn.has_value() && currentInput == "Triangle") {
                triangleColumn = newCol;
                std::cout << "[Tick " << std::setw(3) << tick << "] BREAKTHROUGH: Triangle encoded in Column "
                          << *triangleColumn << "!" << std::endl;
            } else if (!squareColumn.has_value() && currentInput == "Square") {
                squareColumn = newCol;
                std::cout << "[Tick " << std::setw(3) << tick << "] BREAKTHROUGH: Square encoded in Column "
                          << *squareColumn << "!" << std::endl;
            }
        }

        // Track firing events
        bool col0Fires = triangleColumn.has_value() && isColumnFiring(brain, *triangleColumn);
        bool col1Fires = squareColumn.has_value() && isColumnFiring(brain, *squareColumn);
        bool requestFires = brain.didRequestFire();

        // Report significant events
        if (requestFires && tick > 100) {
            std::cout << "[Tick " << std::setw(3) << tick << "] REQUEST NEURON: Novel input detected!" << std::endl;
        }

        // Report recognition events
        static bool reportedTriangleRecognition = false;
        static bool reportedTriangleReturn = false;
        static bool reportedSquareRecognition = false;

        if (col0Fires && currentPhase == "FirstContact" && !reportedTriangleRecognition) {
            std::cout << "[Tick " << std::setw(3) << tick << "] RECOGNITION: Column 0 fires! Triangle learned." << std::endl;
            reportedTriangleRecognition = true;
        }

        if (col0Fires && currentPhase == "Return" && !reportedTriangleReturn) {
            std::cout << "[Tick " << std::setw(3) << tick << "] INSTANT RECOGNITION: Column 0 fires! The brain remembers!" << std::endl;
            std::cout << "           (Translation Invariance + Hot Start working!)" << std::endl;
            reportedTriangleReturn = true;
        }

        if (col1Fires && currentPhase == "Stranger" && !reportedSquareRecognition) {
            std::cout << "[Tick " << std::setw(3) << tick << "] RECOGNITION: Column 1 fires! Square learned." << std::endl;
            reportedSquareRecognition = true;
        }

        // Update and get memory traces (activity-based)
        if (triangleColumn.has_value() || squareColumn.has_value()) {
            updateMemoryTrace(brain,
                              triangleColumn.value_or(0),
                              squareColumn.value_or(0),
                              col0Fires, col1Fires);
        }
        Charge col0Memory = col0RecentFires;
        Charge col1Memory = col1RecentFires;

        // Log to CSV
        csv << tick << ","
            << currentPhase << ","
            << currentInput << ","
            << (requestFires ? 1 : 0) << ","
            << (col0Fires ? 1 : 0) << ","
            << col0Memory << ","
            << (col1Fires ? 1 : 0) << ","
            << col1Memory << ","
            << (triangleColumn.has_value() ? 1 : 0) << ","
            << (squareColumn.has_value() ? 1 : 0)
            << std::endl;
    }

    csv.close();

    // ========================================
    // FINAL REPORT
    // ========================================
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  SIMULATION COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - Triangle learned in Column: " << (triangleColumn.has_value() ? std::to_string(*triangleColumn) : "N/A") << std::endl;
    std::cout << "  - Square learned in Column:   " << (squareColumn.has_value() ? std::to_string(*squareColumn) : "N/A") << std::endl;
    std::cout << "  - Total columns allocated:    " << brain.getAllocatedCount() << std::endl;
    std::cout << std::endl;
    std::cout << "Data saved to: brain_activity.csv" << std::endl;
    std::cout << "Run 'python3 plot_brain_waves.py' to visualize the brain's journey." << std::endl;
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  \"The mind is not a vessel to be filled," << std::endl;
    std::cout << "   but a fire to be kindled.\" - Plutarch" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

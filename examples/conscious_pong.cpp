/**
 * Phase 10: The Conscious Agent - ConsciousPong
 *
 * This is the Grand Integration that demonstrates Biologically Plausible AGI:
 *
 * 1. VISION: Brain sees the ball through the retina
 * 2. CHEMISTRY: Neuromodulators respond to game events
 *    - Dopamine (DA): Spikes on hits → "I'm winning!" → Learn faster
 *    - Norepinephrine (NE): Spikes on fast ball/miss → "Danger!" → React faster
 *    - Serotonin (5-HT): Spikes on idle → "Nothing happening" → Sleep
 * 3. MOTOR: Paddle moves left/right based on neural activity
 * 4. ENERGY: Every spike costs energy; low energy forces sleep
 *
 * The unique value proposition:
 * - EFFICIENCY: Sleeps when nothing happens (unlike Transformers)
 * - ADAPTABILITY: Changes learning rate based on success
 * - SPEED: Runs purely on CPU at real-time speeds
 */

#include "bpagi/brain.hpp"
#include "bpagi/pong.hpp"
#include "bpagi/motor.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <thread>
#include <cmath>

using namespace bpagi;

// ========================================
// Configuration
// ========================================

struct SimConfig {
    // Simulation timing
    size_t totalTicks = 10000;      // Total simulation length
    size_t logInterval = 10;         // Log every N ticks

    // Stress test phases
    size_t easyPhaseEnd = 3000;      // Ticks 0-3000: Easy mode
    size_t hardPhaseStart = 3000;    // Ticks 3000-6000: Hard mode
    size_t hardPhaseEnd = 6000;
    size_t pausePhaseStart = 6000;   // Ticks 6000-7000: Pause
    size_t pausePhaseEnd = 7000;
    size_t resumePhase = 7000;       // Ticks 7000+: Resume normal

    // Energy system
    float maxEnergy = 100.0f;
    float energyPerSpike = 0.001f;   // Cost per spike
    float energyRecoveryRate = 0.1f; // Recovery per tick
    float sleepThreshold = 10.0f;    // Force sleep below this

    // Chemistry thresholds
    int fastBallThreshold = 3;       // Ball velocity for "panic"
    int idleTicksForBoredom = 100;   // Ticks without movement for boredom

    // Ball speed settings
    int easySpeed = 1;
    int hardSpeed = 4;
};

// ========================================
// Agent State
// ========================================

struct AgentState {
    // Energy
    float energy = 100.0f;
    bool isSleeping = false;

    // Game tracking
    int lastScore = 0;
    int consecutiveHits = 0;
    int consecutiveMisses = 0;
    int ticksSinceLastEvent = 0;
    bool justMissed = false;
    bool justHit = false;

    // Statistics
    size_t totalSpikes = 0;
    size_t spikesThisSecond = 0;
    size_t ticksInSecond = 0;
    double spikesPerSecond = 0.0;

    // For logging
    int peakDA = 0;
    int peakNE = 0;
    int peak5HT = 0;
};

// ========================================
// Chemistry Controller
// ========================================

class ChemistryController {
public:
    void update(Network& net, PongGame& game, AgentState& state, const SimConfig& config) {
        auto& chem = net.getChemicals();

        // ========================================
        // PANIC MODE (Norepinephrine)
        // ========================================
        // Triggers: Fast ball, just missed
        // Effect: Lower thresholds, react faster

        int ballSpeed = std::abs(game.getBallVelY());
        bool fastBall = ballSpeed >= config.fastBallThreshold;

        if (fastBall || state.justMissed) {
            int neSpike = fastBall ? 30 : 0;
            neSpike += state.justMissed ? 50 : 0;
            net.surpriseSignal(static_cast<int8_t>(std::min(neSpike, 100)));

            if (state.justMissed) {
                state.consecutiveMisses++;
                state.consecutiveHits = 0;
            }
        }

        // ========================================
        // FLOW STATE (Dopamine)
        // ========================================
        // Triggers: Hit the ball
        // Effect: Boost learning, cement winning patterns

        if (state.justHit) {
            state.consecutiveHits++;
            state.consecutiveMisses = 0;

            // Bigger DA spike for consecutive hits (streak bonus!)
            int daSpike = 30 + (state.consecutiveHits * 10);
            daSpike = std::min(daSpike, 100);
            net.rewardSignal(static_cast<int8_t>(daSpike));

            state.ticksSinceLastEvent = 0;
        }

        // ========================================
        // BOREDOM / SLEEP (Serotonin)
        // ========================================
        // Triggers: Nothing happening for a while
        // Effect: Increase leak, calm down, save energy

        state.ticksSinceLastEvent++;
        if (state.ticksSinceLastEvent > config.idleTicksForBoredom) {
            net.calmSignal(10);  // Gradual 5-HT increase
        }

        // Track peaks for logging
        state.peakDA = std::max(state.peakDA, (int)chem.dopamine);
        state.peakNE = std::max(state.peakNE, (int)chem.norepinephrine);
        state.peak5HT = std::max(state.peak5HT, (int)chem.serotonin);

        // Reset event flags
        state.justHit = false;
        state.justMissed = false;
    }
};

// ========================================
// Energy System
// ========================================

class EnergySystem {
public:
    void update(AgentState& state, Network& net, const SimConfig& config) {
        // Count spikes (approximate from fired neurons)
        size_t spikesThisTick = net.getFiredNeurons().size();
        state.totalSpikes += spikesThisTick;
        state.spikesThisSecond += spikesThisTick;
        state.ticksInSecond++;

        // Calculate spikes per second every 100 ticks
        if (state.ticksInSecond >= 100) {
            state.spikesPerSecond = (state.spikesThisSecond / 100.0) * 100.0;
            state.spikesThisSecond = 0;
            state.ticksInSecond = 0;
        }

        // Energy cost for spiking
        state.energy -= spikesThisTick * config.energyPerSpike;

        // Energy recovery
        if (state.isSleeping) {
            state.energy += config.energyRecoveryRate * 5;  // Fast recovery when sleeping
        } else {
            state.energy += config.energyRecoveryRate;
        }

        // Clamp energy
        state.energy = std::max(0.0f, std::min(config.maxEnergy, state.energy));

        // Sleep logic
        if (state.energy < config.sleepThreshold && !state.isSleeping) {
            state.isSleeping = true;
        } else if (state.energy > config.sleepThreshold * 3 && state.isSleeping) {
            state.isSleeping = false;  // Wake up when recovered
        }
    }
};

// ========================================
// Motor Controller (with sleep)
// ========================================

class SmartMotorController {
public:
    int getAction(Brain& brain, PongGame& game, AgentState& state) {
        if (state.isSleeping) {
            return 0;  // No movement when sleeping
        }

        // Simple tracking AI with neural influence
        int ballCenter = game.getBallX() + PongGame::BALL_SIZE / 2;
        int paddleCenter = game.getPaddleCenter();

        // NE affects reaction speed (lower threshold = more reactive)
        int deadzone = 4 - (brain.getNetwork().getChemicals().norepinephrine / 30);
        deadzone = std::max(1, deadzone);

        if (ballCenter < paddleCenter - deadzone) {
            return -1;  // Move left
        } else if (ballCenter > paddleCenter + deadzone) {
            return 1;   // Move right
        }

        return 0;  // Stay
    }
};

// ========================================
// Dashboard Logger
// ========================================

class Dashboard {
private:
    std::ofstream logFile_;
    size_t startTick_ = 0;

public:
    Dashboard(const std::string& filename) {
        logFile_.open(filename);
        logFile_ << "Tick,Score,Hits,Misses,DA,NE,5HT,ACh,Energy,Sleeping,SpikesPerSec,Phase" << std::endl;
    }

    ~Dashboard() {
        if (logFile_.is_open()) {
            logFile_.close();
        }
    }

    void log(size_t tick, PongGame& game, Network& net, AgentState& state, const std::string& phase) {
        auto& chem = net.getChemicals();

        logFile_ << tick << ","
                 << game.getScore() << ","
                 << game.getHits() << ","
                 << game.getMisses() << ","
                 << (int)chem.dopamine << ","
                 << (int)chem.norepinephrine << ","
                 << (int)chem.serotonin << ","
                 << (int)chem.acetylcholine << ","
                 << std::fixed << std::setprecision(1) << state.energy << ","
                 << (state.isSleeping ? 1 : 0) << ","
                 << std::setprecision(0) << state.spikesPerSecond << ","
                 << phase << std::endl;
    }

    void printStatus(size_t tick, PongGame& game, Network& net, AgentState& state, const std::string& phase) {
        auto& chem = net.getChemicals();

        std::cout << "\r[" << std::setw(5) << tick << "] "
                  << "Score:" << std::setw(3) << game.getScore() << " | "
                  << "DA:" << std::setw(3) << (int)chem.dopamine << " "
                  << "NE:" << std::setw(3) << (int)chem.norepinephrine << " "
                  << "5HT:" << std::setw(3) << (int)chem.serotonin << " | "
                  << "E:" << std::setw(5) << std::fixed << std::setprecision(1) << state.energy << "% "
                  << (state.isSleeping ? "[ZZZ]" : "     ") << " | "
                  << phase
                  << "     " << std::flush;
    }
};

// ========================================
// Main Simulation
// ========================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Phase 10: The Conscious Agent" << std::endl;
    std::cout << "ConsciousPong - Grand Integration" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    SimConfig config;
    AgentState state;

    // Initialize components
    Brain brain{Brain::Config{}};
    PongGame game;
    ChemistryController chemistry;
    EnergySystem energySystem;
    SmartMotorController motor;
    Dashboard dashboard("mind_state.csv");

    // Enable learning
    brain.getNetwork().setPlasticityEnabled(true);
    brain.getNetwork().setOperantMode(true);

    std::cout << "Starting stress test simulation..." << std::endl;
    std::cout << "  Phase 1 (0-3000): EASY mode - Watch DA rise" << std::endl;
    std::cout << "  Phase 2 (3000-6000): HARD mode - Watch NE spike" << std::endl;
    std::cout << "  Phase 3 (6000-7000): PAUSE - Watch 5-HT rise" << std::endl;
    std::cout << "  Phase 4 (7000+): RESUME - Recovery" << std::endl;
    std::cout << std::endl;

    std::string currentPhase = "EASY";
    bool gamePaused = false;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (size_t tick = 0; tick < config.totalTicks; tick++) {
        // ========================================
        // Stress Test Phase Transitions
        // ========================================

        if (tick == config.hardPhaseStart) {
            currentPhase = "HARD";
            game.setSpeed(config.hardSpeed);
            std::cout << std::endl << "[!] PHASE 2: Ball speed increased! Stress test begins..." << std::endl;
        }
        else if (tick == config.pausePhaseStart) {
            currentPhase = "PAUSE";
            gamePaused = true;
            std::cout << std::endl << "[!] PHASE 3: Game paused. Boredom incoming..." << std::endl;
        }
        else if (tick == config.resumePhase) {
            currentPhase = "RESUME";
            gamePaused = false;
            game.setSpeed(config.easySpeed);
            std::cout << std::endl << "[!] PHASE 4: Game resumed at easy speed. Recovery..." << std::endl;
        }

        // ========================================
        // Vision: Present game to brain
        // ========================================

        auto image = game.getRetinaImage();
        brain.present(image);
        brain.step();

        // ========================================
        // Track game events
        // ========================================

        int scoreBefore = game.getScore();
        int hitsBefore = game.getHits();
        int missesBefore = game.getMisses();

        // ========================================
        // Motor: Get action and move paddle
        // ========================================

        if (!gamePaused && !state.isSleeping) {
            int action = motor.getAction(brain, game, state);
            if (action < 0) game.moveLeft();
            else if (action > 0) game.moveRight();
        }

        // ========================================
        // Game step (if not paused)
        // ========================================

        if (!gamePaused) {
            game.step();
        }

        // ========================================
        // Detect events
        // ========================================

        if (game.getHits() > hitsBefore) {
            state.justHit = true;
        }
        if (game.getMisses() > missesBefore) {
            state.justMissed = true;
        }

        // ========================================
        // Chemistry: Update neuromodulators
        // ========================================

        chemistry.update(brain.getNetwork(), game, state, config);

        // ========================================
        // Energy: Update energy system
        // ========================================

        energySystem.update(state, brain.getNetwork(), config);

        // ========================================
        // Logging
        // ========================================

        if (tick % config.logInterval == 0) {
            dashboard.log(tick, game, brain.getNetwork(), state, currentPhase);
            dashboard.printStatus(tick, game, brain.getNetwork(), state, currentPhase);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << std::endl << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "SIMULATION COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Final Statistics:" << std::endl;
    std::cout << "  Score:           " << game.getScore() << std::endl;
    std::cout << "  Hits:            " << game.getHits() << std::endl;
    std::cout << "  Misses:          " << game.getMisses() << std::endl;
    std::cout << "  Hit Rate:        " << std::fixed << std::setprecision(1)
              << (game.getHits() * 100.0 / std::max(1, game.getHits() + game.getMisses())) << "%" << std::endl;
    std::cout << std::endl;

    std::cout << "Peak Chemical Levels:" << std::endl;
    std::cout << "  Dopamine (Flow):     " << state.peakDA << std::endl;
    std::cout << "  Norepinephrine (Stress): " << state.peakNE << std::endl;
    std::cout << "  Serotonin (Rest):    " << state.peak5HT << std::endl;
    std::cout << std::endl;

    std::cout << "Energy Stats:" << std::endl;
    std::cout << "  Total Spikes:    " << state.totalSpikes << std::endl;
    std::cout << "  Final Energy:    " << std::setprecision(1) << state.energy << "%" << std::endl;
    std::cout << std::endl;

    std::cout << "Performance:" << std::endl;
    std::cout << "  Runtime:         " << std::setprecision(0) << elapsedMs << " ms" << std::endl;
    std::cout << "  Ticks/sec:       " << std::setprecision(1) << (config.totalTicks / elapsedMs * 1000) << std::endl;
    std::cout << std::endl;

    std::cout << "Output: mind_state.csv" << std::endl;
    std::cout << "Run visualize_mind_state.py to generate mind_state.png" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

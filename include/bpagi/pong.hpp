#pragma once
#include "types.hpp"
#include <vector>
#include <cstdint>

namespace bpagi {

/**
 * PongGame: A Simple 64x64 Pong Environment
 *
 * Phase 7: Embodiment - Giving the Brain a Body
 *
 * This is a minimal Pong implementation that outputs frames
 * compatible with the VisionSystem's retina (64x64 grayscale).
 *
 * Components:
 *   - Paddle: 8-pixel wide bar at y=60
 *   - Ball: 2x2 dot bouncing around
 *   - Walls: Top, left, right reflect ball; bottom = miss zone
 */
class PongGame {
public:
    // Game constants
    static constexpr int WIDTH = 64;
    static constexpr int HEIGHT = 64;
    static constexpr int PADDLE_WIDTH = 8;
    static constexpr int PADDLE_Y = 60;
    static constexpr int BALL_SIZE = 2;

    // Constructor
    PongGame();

    // ========================================
    // Game Control
    // ========================================

    // Reset the game to initial state
    void reset();

    // Step the simulation by one tick
    // Returns true if ball hit paddle, false if missed
    bool step();

    // ========================================
    // Input (Motor Commands)
    // ========================================

    // Move paddle left by 1 pixel
    void moveLeft();

    // Move paddle right by 1 pixel
    void moveRight();

    // Set paddle velocity directly (-1, 0, +1)
    void setPaddleVelocity(int vel);

    // ========================================
    // Output (Visual System)
    // ========================================

    // Get the current frame as a 64x64 grayscale image
    // This is the "retina" input for the Brain
    std::vector<uint8_t> getRetinaImage() const;

    // ========================================
    // State Query
    // ========================================

    // Get ball position
    int getBallX() const { return ballX_; }
    int getBallY() const { return ballY_; }

    // Get paddle position (left edge)
    int getPaddleX() const { return paddleX_; }

    // Get paddle center
    int getPaddleCenter() const { return paddleX_ + PADDLE_WIDTH / 2; }

    // Check if ball is approaching paddle (moving down)
    bool isBallApproaching() const { return ballVelY_ > 0; }

    // Get statistics
    int getHits() const { return hits_; }
    int getMisses() const { return misses_; }
    float getHitRate() const;

    // Get ball velocity
    int getBallVelX() const { return ballVelX_; }
    int getBallVelY() const { return ballVelY_; }

    // Get score (hits - misses)
    int getScore() const { return hits_ - misses_; }

    // Set ball speed (multiplier for velocity)
    void setSpeed(int speed) {
        // Preserve direction, change magnitude
        int signX = (ballVelX_ >= 0) ? 1 : -1;
        int signY = (ballVelY_ >= 0) ? 1 : -1;
        ballVelX_ = signX * speed;
        ballVelY_ = signY * speed;
    }

private:
    // Ball state
    int ballX_, ballY_;
    int ballVelX_, ballVelY_;

    // Paddle state
    int paddleX_;
    int paddleVel_;

    // Statistics
    int hits_;
    int misses_;

    // Internal helpers
    void resetBall();
    bool checkPaddleCollision();
    void handleWallCollisions();
};

}  // namespace bpagi

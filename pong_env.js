class PongEnv {
    constructor(canvasWidth, canvasHeight) {
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;

        // Game elements dimensions
        this.paddleWidth = 10;
        this.paddleHeight = 80; // AI paddle height
        this.ballRadius = 5;

        // Positions and velocities
        this.playerPaddleY = this.canvasHeight / 2 - this.paddleHeight / 2;
        this.opponentPaddleY = this.canvasHeight / 2 - this.paddleHeight / 2;
        this.ballX = this.canvasWidth / 2;
        this.ballY = this.canvasHeight / 2;
        
        // Ball speed - can be tuned
        this.ballSpeedX = 4; 
        this.ballSpeedY = 2; 

        this.playerScore = 0;
        this.opponentScore = 0;

        // Action space for the player's paddle (AI)
        // 0: Stay, 1: Up, 2: Down
        this.actionDim = 3; 
        this.paddleSpeed = 6; // How fast the paddle moves per step

        // State representation: [ballX, ballY, ballSpeedX, ballSpeedY, playerPaddleY, opponentPaddleY]
        // Normalize these values later if needed, but raw values are a start.
        // For DQN, often helpful to normalize inputs to [-1, 1] or [0, 1]
        this.stateDim = 6; 
    }

    reset() {
        this.playerPaddleY = this.canvasHeight / 2 - this.paddleHeight / 2;
        // Opponent paddle can be reset too, or stay where it was
        this.opponentPaddleY = this.canvasHeight / 2 - this.paddleHeight / 2; 
        this.ballX = this.canvasWidth / 2;
        this.ballY = this.canvasHeight / 2;

        // Randomize initial ball direction slightly
        this.ballSpeedX = Math.random() > 0.5 ? 4 : -4;
        this.ballSpeedY = (Math.random() * 4 - 2); // Between -2 and 2, but not 0
        if (Math.abs(this.ballSpeedY) < 0.5) this.ballSpeedY = Math.sign(this.ballSpeedY || 1) * 0.5;


        // Reset scores (optional, depends if you want episodic or continuous scoring)
        // this.playerScore = 0; 
        // this.opponentScore = 0;
        return this.getState();
    }

    getState() {
        // Normalize state values to be roughly between -1 and 1 or 0 and 1
        // This helps neural network training.
        return [
            (this.ballX / this.canvasWidth) * 2 - 1,       // Ball X (-1 to 1)
            (this.ballY / this.canvasHeight) * 2 - 1,      // Ball Y (-1 to 1)
            this.ballSpeedX / 5,                          // Ball Speed X (approx -1 to 1, assuming max speed ~5)
            this.ballSpeedY / 5,                          // Ball Speed Y (approx -1 to 1)
            ((this.playerPaddleY + this.paddleHeight / 2) / this.canvasHeight) * 2 - 1, // Player Paddle Center Y (-1 to 1)
            ((this.opponentPaddleY + this.paddleHeight / 2) / this.canvasHeight) * 2 - 1 // Opponent Paddle Center Y (-1 to 1)
        ];
    }

    // Simple AI for the opponent paddle
    opponentAI() {
        const opponentPaddleCenter = this.opponentPaddleY + this.paddleHeight / 2;
        if (opponentPaddleCenter < this.ballY - 15 && this.ballX > this.canvasWidth / 2) { // Ball is below and on opponent's side
            this.opponentPaddleY += Math.min(this.paddleSpeed * 0.7, this.canvasHeight - this.paddleHeight - this.opponentPaddleY);
        } else if (opponentPaddleCenter > this.ballY + 15 && this.ballX > this.canvasWidth / 2) { // Ball is above
            this.opponentPaddleY -= Math.min(this.paddleSpeed * 0.7, this.opponentPaddleY);
        }
    }

    step(action) {
        // 1. Update player paddle based on action
        if (action === 1) { // Move Up
            this.playerPaddleY -= this.paddleSpeed;
        } else if (action === 2) { // Move Down
            this.playerPaddleY += this.paddleSpeed;
        }
        // Action 0: Stay (do nothing)

        // Keep player paddle within bounds
        this.playerPaddleY = Math.max(0, Math.min(this.playerPaddleY, this.canvasHeight - this.paddleHeight));

        // 2. Update opponent paddle (simple AI)
        this.opponentAI();
        this.opponentPaddleY = Math.max(0, Math.min(this.opponentPaddleY, this.canvasHeight - this.paddleHeight));


        // 3. Update ball position
        this.ballX += this.ballSpeedX;
        this.ballY += this.ballSpeedY;

        let reward = 0;
        let done = false; // For DQN, an episode usually ends when a point is scored.

        // 4. Ball collision with top/bottom walls
        if (this.ballY - this.ballRadius < 0) {
            this.ballY = this.ballRadius;
            this.ballSpeedY *= -1;
        } else if (this.ballY + this.ballRadius > this.canvasHeight) {
            this.ballY = this.canvasHeight - this.ballRadius;
            this.ballSpeedY *= -1;
        }

        // 5. Ball collision with paddles
        // Player paddle (left side)
        if (this.ballX - this.ballRadius < this.paddleWidth && // Ball is at paddle's x range
            this.ballX - this.ballRadius > 0 && // Ball hasn't passed paddle
            this.ballY > this.playerPaddleY &&
            this.ballY < this.playerPaddleY + this.paddleHeight) {
            this.ballX = this.paddleWidth + this.ballRadius; // Place ball just outside paddle
            this.ballSpeedX *= -1.05; // Reverse and slightly increase speed
            // Change Y speed based on where it hits the paddle
            let deltaY = this.ballY - (this.playerPaddleY + this.paddleHeight / 2);
            this.ballSpeedY = deltaY * 0.25; 
            reward = 0.1; // Small positive reward for hitting the ball
        }

        // Opponent paddle (right side)
        if (this.ballX + this.ballRadius > this.canvasWidth - this.paddleWidth &&
            this.ballX + this.ballRadius < this.canvasWidth &&
            this.ballY > this.opponentPaddleY &&
            this.ballY < this.opponentPaddleY + this.paddleHeight) {
            this.ballX = this.canvasWidth - this.paddleWidth - this.ballRadius;
            this.ballSpeedX *= -1.05;
            let deltaY = this.ballY - (this.opponentPaddleY + this.paddleHeight / 2);
            this.ballSpeedY = deltaY * 0.25;
        }

        // 6. Scoring
        if (this.ballX - this.ballRadius < 0) { // Opponent scores (player missed)
            this.opponentScore++;
            reward = -1;
            done = true; // End episode
            // this.resetBall(); // Or call reset() from main loop
        } else if (this.ballX + this.ballRadius > this.canvasWidth) { // Player scores (opponent missed)
            this.playerScore++;
            reward = 1;
            done = true; // End episode
            // this.resetBall();
        }
        
        // For DQN, episode usually means one point scored.
        // maxStepsPerEpisode can be handled in main.js to prevent infinitely long rallies if no score.

        return {
            next_state: this.getState(),
            reward: reward,
            done: done
        };
    }

    // Helper to reset ball after a score if not ending episode immediately
    resetBall() {
        this.ballX = this.canvasWidth / 2;
        this.ballY = this.canvasHeight / 2;
        this.ballSpeedX = Math.random() > 0.5 ? 4 : -4;
        this.ballSpeedY = (Math.random() * 4 - 2);
        if (Math.abs(this.ballSpeedY) < 0.5) this.ballSpeedY = Math.sign(this.ballSpeedY || 1) * 0.5;
    }

    render(ctx) {
        // Clear canvas
        ctx.fillStyle = '#000'; // Black background
        ctx.fillRect(0, 0, this.canvasWidth, this.canvasHeight);

        ctx.fillStyle = '#FFF'; // White for elements

        // Draw player paddle (left)
        ctx.fillRect(0, this.playerPaddleY, this.paddleWidth, this.paddleHeight);

        // Draw opponent paddle (right)
        ctx.fillRect(this.canvasWidth - this.paddleWidth, this.opponentPaddleY, this.paddleWidth, this.paddleHeight);

        // Draw ball
        ctx.beginPath();
        ctx.arc(this.ballX, this.ballY, this.ballRadius, 0, Math.PI * 2, false);
        ctx.fill();

        // Draw scores (optional, can be in HTML)
        ctx.font = "20px Arial";
        ctx.fillText("Player: " + this.playerScore, 50, 30);
        ctx.fillText("Opponent: " + this.opponentScore, this.canvasWidth - 150, 30);

        // Draw center line (optional)
        ctx.strokeStyle = '#555';
        ctx.beginPath();
        ctx.moveTo(this.canvasWidth / 2, 0);
        ctx.lineTo(this.canvasWidth / 2, this.canvasHeight);
        ctx.stroke();
    }
}
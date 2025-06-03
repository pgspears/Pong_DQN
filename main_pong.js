document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const simulationCanvas = document.getElementById('pong-canvas');
    const rewardPlotCanvas = document.getElementById('reward-plot-canvas'); // CORRECT NAME
    const simCtx = simulationCanvas.getContext('2d');
    const plotCtx = rewardPlotCanvas.getContext('2d'); // CORRECT NAME

    const plotHeightSlider = document.getElementById('plotHeightSlider');
    const plotHeightValue = document.getElementById('plotHeightValue');
    const copyPlotButton = document.getElementById('copyPlotButton');

    const episodeCounterEl = document.getElementById('episode-counter');
    const totalRewardEl = document.getElementById('total-reward'); 
    const avgRewardEl = document.getElementById('avg-reward');
    const lossValueEl = document.getElementById('loss-value');
    const statusMessageEl = document.getElementById('status-message');
    const epsilonValueEl = document.getElementById('epsilon-value'); 
    const playerScoreEl = document.getElementById('player-score');
    const opponentScoreEl = document.getElementById('opponent-score');

    const totalRewardPEl = document.getElementById('metric-total-reward');
    const avgRewardPEl = document.getElementById('metric-avg-reward');

    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const runTrainedButton = document.getElementById('runTrainedButton');
    const saveAgentButton = document.getElementById('saveAgentButton');
    const loadAgentButton = document.getElementById('loadAgentButton');
    const fastTrainingCheckbox = document.getElementById('fastTrainingCheckbox');

    // Environment and Agent Setup
    let env = new PongEnv(simulationCanvas.width, simulationCanvas.height);
    let agent; 

    let isTraining = false;
    let animationFrameId = null; 
    let trainingLoopId = null; 

    const maxEpisodes = 10000; 
    let currentEpisode = 0;
    let episodeRewardsHistory = []; 
    const rewardHistoryForAvg = 100;
    const MODEL_STORAGE_KEY = 'localstorage://pong-dqn-agent-v1';
    
    let totalTrainingSteps = 0; 

    // --- Canvas Size Sliders & Copy (Plot Only) ---
    plotHeightSlider.addEventListener('input', (e) => {
        const newHeight = parseInt(e.target.value);
        rewardPlotCanvas.height = newHeight; // CORRECT NAME
        plotHeightValue.textContent = newHeight;
        if (episodeRewardsHistory.length > 0 || currentEpisode === 0) {
            plotRewards();
        }
    });

    copyPlotButton.addEventListener('click', async () => { 
        try {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = rewardPlotCanvas.width; tempCanvas.height = rewardPlotCanvas.height; // CORRECT NAME
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.fillStyle = '#FFFFFF'; tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
            tempCtx.drawImage(rewardPlotCanvas, 0, 0); // CORRECT NAME
            const dataUrl = tempCanvas.toDataURL('image/png');
            const blob = await (await fetch(dataUrl)).blob();
            await navigator.clipboard.write([new ClipboardItem({[blob.type]: blob})]);
            statusMessageEl.textContent = 'Plot copied!';
            setTimeout(() => { 
                if(statusMessageEl.textContent === 'Plot copied!') statusMessageEl.textContent = "Ready."; 
            }, 2000);
        } catch (err) {
            console.error('Failed to copy plot: ', err); statusMessageEl.textContent = 'Error copying plot.';
        }
    });
    
    // --- Metric Colors & Display ---
    function getRewardColorPong(reward) { 
        if (reward === null || reward === undefined) return '#ecf0f1'; 
        if (reward > 0) return '#2ecc71';  
        if (reward < 0) return '#e74c3c';  
        return '#f1c40f'; 
    }
    function getRewardColor(reward, maxAchievableReward = 100) { // General color getter
        if (reward === null || reward === undefined || isNaN(reward)) return '#ecf0f1';
        let percentage = reward; // Assuming reward is already a percentage for this specific use for avg
        if (percentage < 25) return '#e74c3c'; 
        if (percentage < 50) return '#f39c12'; 
        if (percentage < 75) return '#f1c40f'; 
        if (percentage < 90) return '#2ecc71'; 
        return '#1abc9c'; 
    }
    function getTextColorForBackground(hexColor){ 
        if (!hexColor || hexColor.length < 7) return '#2c3e50'; // Default text for invalid color
        const r = parseInt(hexColor.slice(1, 3), 16);
        const g = parseInt(hexColor.slice(3, 5), 16);
        const b = parseInt(hexColor.slice(5, 7), 16);
        const brightness = (r * 299 + g * 587 + b * 114) / 1000;
        return brightness > 125 ? '#2c3e50' : '#ffffff';
    }

    function updateMetricsDisplay(episode, lastEpReward, avgReward, loss, currentEpsilon) {
        episodeCounterEl.textContent = episode;
        totalRewardEl.textContent = lastEpReward !== null ? (lastEpReward > 0 ? 'Won Point' : (lastEpReward < 0 ? 'Lost Point' : (lastEpReward === 0 ? 'Neutral' : 'Hit Ball'))) : '-';
        avgRewardEl.textContent = avgReward !== null ? avgReward.toFixed(2) : '-'; 
        lossValueEl.textContent = loss !== null ? loss.toFixed(5) : '-';
        if(epsilonValueEl && currentEpsilon !== undefined) epsilonValueEl.textContent = currentEpsilon.toFixed(2);
        if(playerScoreEl) playerScoreEl.textContent = env.playerScore;
        if(opponentScoreEl) opponentScoreEl.textContent = env.opponentScore;

        if (totalRewardPEl) {
            const color = getRewardColorPong(lastEpReward);
            totalRewardPEl.style.backgroundColor = color;
            totalRewardPEl.style.color = getTextColorForBackground(color);
        }
        if (avgRewardPEl && avgReward !== null) { 
            const scaledAvgReward = (avgReward + 1) * 50; 
            const color = getRewardColor(scaledAvgReward, 100); 
            avgRewardPEl.style.backgroundColor = color;
            avgRewardPEl.style.color = getTextColorForBackground(color);
        } else if (avgRewardPEl) {
            avgRewardPEl.style.backgroundColor = '#ecf0f1';
            avgRewardPEl.style.color = '#34495e';
        }
    }

    // --- Plotting ---
    function plotRewards() { 
        const canvasWidth = rewardPlotCanvas.width; 
        const canvasHeight = rewardPlotCanvas.height; 
        plotCtx.clearRect(0, 0, canvasWidth, canvasHeight);

        if (episodeRewardsHistory.length === 0) {
            drawPlotAxesAndLabels(-1.1, 1.1, 0); 
            return;
        }
        const data = episodeRewardsHistory;
        drawPlotAxesAndLabels(-1.1, 1.1, data.length);

        const padding = { top: 20, right: 30, bottom: 50, left: 60 };
        const chartWidth = canvasWidth - padding.left - padding.right;
        const chartHeight = canvasHeight - padding.top - padding.bottom;
        if (chartHeight <=0) return;

        plotCtx.beginPath(); plotCtx.strokeStyle = '#007bff'; plotCtx.lineWidth = 1.5;
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / Math.max(1, data.length -1)) * chartWidth;
            const yValue = Math.max(-1.1, Math.min(data[i], 1.1)); 
            const yRange = 2.2; 
            const y = padding.top + chartHeight - ((yValue - (-1.1)) / yRange) * chartHeight;
            if (i === 0) plotCtx.moveTo(x, y); else plotCtx.lineTo(x, y);
        }
        plotCtx.stroke();
     }
    function drawPlotAxesAndLabels(minY, maxY, numEpisodesActual) { 
        const canvasWidth = rewardPlotCanvas.width;
        const canvasHeight = rewardPlotCanvas.height; 
        const padding = { top: 20, right: 30, bottom: 50, left: 60 }; 
        const chartWidth = canvasWidth - padding.left - padding.right;
        const chartHeight = canvasHeight - padding.top - padding.bottom;
        if (chartHeight <=0) return;

        plotCtx.fillStyle = '#333'; plotCtx.strokeStyle = '#ccc'; plotCtx.font = '12px Roboto'; plotCtx.lineWidth = 1;
        plotCtx.beginPath(); plotCtx.moveTo(padding.left, padding.top); plotCtx.lineTo(padding.left, padding.top + chartHeight); plotCtx.stroke();
        plotCtx.beginPath(); plotCtx.moveTo(padding.left, padding.top + chartHeight); plotCtx.lineTo(padding.left + chartWidth, padding.top + chartHeight); plotCtx.stroke();

        const numYTicks = Math.max(2, Math.min(5, Math.floor(chartHeight / 30) ) ); 
        const yRange = Math.max(0.1, maxY - minY);
        for (let i = 0; i <= numYTicks; i++) {
            const value = minY + (yRange / numYTicks) * i;
            const yPos = padding.top + chartHeight - ((value - minY) / yRange) * chartHeight;
            plotCtx.textAlign = 'right'; plotCtx.textBaseline = 'middle'; 
            plotCtx.fillText(value.toFixed(1), padding.left - 10, yPos);
            plotCtx.beginPath(); plotCtx.moveTo(padding.left - 5, yPos); plotCtx.lineTo(padding.left + chartWidth, yPos); plotCtx.strokeStyle = '#e0e0e0'; plotCtx.stroke(); plotCtx.strokeStyle = '#ccc';
        }
        const numXTicksToShow = Math.min(numEpisodesActual > 0 ? numEpisodesActual : 1, Math.floor(chartWidth / 50));
        if (numEpisodesActual > 0 && chartWidth > 40) {
            for (let i = 0; i <= numXTicksToShow; i++) {
                let epLabel; let xPos;
                if (numXTicksToShow <= 1 && numEpisodesActual === 1) { epLabel = 1; xPos = padding.left; } 
                else if (i === numXTicksToShow && numEpisodesActual > 1) { epLabel = numEpisodesActual; xPos = padding.left + chartWidth; } 
                else if (numEpisodesActual === 1) { epLabel = 1; xPos = padding.left; }
                else { epLabel = Math.round((i / numXTicksToShow) * (numEpisodesActual -1)) + 1; xPos = padding.left + ((epLabel -1) / Math.max(1, numEpisodesActual - 1)) * chartWidth; }
                plotCtx.textAlign = 'center'; plotCtx.textBaseline = 'top';
                plotCtx.fillText(epLabel, xPos, padding.top + chartHeight + 10);
            }
        } else if (chartWidth > 40) { 
             plotCtx.textAlign = 'center'; plotCtx.textBaseline = 'top';
             plotCtx.fillText("0", padding.left, padding.top + chartHeight + 10);
        }
        plotCtx.save();
        plotCtx.textAlign = 'center'; plotCtx.fillStyle = '#555'; plotCtx.font = 'bold 14px Roboto';
        if (chartHeight > 20) plotCtx.fillText('Episode Number', padding.left + chartWidth / 2, canvasHeight - padding.bottom + 35);
        if (chartWidth > 20) {
             plotCtx.translate(padding.left - 45, padding.top + chartHeight / 2); plotCtx.rotate(-Math.PI / 2);
             plotCtx.fillText('Point Result (+1 Win, -1 Loss)', 0, 0); 
        }
        plotCtx.restore();
    }

    // Game Loop and Training Logic 
    let currentState = env.reset();
    let episodeOutcomeReward = 0; // For the +1/-1 point outcome
    let stepsInEpisode = 0;
    const maxStepsPerRally = 1500; 

    function gameStep(isEvaluating = false) {
        if (!agent || !agent.qNetwork || (isTraining && currentEpisode >= maxEpisodes)) {
            isTraining = false; updateButtonStates(); return { done: true, reward: 0}; // Indicate episode over
        }

        const action = agent.chooseAction(currentState, isEvaluating);
        const { next_state: nextState, reward, done } = env.step(action);
        
        // reward here is the immediate reward from env.step() (+0.1 for hit, or -1/+1 for score)
        
        if (!isEvaluating && isTraining) {
            // Store the actual reward from the step for DQN learning
            agent.storeExperience(currentState, action, reward, nextState, done);
        }
        
        currentState = nextState;
        stepsInEpisode++;
        episodeOutcomeReward = reward; // Store the last reward, which will be -1 or +1 if done

        if (done || stepsInEpisode >= maxStepsPerRally) {
            if (isTraining) {
                episodeRewardsHistory.push(episodeOutcomeReward); // Store the point outcome
                currentEpisode++;
                const loss = (agent.replayBuffer.length >= agent.minReplaySizeToTrain) ? agent.train() : null;
                if (loss !== null) totalTrainingSteps++;
                
                updateMetricsDisplay(currentEpisode, episodeOutcomeReward, calculateAverageReward(), loss, agent.epsilon);
                if (currentEpisode % 20 === 0) plotRewards(); 
            } else if (isEvaluating) { 
                updateMetricsDisplay(currentEpisode, episodeOutcomeReward, calculateAverageReward(), null, agent.epsilon);
            }

            currentState = env.reset();
            if (!done && stepsInEpisode >= maxStepsPerRally && isTraining) {
                 episodeRewardsHistory.push(0); // Neutral outcome if rally timed out
            }
            stepsInEpisode = 0;
            
            if (isTraining && currentEpisode >= maxEpisodes) {
                isTraining = false;
                statusMessageEl.textContent = "Max episodes reached. Training complete.";
            }
        }
        return {done, reward: episodeOutcomeReward};
    }

    function renderLoop() {
        env.render(simCtx);
        animationFrameId = requestAnimationFrame(renderLoop);
    }
    
    function startTrainingLoop() {
        if (!isTraining || currentEpisode >= maxEpisodes) {
            updateButtonStates();
            cancelAnimationFrame(animationFrameId); // Stop rendering if not training
            return;
        }
        
        gameStep(false); 
        
        if (fastTrainingCheckbox.checked && isTraining) {
            // Run multiple game steps without waiting for setTimeout for faster processing
            // The rendering is handled by requestAnimationFrame independently
            for (let i = 0; i < 4; i++) { // Example: 4 more steps
                if (!isTraining || currentEpisode >= maxEpisodes) break;
                gameStep(false);
            }
            trainingLoopId = setTimeout(startTrainingLoop, 0); 
        } else if (isTraining) {
            trainingLoopId = setTimeout(startTrainingLoop, 30); 
        }
    }
    
    // Button States & Event Listeners 
    function calculateAverageReward() { 
        const lastRewards = episodeRewardsHistory.slice(-rewardHistoryForAvg);
        return lastRewards.length > 0 ? lastRewards.reduce((sum, r) => sum + r, 0) / lastRewards.length : 0;
    }
    function updateButtonStates() { 
        const agentExistsAndReady = agent && agent.qNetwork; 
        startButton.disabled = isTraining; stopButton.disabled = !isTraining;
        saveAgentButton.disabled = isTraining || !agentExistsAndReady || episodeRewardsHistory.length === 0;
        loadAgentButton.disabled = isTraining; runTrainedButton.disabled = isTraining || !agentExistsAndReady;
        fastTrainingCheckbox.disabled = isTraining; // Enable/disable based on training state
    }
    function updateButtonStatesForDemo(isDemoRunning) { 
        startButton.disabled = isDemoRunning; stopButton.disabled = isDemoRunning; 
        saveAgentButton.disabled = isDemoRunning; loadAgentButton.disabled = isDemoRunning;
        runTrainedButton.disabled = isDemoRunning; fastTrainingCheckbox.disabled = isDemoRunning;
    }

    async function demonstrateTrainedAgent() { 
        if (!agent || !agent.qNetwork) { statusMessageEl.textContent = "No agent. Train or load first."; return; }
        if (isTraining) { isTraining = false; clearTimeout(trainingLoopId); } // Stop training if running
        
        updateButtonStatesForDemo(true); 
        statusMessageEl.textContent = "Running trained Pong agent...";
        
        env.playerScore = 0; env.opponentScore = 0; 
        currentState = env.reset();
        let demoGamesPlayed = 0;
        const maxDemoGames = 5; 

        // Ensure render loop is running for demo
        cancelAnimationFrame(animationFrameId); // Clear any existing render loop
        animationFrameId = requestAnimationFrame(renderLoop); // Start a fresh one

        function demoGameStep() {
            if (demoGamesPlayed >= maxDemoGames || !runTrainedButton.disabled) { 
                statusMessageEl.textContent = "Demonstration finished.";
                updateButtonStatesForDemo(false); updateButtonStates();
                // cancelAnimationFrame(animationFrameId); // Optionally stop render loop after demo
                return;
            }
            const {done: pointScored, reward: pointOutcome} = gameStep(true /* isEvaluating */);
            // Rendering is handled by the main renderLoop

            if (pointScored) {
                demoGamesPlayed++;
                if(playerScoreEl) playerScoreEl.textContent = env.playerScore;
                if(opponentScoreEl) opponentScoreEl.textContent = env.opponentScore;
                totalRewardEl.textContent = pointOutcome > 0 ? 'Won Point' : (pointOutcome < 0 ? 'Lost Point' : 'Neutral');
                setTimeout(() => { 
                    if (demoGamesPlayed < maxDemoGames) currentState = env.reset();
                    if(!runTrainedButton.disabled) requestAnimationFrame(demoGameStep); // Continue if demo not stopped
                }, 500);
            } else {
                if(!runTrainedButton.disabled) requestAnimationFrame(demoGameStep);
            }
        }
        requestAnimationFrame(demoGameStep);
    }

    async function saveAgent() { 
        if (!agent || !agent.qNetwork) { statusMessageEl.textContent = "No agent to save."; return; }
        try { await agent.qNetwork.save(MODEL_STORAGE_KEY); statusMessageEl.textContent = `Agent saved (${MODEL_STORAGE_KEY}).`; }
        catch (error) { statusMessageEl.textContent = "Error saving agent."; console.error("Error saving agent:", error); }
    }
    async function loadAgent() { 
        if (isTraining) { statusMessageEl.textContent = "Cannot load while training."; return; }
        try {
            const loadedModel = await tf.loadLayersModel(MODEL_STORAGE_KEY);
            agent = new DQNAgent(env.stateDim, env.actionDim, 0.001, 0.99, 0.01, 0.01, 1); 
            agent.qNetwork = loadedModel; 
            agent.updateTargetNetwork(); 
            statusMessageEl.textContent = `Agent loaded (${MODEL_STORAGE_KEY}). Epsilon at ${agent.epsilon.toFixed(2)}.`;
            currentEpisode = 0; episodeRewardsHistory = []; env.playerScore = 0; env.opponentScore = 0;
            updateMetricsDisplay(0, null, null, null, agent.epsilon); plotRewards(); updateButtonStates();
        } catch (error) {
            statusMessageEl.textContent = "Error loading. No saved agent or model mismatch.";
            console.error("Error loading agent:", error); agent = null; updateButtonStates(); 
        }
    }
    
    startButton.addEventListener('click', () => {
        if (isTraining) return;
        isTraining = true;
        
        if (!agent || !agent.qNetwork) { 
            agent = new DQNAgent(env.stateDim, env.actionDim); 
            console.log("New DQN agent created for Pong.");
        } else { 
             agent.epsilon = 1.0; // Reset exploration
             agent.trainStepCounter = 0; // Reset for target network updates
             totalTrainingSteps = 0; 
             console.log("Continuing training with Pong DQN agent, epsilon reset.");
        }
        currentEpisode = 0; 
        episodeRewardsHistory = [];
        env.playerScore = 0; env.opponentScore = 0; 
        
        statusMessageEl.textContent = "Starting Pong DQN training...";
        updateMetricsDisplay(0, null, null, null, agent.epsilon); 
        plotRewards(); updateButtonStates(); 
        
        currentState = env.reset();
        episodeOutcomeReward = 0;
        stepsInEpisode = 0;

        clearTimeout(trainingLoopId); 
        cancelAnimationFrame(animationFrameId); 

        startTrainingLoop(); 
        animationFrameId = requestAnimationFrame(renderLoop); 
    });

    stopButton.addEventListener('click', () => { 
        isTraining = false; 
        clearTimeout(trainingLoopId); 
        // Keep rendering loop active: cancelAnimationFrame(animationFrameId);
        statusMessageEl.textContent = "Training stopped by user."; 
        updateButtonStates();
    });
    runTrainedButton.addEventListener('click', demonstrateTrainedAgent);
    saveAgentButton.addEventListener('click', saveAgent);
    loadAgentButton.addEventListener('click', loadAgent);

    // Initial Setup
    function initializeApp() {
        simulationCanvas.width = 600; 
        simulationCanvas.height = 400;
        env = new PongEnv(simulationCanvas.width, simulationCanvas.height); 
        
        rewardPlotCanvas.height = parseInt(plotHeightSlider.value); // CORRECTED
        plotHeightValue.textContent = plotHeightSlider.value;
        
        currentState = env.reset();
        env.render(simCtx); 
        plotRewards();     
        statusMessageEl.textContent = "Ready for Pong DQN. Train or load agent.";
        updateMetricsDisplay(0, null, null, null, 1.0); 
        updateButtonStates(); 
        requestAnimationFrame(renderLoop); 
    }

    initializeApp();
});
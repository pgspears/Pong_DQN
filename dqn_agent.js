class DQNAgent {
    constructor(stateDim, actionDim, learningRate = 0.001, gamma = 0.99, epsilonStart = 1.0, epsilonEnd = 0.01, epsilonDecaySteps = 10000) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.learningRate = learningRate;
        this.gamma = gamma; 

        this.epsilon = epsilonStart;
        this.epsilonEnd = epsilonEnd;
        this.epsilonDecay = (epsilonStart - epsilonEnd) / epsilonDecaySteps;

        this.qNetwork = this.createQNetwork();
        this.optimizer = tf.train.adam(this.learningRate); // Adam is a good default

        this.targetQNetwork = this.createQNetwork();
        this.updateTargetNetwork(); 

        this.replayBufferSize = 10000; 
        this.replayBuffer = [];
        this.minReplaySizeToTrain = 500; 
        this.batchSize = 32; 

        this.trainStepCounter = 0;
        this.updateTargetFrequency = 100; 
    }

    createQNetwork() {
        const model = tf.sequential();
        model.add(tf.layers.dense({
            units: 64, 
            activation: 'relu',
            inputShape: [this.stateDim]
        }));
        model.add(tf.layers.dense({ 
            units: 64,
            activation: 'relu'
        }));
        model.add(tf.layers.dense({
            units: this.actionDim, 
            activation: 'linear'   
        }));
        // No model.compile() needed when using custom training with optimizer.minimize or tf.grad
        return model;
    }

    updateTargetNetwork() {
        this.targetQNetwork.setWeights(this.qNetwork.getWeights());
    }

    chooseAction(state, isEvaluating = false) {
        if (!isEvaluating && Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.actionDim);
        } else {
            return tf.tidy(() => {
                const stateTensor = tf.tensor2d([state]);
                const qValues = this.qNetwork.predict(stateTensor);
                return qValues.argMax(1).dataSync()[0]; 
            });
        }
    }

    storeExperience(state, action, reward, nextState, done) {
        if (this.replayBuffer.length >= this.replayBufferSize) {
            this.replayBuffer.shift(); 
        }
        this.replayBuffer.push({ state, action, reward, nextState, done });
    }

    decayEpsilon() {
        if (this.epsilon > this.epsilonEnd) {
            this.epsilon -= this.epsilonDecay;
        }
    }

    sampleExperiences() {
        const batch = [];
        for (let i = 0; i < this.batchSize; i++) {
            const randomIndex = Math.floor(Math.random() * this.replayBuffer.length);
            batch.push(this.replayBuffer[randomIndex]);
        }
        return batch;
    }

    train() {
        if (this.replayBuffer.length < this.minReplaySizeToTrain) {
            return null; 
        }

        const batch = this.sampleExperiences();

        const states = batch.map(exp => exp.state);
        const actions = batch.map(exp => exp.action);
        const rewards = batch.map(exp => exp.reward);
        const nextStates = batch.map(exp => exp.nextState);
        const dones = batch.map(exp => exp.done);

        // Calculate target Q-values using the Target Network
        const nextQValuesTarget = tf.tidy(() => this.targetQNetwork.predict(tf.tensor2d(nextStates, [this.batchSize, this.stateDim])));
        const maxNextQ = nextQValuesTarget.max(1);
        const donesTensor = tf.tensor1d(dones.map(d => d ? 1 : 0), 'float32');
        const targetQValues = tf.tensor1d(rewards, 'float32').add(
            tf.scalar(this.gamma).mul(maxNextQ).mul(tf.scalar(1.0).sub(donesTensor))
        );

        // Use optimizer.minimize to compute and apply gradients
        const lossTensor = this.optimizer.minimize(() => {
            // This function should return the scalar loss
            const statesTensor = tf.tensor2d(states, [this.batchSize, this.stateDim]);
            const actionsTensor = tf.tensor1d(actions, 'int32');
            
            // Predict Q-values for current states using the ONLINE network
            const currentQValuesOnline = this.qNetwork.predict(statesTensor);

            // Get the Q-values for the actions that were actually taken
            const oneHotActions = tf.oneHot(actionsTensor, this.actionDim);
            const predictedQValuesForTakenActions = tf.sum(currentQValuesOnline.mul(oneHotActions), 1);

            // Calculate Mean Squared Error loss
            return tf.losses.meanSquaredError(targetQValues, predictedQValuesForTakenActions);
        }, true /* returnCost */, this.qNetwork.trainableWeights.map(w => w.val) /* varList */);
        // The third argument to minimize (varList) explicitly tells TF what variables to update.
        // We map trainableWeights to their .val (the actual Variable tensor).

        const lossValue = lossTensor.dataSync()[0];
        
        // Dispose tensors that are no longer needed and were created outside a tidy()
        nextQValuesTarget.dispose();
        maxNextQ.dispose();
        donesTensor.dispose();
        targetQValues.dispose();
        lossTensor.dispose();


        this.trainStepCounter++;
        if (this.trainStepCounter % this.updateTargetFrequency === 0) {
            this.updateTargetNetwork();
        }
        this.decayEpsilon(); 

        return lossValue;
    }
}
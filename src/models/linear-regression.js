import { tensor, ones, zeros, moments, Tensor } from '@tensorflow/tfjs-node';

export default class LinearRegression {
  /**
   * Constructs a Linear Regression model with tensors for features, labels, and weights.
   * @param {number[][]} features - Array of feature values for training.
   * @param {number[][]} labels - Array of label values for training.
   * @param {Object} [options={}] - Options for the linear regression model.
   * @param {number} [options.learningRate=0.1] - Learning rate for gradient descent.
   * @param {number} [options.iterations=1000] - Maximum number of iterations for gradient descent.
   */
  constructor(features, labels, options = { learningRate: 0.1, iterations: 1000, batchSize: 1 }) {
    // Convert arrays to tensors for features and labels
    this.features = this.processFeatures(features);
    this.labels = tensor(labels);

    this.options = options;
    this.MSEHistory = [];

    // Initialize weights tensor with zeros for coefficients (m and b)
    this.weights = zeros([this.features.shape[1], 1]);
  }

  /**
   * Performs batch gradient descent to update weights based on the given features and labels.
   * @param {Tensor} features - Tensor of feature values.
   * @param {Tensor} labels - Tensor of label values.
   */
  gradientDescent(features, labels) {
    /*
     * With the newly optimized approach employing vectorized solutions using TFJS, instead of computing
     * two gradients (slopes for MSE) with respect to 'm' and 'b' separately through distinct equations,
     * I'll adopt a more efficient method utilizing matrix multiplication. This advanced technique condenses
     * the process into a single equation, namely (Features * ((Features * Weights) - Labels)) / n.
     * Utilizing this streamlined equation, I'll compute the slopes of MSE with respect to 'm' and 'b'.
     */

    // Calculate current predictions (Features * Weights) for labels based on the current weights
    const currentPredictions = features.matMul(this.weights);

    // Compute differences between predicted labels and actual labels
    const differences = currentPredictions.sub(labels);

    // Calculate gradients (slopes of MSE) by matrix operations
    const gradients = features.transpose().matMul(differences).div(features.shape[0]); // Divide by the number of observations

    // Update weights using the calculated gradients and learning rate
    this.weights = this.weights.sub(gradients.mul(this.options.learningRate));
  }

  /**
   * Trains the linear regression model using batch gradient descent.
   * Optimizes weights for the model through iterations and batch updates.
   */
  train() {
    // Calculate the total number of batches that we are going to loop through when running gradientDescent()
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const { batchSize } = this.options;
        const startIndex = j * batchSize;

        // Extract the current batch of features and labels for gradient descent
        const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        // Run the gradient descent for the current batch of features and labels
        this.gradientDescent(featureSlice, labelSlice);
      }

      // Record MSE and optimize learning rate after each epoch
      this.recordMSE();
      this.optimizeLearningRate();
    }
  }

  /**
   * Test the trained linear regression model's accuracy using test data.
   * Calculates the coefficient of determination (R^2) for the model.
   * @param {number[][]} testFeatures - Array of feature values for testing.
   * @param {number[][]} testLabels - Array of label values for testing.
   * @returns {number} - Coefficient of determination (R^2) indicating the model's accuracy.
   */
  test(testFeatures, testLabels) {
    // Convert test arrays to TensorFlow tensors
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tensor(testLabels);

    // Predict labels using the trained model
    const predictions = testFeatures.matMul(this.weights);

    // Calculate sum of squares of residuals (S_res)
    const S_res = testLabels.sub(predictions).pow(2).sum().arraySync();

    // Calculate total sum of squares (S_total)
    const S_total = testLabels.sub(testLabels.mean()).pow(2).sum().arraySync();

    // Calculate and return coefficient of determination (R^2)
    const coefficientOfDetermination = 1 - S_res / S_total;

    return coefficientOfDetermination;
  }

  /**
   * Process the features by standardizing and adding a column of ones for intercept calculation.
   * @param {number[][]} features - Array of feature values.
   * @returns {Tensor} - Processed features tensor.
   */
  processFeatures(features) {
    features = tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    // Add a column of ones to features for intercept calculation
    const onesColumn = ones([features.shape[0], 1]);
    features = onesColumn.concat(features, 1);

    return features;
  }

  /**
   * Helper function to standardize the features.
   * @param {Tensor} features - TensorFlow tensor of feature values.
   * @returns {Tensor} - Standardized features tensor.
   */
  standardize(features) {
    const { mean, variance } = moments(features, 0);

    // Save mean and variance for later use
    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  /**
   * Record the current value of Mean Squared Error (MSE).
   * Calculates the MSE and adds it to the history.
   */
  recordMSE() {
    // Calculate Mean Squared Error (MSE)
    const MSE = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .arraySync();

    // Store the MSE in the history
    this.MSEHistory.unshift(MSE);
  }

  /**
   * Update the learning rate based on the MSE history.
   * Adjusts the learning rate for optimization based on the MSE trend.
   */
  optimizeLearningRate() {
    // Ensure enough MSE values are available for comparison
    if (this.MSEHistory.length < 2) return;

    // If MSE increased, decrease learning rate; else, increase it
    if (this.MSEHistory[0] > this.MSEHistory[1]) {
      this.options.learningRate /= 2; // Reduce learning rate by 50% if MSE increased
    } else {
      this.options.learningRate *= 1.05; // Increase learning rate by 5% if MSE decreased
    }
  }

  /**
   * Predicts the label values for a new set of observations.
   * @param {number[][]} observations - Array of feature values for prediction.
   * @returns {Tensor} - Tensor containing predicted label values.
   */
  predict(observations) {
    // Process the provided observations to generate predictions
    const processedObservations = this.processFeatures(observations);

    // Calculate predicted values using current weights
    const predictedLabels = processedObservations.matMul(this.weights);

    return predictedLabels;
  }
}

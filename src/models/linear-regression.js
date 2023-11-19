import { tensor, ones, zeros } from '@tensorflow/tfjs-node';

export default class LinearRegression {
  /**
   * Constructs a Linear Regression model with tensors for features, labels, and weights.
   * @param {number[][]} features - Array of feature values for training.
   * @param {number[][]} labels - Array of label values for training.
   * @param {Object} [options={}] - Options for the linear regression model.
   * @param {number} [options.learningRate=0.1] - Learning rate for gradient descent.
   * @param {number} [options.iterations=1000] - Maximum number of iterations for gradient descent.
   */
  constructor(
    features,
    labels,
    options = { learningRate: 0.1, iterations: 1000 }
  ) {
    // Convert arrays to tensors for features and labels
    this.features = tensor(features);
    this.labels = tensor(labels);

    // Add a column of ones to features for intercept calculation
    const onesColumn = ones([this.features.shape[0], 1]);
    this.features = onesColumn.concat(this.features, 1);

    this.options = options;

    // Initialize weights tensor with zeros for coefficients (m and b)
    this.weights = zeros([this.features.shape[1], 1]);
  }

  /**
   * Performs gradient descent using matrix operations to update weights.
   * Calculates gradients for weights and adjusts them using the learning rate.
   */
  gradientDescent() {
    /*
     * With the newly optimized approach employing vectorized solutions using TFJS, instead of computing
     * two gradients (slopes for MSE) with respect to 'm' and 'b' separately through distinct equations,
     * I'll adopt a more efficient method utilizing matrix multiplication. This advanced technique condenses
     * the process into a single equation, namely (Features * ((Features * Weights) - Labels)) / n.
     * Utilizing this streamlined equation, I'll compute the slopes of MSE with respect to 'm' and 'b'.
     */

    // Calculate current predictions (Features * Weights) for labels based on the current weights
    const currentPredictions = this.features.matMul(this.weights);

    // Compute differences between predicted labels and actual labels
    const differences = currentPredictions.sub(this.labels);

    // Calculate gradients (slopes of MSE) by matrix operations
    const gradients = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]); // Divide by the number of observations

    // Update weights using the calculated gradients and learning rate
    this.weights = this.weights.sub(gradients.mul(this.options.learningRate));
  }

  /**
   * Trains the linear regression model using gradient descent.
   * Executes gradient descent iteratively to optimize weights for the model.
   */
  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
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
    testFeatures = tensor(testFeatures);
    testLabels = tensor(testLabels);

    // Add a column of ones to testFeatures for intercept calculation
    const onesColumn = ones([testFeatures.shape[0], 1]);
    testFeatures = onesColumn.concat(testFeatures, 1);

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
}

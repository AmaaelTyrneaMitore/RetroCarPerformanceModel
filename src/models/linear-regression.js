import _ from 'lodash';

export default class LinearRegression {
  /**
   * Constructs a Linear Regression model with given features, labels, and options.
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
    this.features = features;
    this.labels = labels;
    this.options = options;
    this.m = 0; // Slope for the linear equation: y = mx + b
    this.b = 0; // Intercept for the linear equation: y = mx + b
  }

  /**
   * Performs gradient descent to update the slope (m) and intercept (b) values.
   * Calculates the gradients for m & b and adjusts their values using the learning rate.
   */
  gradientDescent() {
    /*
     * Both equations determining slopes regarding 'm' and 'b' (representing d(MSE)/db and d(MSE)/dm respectively)
     * incorporate the term 'mx + b,' which denotes our present prediction for MPG. To streamline calculations,
     * I'll employ a two-step process. Firstly, I'll iterate through our diverse feature variables (x) to compute 'mx + b.'
     * Subsequently, having an array of these current MPG estimations, the second step involves seamlessly constructing both equations.
     */

    // Calculate current predictions for MPG based on the current slope and intercept values
    const currentPredictionsForMPG = this.features.map((featureRow) => {
      return this.m * featureRow[0] + this.b;
    });

    // Calculate the gradient (slope) with respect to the intercept (b)
    const bSlope =
      (_.sum(
        currentPredictionsForMPG.map((currentPrediction, i) => {
          // Difference between predicted MPG and actual MPG
          return currentPrediction - this.labels[i][0];
        })
      ) *
        2) /
      this.features.length;

    // Calculate the gradient (slope) with respect to the slope (m)
    const mSlope =
      (_.sum(
        currentPredictionsForMPG.map((currentPrediction, i) => {
          // Product of feature and the difference between predicted and actual MPG
          return (
            -1 * this.features[i][0] * (this.labels[i][0] - currentPrediction)
          );
        })
      ) *
        2) /
      this.features.length;

    // Update the slope (m) and intercept (b) using the calculated gradients and learning rate
    this.m -= mSlope * this.options.learningRate;
    this.b -= bSlope * this.options.learningRate;
  }

  /**
   * Trains the linear regression model using gradient descent.
   * Executes gradient descent iteratively to find optimal values for slope and intercept.
   */
  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }
}

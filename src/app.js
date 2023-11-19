import { join } from 'path';
import appRootPath from 'app-root-path';
import plot from 'node-remote-plot';

import loadCSV from './utils/csv-loader.js';
import LinearRegression from './models/linear-regression.js';

// Load the CSV file path
const csvFilePath = join(appRootPath.path, 'data', 'vintage_cars_data.csv');

// Define options for loading CSV data
const loadOptions = {
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg'],
  shuffle: true,
  splitTest: 50,
};

// Load the CSV data using the provided options
const { features, labels, testFeatures, testLabels } = loadCSV(csvFilePath, loadOptions);

// Create a Linear Regression model instance
const regressionModel = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10,
});

// Train the regression model using gradient descent
regressionModel.train();

// Plotting MSE Values
plot({
  x: regressionModel.MSEHistory.reverse(),
  name: 'data/mse_history',
  title: 'MSE History',
  xLabel: 'No. of Iteration #',
  yLabel: 'Mean Squared Error',
});

// Making a prediction based on the trained model
const newObservations = [[120, 2, 380]]; // Example new observations for prediction
const predictedMPG = regressionModel.predict(newObservations);
predictedMPG.print();

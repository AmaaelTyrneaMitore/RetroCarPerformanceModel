import { join } from 'path';
import appRootPath from 'app-root-path';

import loadCSV from './utils/csv-loader.js';
import LinearRegression from './models/linear-regression.js';

// Load the CSV file path
const csvFilePath = join(appRootPath.path, 'data', 'vintage_cars_data.csv');

// Define options for loading CSV data
const loadOptions = {
  dataColumns: ['horsepower'],
  labelColumns: ['mpg'],
  shuffle: true,
  splitTest: 50,
};

// Load the CSV data using the provided options
const { features, labels } = loadCSV(csvFilePath, loadOptions);

// Create a Linear Regression model instance
const regression = new LinearRegression(features, labels, {
  learningRate: 0.0001,
  iterations: 100,
});

// Train the regression model using gradient descent
regression.train();

// Display the updated values of slope (m) and intercept (b) after training
const [b, m] = regression.weights.arraySync();
console.log(
  `\n\n[+] Updated value of m: ${m}\n[+] Updated value of b: ${b}\n\n`
);

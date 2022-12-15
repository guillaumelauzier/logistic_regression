#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Sigmoid function
double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

// Compute the dot product of two vectors
double dot(vector<double> x, vector<double> y) {
  double sum = 0;
  for (int i = 0; i < x.size(); i++) {
    sum += x[i] * y[i];
  }
  return sum;
}

// Compute the prediction of the logistic regression model
// given input features x and model weights w
double predict(vector<double> x, vector<double> w) {
  return sigmoid(dot(x, w));
}

// Compute the gradient of the log loss with respect to the model weights
vector<double> gradient(vector<vector<double>> X, vector<double> y, vector<double> w) {
  vector<double> grad(w.size());
  for (int i = 0; i < X.size(); i++) {
    double y_pred = predict(X[i], w);
    for (int j = 0; j < w.size(); j++) {
      grad[j] += (y[i] - y_pred) * X[i][j];
    }
  }
  return grad;
}

// Perform gradient descent to learn the model weights
void gradient_descent(vector<vector<double>> X, vector<double> y, vector<double>& w,
                      double learning_rate, int num_iters) {
  for (int i = 0; i < num_iters; i++) {
    vector<double> grad = gradient(X, y, w);
    for (int j = 0; j < w.size(); j++) {
      w[j] += learning_rate * grad[j];
    }
  }
}

int main() {
  // Define the training data
  vector<vector<double>> X = {{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}};
  vector<double> y = {0, 0, 0, 1, 1};

  // Initialize model weights
  vector<double> w = {0, 0};

  // Train the model using gradient descent
  gradient_descent(X, y, w, 0.1, 100);

  // Print the learned model weights
  cout << "w0: " << w[0] << endl;
  cout << "w1: " << w[1] << endl;

  return 0;
}

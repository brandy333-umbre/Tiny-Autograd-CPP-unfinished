#include <iostream>
#include <vector>
#include "value.hpp"

int main() {
    std::cout << "=== Demo 1: Simple graph z = x * y + tanh(x) ===\n";

    // Create leaf nodes
    auto x = make_leaf(2.0);
    auto y = make_leaf(3.0);

    // Build computation graph: z = x * y + tanh(x)
    auto xy = x * y;
    auto t  = vtanh(x);
    auto z  = xy + t;

    // Run backward pass
    backward(z);

    std::cout << "x.data = " << x->data << ", y.data = " << y->data << "\n";
    std::cout << "z.data = " << z->data << "\n";
    std::cout << "dz/dx (x.grad) = " << x->grad << "\n";
    std::cout << "dz/dy (y.grad) = " << y->grad << "\n\n";


    std::cout << "=== Demo 2: Fit y = 2x + 1 with gradient descent ===\n";

    // Tiny dataset: y = 2x + 1 with some (optional) noise
    std::vector<double> xs = { -1.0, 0.0, 1.0, 2.0, 3.0 };
    std::vector<double> ys = { -1.0, 1.0, 3.0, 5.0, 7.0 }; // exact 2x + 1

    // Parameters: w and b
    auto w = make_leaf(0.0);  // initial guess
    auto b = make_leaf(0.0);

    double lr = 0.1;
    int epochs = 50;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Build a computation graph for the loss this epoch
        ValuePtr loss = nullptr;

        for (std::size_t i = 0; i < xs.size(); ++i) {
            auto x_i = make_leaf(xs[i]);
            auto y_true = make_leaf(ys[i]);

            // y_pred = w * x_i + b
            auto y_pred = w * x_i + b;

            // squared error: (y_pred - y_true)^2
            auto error = y_pred - y_true;
            auto sq = square(error);

            if (!loss) {
                loss = sq;
            } else {
                loss = loss + sq;
            }
        }

        // Run backward to compute gradients d(loss)/d(w) and d(loss)/d(b)
        backward(loss);

        // Gradient descent update
        w->data -= lr * w->grad;
        b->data -= lr * b->grad;

        std::cout << "Epoch " << epoch
                  << " | loss = " << loss->data
                  << " | w = " << w->data
                  << " | b = " << b->data
                  << "\n";
    }

    std::cout << "\nFinal parameters:\n";
    std::cout << "w ≈ " << w->data << " (target 2.0)\n";
    std::cout << "b ≈ " << b->data << " (target 1.0)\n";

    return 0;
}

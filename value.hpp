#pragma once

#include <cmath>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

// A single scalar node in the computation graph.
struct Value {
    double data{};                             // Forward value
    double grad{};                             // Gradient d(output)/d(this)

    std::vector<Value*> parents;              // Pointers to parents in the graph
    std::function<void()> backward_fn;        // Local backward rule
};

// Shared pointer type for convenience
using ValuePtr = std::shared_ptr<Value>;

// Create a leaf node (no parents, typically an input or parameter)
inline ValuePtr make_leaf(double x) {
    auto v = std::make_shared<Value>();
    v->data = x;
    v->grad = 0.0;
    return v;
}

// ---- Computation graph utilities ----

inline void topo_sort_impl(
    Value* v,
    std::vector<Value*>& order,
    std::unordered_set<Value*>& visited
) {
    if (visited.count(v)) return;
    visited.insert(v);

    for (auto* p : v->parents) {
        topo_sort_impl(p, order, visited);
    }
    order.push_back(v);
}

inline std::vector<Value*> topo_sort(const ValuePtr& out) {
    std::vector<Value*> order;
    std::unordered_set<Value*> visited;
    topo_sort_impl(out.get(), order, visited);
    return order;
}

// Perform reverse-mode autodiff starting from scalar output `out`
inline void backward(const ValuePtr& out) {
    // 1. Build a topological ordering of the graph
    auto order = topo_sort(out);

    // 2. Reset gradients
    for (auto* v : order) {
        v->grad = 0.0;
    }

    // 3. Seed gradient at the output
    out->grad = 1.0;

    // 4. Traverse in reverse topological order and apply local backward rules
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Value* v = *it;
        if (v->backward_fn) {
            v->backward_fn();
        }
    }
}

// ---- Operations ----
// All ops:
//   - Create a new Value on the heap (ValuePtr)
//   - Set `parents`
//   - Set `backward_fn` capturing parents + out via weak_ptr
//   - Return the new node

inline ValuePtr add(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>();
    out->data = a->data + b->data;
    out->parents = { a.get(), b.get() };

    std::weak_ptr<Value> wa = a, wb = b, wout = out;
    out->backward_fn = [wa, wb, wout]() {
        auto a_ptr = wa.lock();
        auto b_ptr = wb.lock();
        auto out_ptr = wout.lock();
        if (!a_ptr || !b_ptr || !out_ptr) return;

        // dz/da = 1, dz/db = 1
        a_ptr->grad += 1.0 * out_ptr->grad;
        b_ptr->grad += 1.0 * out_ptr->grad;
    };

    return out;
}

inline ValuePtr sub(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>();
    out->data = a->data - b->data;
    out->parents = { a.get(), b.get() };

    std::weak_ptr<Value> wa = a, wb = b, wout = out;
    out->backward_fn = [wa, wb, wout]() {
        auto a_ptr = wa.lock();
        auto b_ptr = wb.lock();
        auto out_ptr = wout.lock();
        if (!a_ptr || !b_ptr || !out_ptr) return;

        // dz/da = 1, dz/db = -1
        a_ptr->grad += 1.0 * out_ptr->grad;
        b_ptr->grad -= 1.0 * out_ptr->grad;
    };

    return out;
}

inline ValuePtr mul(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>();
    out->data = a->data * b->data;
    out->parents = { a.get(), b.get() };

    std::weak_ptr<Value> wa = a, wb = b, wout = out;
    out->backward_fn = [wa, wb, wout]() {
        auto a_ptr = wa.lock();
        auto b_ptr = wb.lock();
        auto out_ptr = wout.lock();
        if (!a_ptr || !b_ptr || !out_ptr) return;

        // z = a * b
        // dz/da = b, dz/db = a
        a_ptr->grad += b_ptr->data * out_ptr->grad;
        b_ptr->grad += a_ptr->data * out_ptr->grad;
    };

    return out;
}

// Unary tanh
inline ValuePtr vtanh(const ValuePtr& a) {
    auto out = std::make_shared<Value>();
    double t = std::tanh(a->data);
    out->data = t;
    out->parents = { a.get() };

    std::weak_ptr<Value> wa = a, wout = out;
    out->backward_fn = [wa, wout]() {
        auto a_ptr = wa.lock();
        auto out_ptr = wout.lock();
        if (!a_ptr || !out_ptr) return;

        // d/dx tanh(x) = 1 - tanh(x)^2
        double dt = 1.0 - std::tanh(a_ptr->data) * std::tanh(a_ptr->data);
        a_ptr->grad += dt * out_ptr->grad;
    };

    return out;
}

// Convenience: operator overloads

inline ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) {
    return add(a, b);
}

inline ValuePtr operator-(const ValuePtr& a, const ValuePtr& b) {
    return sub(a, b);
}

inline ValuePtr operator*(const ValuePtr& a, const ValuePtr& b) {
    return mul(a, b);
}

// Scalar + Value
inline ValuePtr operator+(double x, const ValuePtr& b) {
    return add(make_leaf(x), b);
}

inline ValuePtr operator+(const ValuePtr& a, double x) {
    return add(a, make_leaf(x));
}

// Scalar * Value
inline ValuePtr operator*(double x, const ValuePtr& b) {
    return mul(make_leaf(x), b);
}

inline ValuePtr operator*(const ValuePtr& a, double x) {
    return mul(a, make_leaf(x));
}

// Helper: square
inline ValuePtr square(const ValuePtr& a) {
    return a * a;
}

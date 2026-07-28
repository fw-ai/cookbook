double horner(const double *coeffs, int degree, double x) {
    double result = coeffs[degree];
    for (int i = degree - 1; i >= 0; i--) {
        result = result * x + coeffs[i];
    }
    return result;
}

void poly_eval_batch(const double *coeffs, int degree,
                     const double *xs, double *ys, int n) {
    for (int i = 0; i < n; i++) {
        ys[i] = horner(coeffs, degree, xs[i]);
    }
}

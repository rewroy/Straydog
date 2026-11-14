#pragma once
// hyperconnection.hpp: a projection of the streams, for the units that run an expert layer after it.
void hc_project(Device& d, const Matrix& matrix, size_t steps, const float* input, float* output, Matrix* descriptors);

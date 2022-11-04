#include <algorithm>

// Generate random data
void initRandom(float *values, int N) {
    for (int i=0; i<N; i++)
    {
        // random input values
      values[i] = 0.f + 1.99f * static_cast<float>(rand()) / RAND_MAX;
    }
}

// Generate data that gives high relative speedup
void initGood(float *values, int N) {
    /* result:
        serial    : 32951 ms
        ispc      :  5217 ms   6.3x speed up
        task ispc :   472 ms  69.9x speed up
    */
    for (int i=0; i<N; i++)
    {
        // TODO: Choose data values that will yield high speedup
        values[i] = 0.f;
    }
}

// Generate data that gives low relative speedup
void initBad(float *values, int N) {
    /* result:
        serial    :  5716 ms
        ispc      :  1085 ms   5.3x speed up
        task ispc :   107 ms  53.4x speed up
    */
    for (int i=0; i<N; i++)
    {
        // TODO: Choose data values that will yield low speedup

        // This should be much slower than good/random data,
        // but it's not. I haven't figured out it.
        values[i] = N % 8 == 0 ? 1.998f : 1.f;
    }
}


#ifndef _ARGUMENT_UTILS_H_
#define _ARGUMENT_UTILS_H_

typedef struct options_struct {
  unsigned int iterations;
  char *output;
  char *input;
  unsigned int kernelIndex;
  int ret;
} OPTIONS;

OPTIONS *parse_args(int argc, char **argv);

void help(char const *exec, char const opt, char const *optarg);


// Convolutional Kernel Examples, each with dimension 3,
// gaussian kernel with dimension 5

static int sobelYKernel[] = {-1, -2, -1,
                       0,  0,  0,
                       1,  2,  1};

static int sobelXKernel[] = {-1, -0, 1,
                      -2,  0, 2,
                      -1,  0, 1};

static int laplacian1Kernel[] = { -1,  -4,  -1,
                           -4,  20,  -4,
                           -1,  -4,  -1};

static int laplacian2Kernel[] = { 0,  1,  0,
                           1, -4,  1,
                           0,  1,  0};

static int laplacian3Kernel[] = { -1,  -1,  -1,
                           -1,   8,  -1,
                           -1,  -1,  -1};

static int gaussianKernel[] = { 1,  4,  6,  4, 1,
                         4, 16, 24, 16, 4,
                         6, 24, 36, 24, 6,
                         4, 16, 24, 16, 4,
                         1,  4,  6,  4, 1 };

static char* const kernelNames[]       = { "SobelY",     "SobelX",     "Laplacian 1",    "Laplacian 2",    "Laplacian 3",    "Gaussian"     };
static int* const kernels[]            = { sobelYKernel, sobelXKernel, laplacian1Kernel, laplacian2Kernel, laplacian3Kernel, gaussianKernel };
static unsigned int const kernelDims[] = { 3,            3,            3,                3,                3,                5              };
static float const kernelFactors[]     = { 1.0,          1.0,          1.0,              1.0,              1.0,              1.0 / 256.0    };

static int const maxKernelIndex = sizeof(kernelDims) / sizeof(unsigned int);

#endif

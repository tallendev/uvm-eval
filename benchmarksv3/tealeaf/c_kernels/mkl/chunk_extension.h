#pragma once
#include "mkl.h"

typedef double* FieldBufferType;

// Empty extension point
typedef struct ChunkExtension
{
    MKL_INT* a_col_index;
    MKL_INT* a_row_index;
    double* a_non_zeros;

} ChunkExtension;

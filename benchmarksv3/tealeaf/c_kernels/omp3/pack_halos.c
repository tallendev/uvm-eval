#include <stdlib.h>
#include "../../shared.h"

// Packs left data into buffer.
void pack_left(
        const int x,
        const int y,
        const int depth,
        const int halo_depth,
        double* field,
        double* buffer)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < halo_depth+depth; ++kk)
        {
            int bufIndex = (kk-halo_depth) + (jj-halo_depth)*depth;
            buffer[bufIndex] = field[jj*x+kk];
        }
    }
}

// Packs right data into buffer.
void pack_right(
        const int x,
        const int y,
        const int depth,
        const int halo_depth,
        double* field,
        double* buffer)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = x-halo_depth-depth; kk < x-halo_depth; ++kk)
        {
            int bufIndex = (kk-(x-halo_depth-depth)) + (jj-halo_depth)*depth;
            buffer[bufIndex] = field[jj*x+kk];
        }
    }
}

// Packs top data into buffer.
void pack_top(
        const int x,
        const int y,
        const int depth,
        const int halo_depth,
        double* field,
        double* buffer)
{
    const int x_inner = x-2*halo_depth;

#pragma omp parallel for
    for(int jj = y-halo_depth-depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            int bufIndex = (kk-halo_depth) + (jj-(y-halo_depth-depth))*x_inner;
            buffer[bufIndex] = field[jj*x+kk];
        }
    }
}

// Packs bottom data into buffer.
void pack_bottom(
        const int x,
        const int y,
        const int depth,
        const int halo_depth,
        double* field,
        double* buffer)
{
    const int x_inner = x-2*halo_depth;

#pragma omp parallel for
    for(int jj = halo_depth; jj < halo_depth+depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            int bufIndex = (kk-halo_depth) + (jj-halo_depth)*x_inner;
            buffer[bufIndex] = field[jj*x+kk];
        }
    }
}

// Unpacks left data from buffer.
void unpack_left(
        const int x,
        const int y,
        const int depth,
        const int halo_depth,
        double* field,
        double* buffer)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth-depth; kk < halo_depth; ++kk)
        {
            int bufIndex = (kk-(halo_depth-depth)) + (jj-halo_depth)*depth;
            field[jj*x+kk] = buffer[bufIndex];
        }
    }
}

// Unpacks right data from buffer.
void unpack_right(
        const int x,
        const int y,
        const int depth,
        const int halo_depth,
        double* field,
        double* buffer)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = x-halo_depth; kk < x-halo_depth+depth; ++kk)
        {
            int bufIndex = (kk-(x-halo_depth)) + (jj-halo_depth)*depth;
            field[jj*x+kk] = buffer[bufIndex];
        }
    }
}

// Unpacks top data from buffer.
void unpack_top(
        const int x,
        const int y,
        const int depth,
        const int halo_depth,
        double* field,
        double* buffer)
{
    const int x_inner = x-2*halo_depth;

#pragma omp parallel for
    for(int jj = y-halo_depth; jj < y-halo_depth+depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            int bufIndex = (kk-halo_depth) + (jj-(y-halo_depth))*x_inner;
            field[jj*x+kk] = buffer[bufIndex];
        }
    }
}

// Unpacks bottom data from buffer.
void unpack_bottom(
        const int x,
        const int y,
        const int depth,
        const int halo_depth,
        double* field,
        double* buffer)
{
    const int x_inner = x-2*halo_depth;

#pragma omp parallel for
    for(int jj = halo_depth-depth; jj < halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            int bufIndex = (kk-halo_depth) + (jj-(halo_depth-depth))*x_inner;
            field[jj*x+kk] = buffer[bufIndex];
        }
    }
}

typedef void (*pack_kernel_f)(int,int,int,int,double*,double*);

// Either packs or unpacks data from/to buffers.
void pack_or_unpack(
        const int x,
        const int y,
        const int depth,
        const int halo_depth,
        const int face,
        bool pack,
        double *field,
        double* buffer)
{
    pack_kernel_f kernel = NULL;

    switch(face)
    {
        case CHUNK_LEFT:
            kernel = pack ? pack_left : unpack_left;
            break;
        case CHUNK_RIGHT:
            kernel = pack ? pack_right : unpack_right;
            break;
        case CHUNK_TOP:
            kernel = pack ? pack_top : unpack_top;
            break;
        case CHUNK_BOTTOM:
            kernel = pack ? pack_bottom : unpack_bottom;
            break;
        default:
            die(__LINE__, __FILE__, "Incorrect face provided: %d.\n", face);
    }

    kernel(x, y, depth, halo_depth, field, buffer);
}


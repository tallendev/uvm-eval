#include "Kokkos_Core.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosKernels_Handle.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_gauss_seidel.hpp"
#include "KokkosBlas1_nrm2.hpp"


#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <stdlib.h>
#include <stdio.h>



// Optional definitions
// CUDA_TALK: makes macros output error messages as provided by source (default = omitted for performance)

// Abort when error occurs (for CHECK_X() macros)
#ifdef ABORT_ON_CUDA_ERROR
#define SAFECUDA_ASSERT_ALWAYS_EXITS true
#else
#define SAFECUDA_ASSERT_ALWAYS_EXITS false
#endif

// Combination of techniques from:
// variable_args: https://stackoverflow.com/a/26408195
// zero_fix: https://gustedt.wordpress.com/2010/06/08/detect-empty-macro-arguments

// _last_one+2 will be too many arguments for the macros to handle
#define _ARG_N(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9,\
               _10, _11, _12, _13, _14, _15, _16, _17, _18, _19,\
               N, ...) N
// Universal functions (count, parens, pasting)
#define __COUNT_COMMA(...) _ARG_N(__VA_ARGS__)
#define _TRIGGER_PARENTHESIS_(...) ,
#define PASTE2(_0, _1) _PASTE2(_0, _1)
#define _PASTE2(_0, _1) _0 ## _1
#define PASTE5(_0, _1, _2, _3, _4) _PASTE5(_0, _1, _2, _3, _4)
#define _PASTE5(_0, _1, _2, _3, _4) _0 ## _1 ## _2 ## _3 ## _4
// Special comma cases for zeros
#define _COMMA_CASE_0001 0
#define _COMMA_CASE_0010 0
#define _COMMA_CASE_0100 0
#define _COMMA_CASE_1000 0
// Nonzero cases are a series of values-1 repeated #non-variadic-arguments times
#define _COMMA_CASE_0000 1
#define _COMMA_CASE_1111 2
#define _COMMA_CASE_2222 3
#define _COMMA_CASE_3333 4
// EACH __RESQ_N# MUST count backwards from maximum (same number of entries as _ARG_N)
  // All entries greater than max non-variadic arguments should be truncated
  // to the max non-variadic arugment value to simplify case-handling for variadic cases
// Max non-variadic arguments: 3
#define __RESQ_N3 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,\
                 3, 3, 3, 3, 3, 3, 3, 2, 1, 0
#define _COUNT_COMMA_N3(...) __COUNT_COMMA(__VA_ARGS__, __RESQ_N3)
#define VFUNC_NV3(func, ...) PASTE2(func, _VFUNC_NV3(__VA_ARGS__))(__VA_ARGS__)
#define _VFUNC_NV3(...) \

//Parallel Gauss-Seidel Preconditioner/Smoother
//  -Uses graph coloring to find independent row sets,
//   and applies GS to each set in parallel
//  -Here, use to solve a diagonally dominant linear system directly.

//Helper to print out colors in the shape of the grid
int main(int argc, char* argv[])
{
  using Scalar  = default_scalar;
  using Mag     = Kokkos::ArithTraits<Scalar>::mag_type;
  using Ordinal = default_lno_t;
  using Offset  = default_size_type;
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace = typename ExecSpace::memory_space;
  using Device  = Kokkos::Device<ExecSpace, MemSpace>;
  using Handle  = KokkosKernels::Experimental::
    KokkosKernelsHandle<Offset, Ordinal, default_scalar, ExecSpace, MemSpace, MemSpace>;
  using Matrix  = KokkosSparse::CrsMatrix<Scalar, Ordinal, Device, void, Offset>;
  using Vector  = typename Matrix::values_type;
  //constexpr Ordinal numRows = 10000;
  Ordinal numRows;
  if (argc == 1)
  {
    numRows = 10000;
  }
  else
  {
    numRows = atoi(argv[1]);
  }
  const Scalar one = Kokkos::ArithTraits<Scalar>::one();
  const Mag magOne = Kokkos::ArithTraits<Mag>::one();
  //Solve tolerance
  const Mag tolerance = 1e-6 * magOne;
  Kokkos::initialize();
  {
    //Generate a square, strictly diagonally dominant, but nonsymmetric matrix on which Gauss-Seidel should converge.
    //Get approx. 20 entries per row
    //Diagonals are 2x the absolute sum of all other entries.
    //Iterate until reaching the tolerance
    cudaEvent_t start;
    cudaEventCreate(&start);

    cudaEvent_t stop;
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, NULL);
    Offset nnz = numRows * 20;
    Matrix A = KokkosKernels::Impl::kk_generate_diagonally_dominant_sparse_matrix<Matrix>(numRows, numRows, nnz, 2, 100, 1.05 * one);
    std::cout << "Generated a matrix with " << numRows << " rows/cols, and " << nnz << " entries.\n";
    //Create a kernel handle, then a Gauss-Seidel handle with the default algorithm
    Handle handle;
    handle.create_gs_handle(KokkosSparse::GS_DEFAULT);
    //Set up Gauss-Seidel for the graph (matrix sparsity pattern)
    KokkosSparse::Experimental::gauss_seidel_symbolic(&handle, numRows, numRows, A.graph.row_map, A.graph.entries, false);
    //Set up Gauss-Seidel for the matrix values (numeric)
    //Another matrix with the same sparsity pattern could re-use the handle and symbolic phase, and only call numeric.
    KokkosSparse::Experimental::gauss_seidel_numeric(&handle, numRows, numRows, A.graph.row_map, A.graph.entries, A.values, false);
    //Now, preconditioner is ready to use. Set up an unknown vector (uninitialized) and randomized right-hand-side vector.
    Vector x(Kokkos::ViewAllocateWithoutInitializing("x"), numRows);
    Vector b(Kokkos::ViewAllocateWithoutInitializing("b"), numRows);
    Vector res(Kokkos::ViewAllocateWithoutInitializing("res"), numRows);
    auto bHost = Kokkos::create_mirror_view(b);
    for(Ordinal i = 0; i < numRows; i++)
      bHost(i) = 3 * ((one * rand()) / RAND_MAX);
    Kokkos::deep_copy(b, bHost);
    //Measure initial residual norm ||Ax - b||, where x is 0
    Mag initialRes = KokkosBlas::nrm2(b);
    Mag scaledResNorm = magOne;
    bool firstIter = true;
    
    
    int numIters = 0;
    while(scaledResNorm > tolerance)
    {
      //Run one sweep of forward Gauss-Seidel (SOR with omega = 1.0)
      //If this is the first iteration, tell apply:
      //  * to zero out x (it was uninitialized)
      //  * that b has changed since the previous apply (since there was no previous apply)
      KokkosSparse::Experimental::forward_sweep_gauss_seidel_apply(
          &handle, numRows, numRows,
          A.graph.row_map, A.graph.entries, A.values,
          x, b, firstIter, firstIter, one, 1);
      firstIter = false;
      //Now, compute the new residual norm using SPMV
      Kokkos::deep_copy(res, b);
      //Compute res := Ax - res (since res is now equal to b, this is Ax - b)
      KokkosSparse::spmv("N", one, A, x, -one, res);
      //Recompute the scaled norm
      scaledResNorm = KokkosBlas::nrm2(res) / initialRes;
      numIters++;
      std::cout << "Iteration " << numIters << " scaled residual norm: " << scaledResNorm << '\n';
    }
    // Record the stop event
    cudaEventRecord(stop, NULL);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);
    
    //time.stop();
    //std::cout << "time:" << time.elapsedSeconds() << std::endl;

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("perf,%f\n", msecTotal);
    std::cout << "SUCCESS: converged in " << numIters << " iterations.\n";
  }
  Kokkos::finalize();
  return 0;
}


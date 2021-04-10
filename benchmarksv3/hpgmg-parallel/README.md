HPGMG: High-performance Geometric Multigrid
===========================================

This is a heterogeneous implementation of HPGMG-FV using CUDA with Unified
Memory. The code is multi-GPU ready and uses one rank per GPU. Read [my Parallel Forall blog post](https://devblogs.nvidia.com/parallelforall/high-performance-geometric-multi-grid-gpu-acceleration/) to learn more about the implementation!

#General installation

Use build.sh script as a reference for configure and make.  Note that currently
only the finite-volume solver is enabled on GPU.  NVIDIA Kepler architecture
GPU and CUDA >= 6.0 is required to run this code.  There are ready scripts
available for ORNL Titan cluster: use build_titan.sh to compile,
finite-volume/example_jobs/job.titan to submit a job.  Default is the 4th order
scheme (fv4) using GSRB smoother.  It is possible to compile the 2nd order
(fv2) by updating local.mk and specify a different smoother by using
--fv-smoother config option (see build.sh).

# HPGMG-FV: Finite Volume solver

The finite-volume solver uses cell-centered methods with constant or variable
coefficients.  This implementation requires CUDA >= 6.0 and OpenMP and cannot
be configured at run-time.  Be sure to pass suitable NVCC and OpenMP flags.
See build.sh for recommended GPU settings.  More details about the GPU
implementation and a brief description of various options is available in the
corresponding [finite-volume readme](finite-volume/source/README).

## Running

For multi-GPU configurations it is recommended to run as many MPI ranks as you
have GPUs in your system.  Please note that if peer mappings are not available
between GPUs then the system will fall back to using zero-copy memory which can
perform very slowly.  This issue can be resolved by setting
CUDA_VISIBLE_DEVICES environment variable to constrain which GPUs are visible
for the system, or by setting CUDA_MANAGED_FORCE_DEVICE_ALLOC to a non-zero
value. See [CUDA Programming
Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory)
for more details.

Below is a sample application output from run.sh script with MAX_SOLVES=10
using single rank with 4 OMP threads and NVIDIA Tesla K20:

```
$ ./run.sh

rank 0:  Number of visible GPUs:  1
rank 0:  Selecting device 0 (Tesla K20c)


********************************************************************************
***                            HPGMG-FV Benchmark                            ***
********************************************************************************
Requested MPI_THREAD_FUNNELED, got MPI_THREAD_FUNNELED
1 MPI Tasks of 4 threads


===== Benchmark setup ==========================================================

attempting to create a 256^3 level from 8 x 128^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the GPU
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000045 seconds)
  Calculating boxes per process... target=8.000, max=8
  Creating Poisson (a=0.000000, b=1.000000) test problem
  calculating D^{-1} exactly for level h=3.906250e-03 using 64 colors...  done (3.239863 seconds)
  estimating  lambda_max... <2.223326055334546e+00

attempting to create a 128^3 level from 8 x 64^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the GPU
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000030 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 64^3 level from 8 x 32^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the GPU
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000026 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 32^3 level from 8 x 16^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the GPU
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000012 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 16^3 level from 8 x 8^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the host
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000022 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 8^3 level from 1 x 8^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the host
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000009 seconds)
  Calculating boxes per process... target=1.000, max=1

attempting to create a 4^3 level from 1 x 4^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the host
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000008 seconds)
  Calculating boxes per process... target=1.000, max=1

attempting to create a 2^3 level from 1 x 2^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the host
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000008 seconds)
  Calculating boxes per process... target=1.000, max=1

  Building restriction and interpolation lists... done

  Building MPI subcommunicator for level 1... done (0.000037 seconds)
  Building MPI subcommunicator for level 2... done (0.000011 seconds)
  Building MPI subcommunicator for level 3... done (0.000006 seconds)
  Building MPI subcommunicator for level 4... done (0.000007 seconds)
  Building MPI subcommunicator for level 5... done (0.000006 seconds)
  Building MPI subcommunicator for level 6... done (0.000005 seconds)
  Building MPI subcommunicator for level 7... done (0.000007 seconds)

  calculating D^{-1} exactly for level h=7.812500e-03 using 64 colors...  done (0.482064 seconds)
  estimating  lambda_max... <2.223332976449110e+00
  calculating D^{-1} exactly for level h=1.562500e-02 using 64 colors...  done (0.072692 seconds)
  estimating  lambda_max... <2.223387382550970e+00
  calculating D^{-1} exactly for level h=3.125000e-02 using 64 colors...  done (0.019081 seconds)
  estimating  lambda_max... <2.223793919679896e+00
  calculating D^{-1} exactly for level h=6.250000e-02 using 64 colors...  done (0.007788 seconds)
  estimating  lambda_max... <2.226274210000863e+00
  calculating D^{-1} exactly for level h=1.250000e-01 using 64 colors...  done (0.001580 seconds)
  estimating  lambda_max... <2.230456244760858e+00
  calculating D^{-1} exactly for level h=2.500000e-01 using 64 colors...  done (0.000504 seconds)
  estimating  lambda_max... <2.232895109443501e+00
  calculating D^{-1} exactly for level h=5.000000e-01 using 8 colors...  done (0.000044 seconds)
  estimating  lambda_max... <1.375886524822695e+00



===== Warming up by running 10 solves ==========================================
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.443253 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.442943 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.443094 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.441556 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.441347 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.441487 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.441615 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.441057 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.441484 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.442342 seconds)


===== Running 10 solves ========================================================
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.442112 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.441752 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.443788 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.444238 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.444546 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.445545 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.448277 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.445913 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.446068 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.445078 seconds)


===== Timing Breakdown =========================================================


level                                0            1            2            3            4            5            6            7 
level dimension                  256^3        128^3         64^3         32^3         16^3          8^3          4^3          2^3 
box dimension                    128^3         64^3         32^3         16^3          8^3          8^3          4^3          2^3        total
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
smooth                        0.000083     0.000168     0.000257     0.000345     0.002660     0.000764     0.000292     0.000000     0.004570
residual                      0.000015     0.000016     0.000023     0.000030     0.000332     0.000087     0.000032     0.000015     0.000549
applyOp                       0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000036     0.000036
BLAS1                         0.176517     0.000007     0.000013     0.000019     0.000031     0.000015     0.000010     0.000154     0.176767
BLAS3                         0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Boundary Conditions           0.000170     0.000345     0.000515     0.000698     0.000815     0.000382     0.000305     0.000080     0.003310
Restriction                   0.000016     0.000023     0.000030     0.256697     0.000029     0.000015     0.000009     0.000000     0.256819
  local restriction           0.000016     0.000023     0.000029     0.010672     0.000028     0.000014     0.000008     0.000000     0.010789
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Interpolation                 0.000016     0.000024     0.000032     0.000528     0.000348     0.000181     0.000036     0.000000     0.001164
  local interpolation         0.000016     0.000023     0.000031     0.000527     0.000347     0.000181     0.000035     0.000000     0.001159
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Ghost Zone Exchange           0.000089     0.000183     0.000281     0.000382     0.000409     0.000005     0.000005     0.000002     0.001358
  local exchange              0.000088     0.000180     0.000276     0.000375     0.000401     0.000000     0.000000     0.000000     0.001319
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
MPI_collectives               0.000002     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000013     0.000015
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
Total by level                0.176940     0.000807     0.001187     0.258741     0.004571     0.001456     0.000707     0.000296     0.444705

   Total time in MGBuild      3.122620 seconds
   Total time in MGSolve      0.444728 seconds
      number of v-cycles             1
Bottom solver iterations            14




===== Warming up by running 10 solves ==========================================
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083885 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.082871 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.082421 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.082890 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083145 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083231 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.082492 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083795 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.082478 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.080400 seconds)


===== Running 10 solves ========================================================
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.080801 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.080649 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.080445 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.080657 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.080496 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.081046 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.080536 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.080628 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.080579 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.080577 seconds)


===== Timing Breakdown =========================================================


level                                0            1            2            3            4            5            6 
level dimension                  128^3         64^3         32^3         16^3          8^3          4^3          2^3 
box dimension                     64^3         32^3         16^3          8^3          8^3          4^3          2^3        total
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
smooth                        0.000062     0.000130     0.000189     0.001406     0.000482     0.000191     0.000000     0.002460
residual                      0.000011     0.000011     0.000017     0.000165     0.000052     0.000020     0.000010     0.000284
applyOp                       0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000025     0.000025
BLAS1                         0.026066     0.000005     0.000010     0.000018     0.000010     0.000006     0.000104     0.026218
BLAS3                         0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Boundary Conditions           0.000126     0.000253     0.000380     0.000561     0.000287     0.000236     0.000059     0.001902
Restriction                   0.000013     0.000018     0.048210     0.000017     0.000011     0.000006     0.000000     0.048275
  local restriction           0.000013     0.000018     0.008114     0.000016     0.000010     0.000006     0.000000     0.008176
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Interpolation                 0.000012     0.000018     0.000324     0.000180     0.000100     0.000022     0.000000     0.000657
  local interpolation         0.000012     0.000018     0.000324     0.000180     0.000099     0.000021     0.000000     0.000654
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Ghost Zone Exchange           0.000070     0.000141     0.000214     0.000291     0.000003     0.000003     0.000001     0.000724
  local exchange              0.000069     0.000138     0.000210     0.000286     0.000000     0.000000     0.000000     0.000702
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
MPI_collectives               0.000001     0.000000     0.000000     0.000000     0.000000     0.000000     0.000009     0.000010
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
Total by level                0.026384     0.000602     0.049382     0.002604     0.000950     0.000497     0.000203     0.080622

   Total time in MGBuild      3.122620 seconds
   Total time in MGSolve      0.080638 seconds
      number of v-cycles             1
Bottom solver iterations            12




===== Warming up by running 10 solves ==========================================
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025235 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024648 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025121 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024575 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024636 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024503 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024547 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024541 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024953 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024485 seconds)


===== Running 10 solves ========================================================
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024504 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024648 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024851 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024577 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024272 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024458 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024544 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024517 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024725 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024376 seconds)


===== Timing Breakdown =========================================================


level                                0            1            2            3            4            5 
level dimension                   64^3         32^3         16^3          8^3          4^3          2^3 
box dimension                     32^3         16^3          8^3          8^3          4^3          2^3        total
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
smooth                        0.000062     0.000125     0.001051     0.000385     0.000159     0.000000     0.001782
residual                      0.000011     0.000011     0.000125     0.000042     0.000017     0.000008     0.000213
applyOp                       0.000000     0.000000     0.000000     0.000000     0.000000     0.000020     0.000020
BLAS1                         0.005481     0.000005     0.000012     0.000008     0.000005     0.000086     0.005598
BLAS3                         0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Boundary Conditions           0.000126     0.000252     0.000419     0.000229     0.000197     0.000049     0.001273
Restriction                   0.000013     0.014590     0.000014     0.000009     0.000005     0.000000     0.014630
  local restriction           0.000013     0.006021     0.000013     0.000008     0.000005     0.000000     0.006060
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Interpolation                 0.000015     0.000237     0.000159     0.000089     0.000020     0.000000     0.000520
  local interpolation         0.000014     0.000237     0.000159     0.000089     0.000019     0.000000     0.000517
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Ghost Zone Exchange           0.000068     0.000145     0.000221     0.000003     0.000003     0.000001     0.000441
  local exchange              0.000067     0.000142     0.000217     0.000000     0.000000     0.000000     0.000426
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
MPI_collectives               0.000001     0.000000     0.000000     0.000000     0.000000     0.000007     0.000008
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
Total by level                0.005802     0.015397     0.001978     0.000769     0.000416     0.000168     0.024530

   Total time in MGBuild      3.122620 seconds
   Total time in MGSolve      0.024544 seconds
      number of v-cycles             1
Bottom solver iterations            10




===== Performance Summary ======================================================
  h=3.906250000000000e-03  DOF=1.677721600000000e+07  time=0.444728  DOF/s=3.772e+07  MPI=1  OMP=4
  h=7.812500000000000e-03  DOF=2.097152000000000e+06  time=0.080638  DOF/s=2.601e+07  MPI=1  OMP=4
  h=1.562500000000000e-02  DOF=2.621440000000000e+05  time=0.024544  DOF/s=1.068e+07  MPI=1  OMP=4


===== Richardson error analysis ================================================
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.443040 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.081257 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.024723 seconds)
  h=3.906250000000000e-03  ||error||=1.486406621094630e-08
  order=3.978


===== Deallocating memory ======================================================
attempting to free the restriction and interpolation lists... done
attempting to free the     2^3 level... done
attempting to free the     4^3 level... done
attempting to free the     8^3 level... done
attempting to free the    16^3 level... done
attempting to free the    32^3 level... done
attempting to free the    64^3 level... done
attempting to free the   128^3 level... done
attempting to free the   256^3 level... done


===== Done =====================================================================
```

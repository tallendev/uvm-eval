#include "comms.h"
#include "settings.h"

#ifndef NO_MPI

// Initialise MPI
void initialise_comms(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
}

// Initialise the rank information
void initialise_ranks(Settings* settings)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &settings->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &settings->num_ranks);

  if(settings->rank == MASTER)
  {
    printf("Successfully initialised %d MPI ranks.\n", settings->num_ranks);
  }
}

// Teardown MPI
void finalise_comms()
{
  MPI_Finalize();
}

// Sends a message out and receives a message in
void send_recv_message(Settings* settings, double* send_buffer, 
    double* recv_buffer, int buffer_len, int neighbour, int send_tag, 
    int recv_tag, MPI_Request* send_request, MPI_Request* recv_request)
{
  START_PROFILING(settings->kernel_profile);

  MPI_Isend(send_buffer, buffer_len, MPI_DOUBLE, 
      neighbour, send_tag, MPI_COMM_WORLD, send_request);
  MPI_Irecv(recv_buffer, buffer_len, MPI_DOUBLE,
      neighbour, recv_tag, MPI_COMM_WORLD, recv_request);

  STOP_PROFILING(settings->kernel_profile, __func__);
}

// Waits for all requests to complete
void wait_for_requests(
    Settings* settings, int num_requests, MPI_Request* requests)
{
  START_PROFILING(settings->kernel_profile);
  MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
  STOP_PROFILING(settings->kernel_profile, __func__);
}

// Reduce over all ranks to get sum
void sum_over_ranks(Settings* settings, double* a)
{
  START_PROFILING(settings->kernel_profile);
  double temp = *a;
  MPI_Allreduce(&temp, a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  STOP_PROFILING(settings->kernel_profile, __func__);
}

// Reduce across all ranks to get minimum value
void min_over_ranks(Settings* settings, double* a)
{
  START_PROFILING(settings->kernel_profile);
  double temp = *a;
  MPI_Allreduce(&temp, a, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  STOP_PROFILING(settings->kernel_profile, __func__);
}

// Synchronise all ranks
void barrier()
{
  MPI_Barrier(MPI_COMM_WORLD);
}

// End the application
void abort_comms()
{
  MPI_Abort(MPI_COMM_WORLD, 1);
}

#else

void initialise_comms(int argc, char** argv) { }
void initialise_ranks(Settings* settings) 
{ 
  settings->rank = MASTER;
  settings->num_ranks = 1;
}
void finalise_comms() { }
void sum_over_ranks(Settings* settings, double* a) { }
void min_over_ranks(Settings* settings, double* a) { }
void barrier() { }
void abort_comms() 
{ 
  exit(1);
}

#endif

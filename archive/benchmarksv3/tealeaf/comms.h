#pragma once 
#ifndef __COMMSH
#define __COMMSH

#ifndef NO_MPI
#include <mpi.h>
#include "settings.h"
#include "chunk.h"

void barrier();
void abort_comms(); void finalise_comms();
void initialise_comms(int argc, char** argv);
void initialise_ranks(Settings* settings);
void sum_over_ranks(Settings* settings, double* a);
void min_over_ranks(Settings* settings, double* a);
void wait_for_requests(
        Settings* settings, int num_requests, MPI_Request* requests);
void send_recv_message(Settings* settings, double* send_buffer, 
        double* recv_buffer, int buffer_len, int neighbour, int send_tag, 
        int recv_tag, MPI_Request*  send_request, MPI_Request* recv_request);
#else
#include "settings.h"

void barrier();
void abort_comms();
void finalise_comms();
void initialise_comms(int argc, char** argv);
void initialise_ranks(Settings* settings);
void sum_over_ranks(Settings* settings, double* a);
void min_over_ranks(Settings* settings, double* a);

#endif 

#endif

#include "../comms.h"
#include "drivers.h"
#include "../kernel_interface.h"
#include "../chunk.h"

int invoke_pack_or_unpack(
        Chunk* chunk, Settings* settings, int face, int depth, int offset, 
        bool pack, double* buffer);

// Invokes the kernels that perform remote halo exchanges
void remote_halo_driver(
        Chunk* chunks, Settings* settings, int depth)
{
#ifndef NO_MPI
    // Two sends and two receives
    int max_messages = settings->num_chunks_per_rank*4;
    MPI_Request requests[max_messages];

    int num_messages = 0;

    // TODO: THE TAGS NEED TO BE DIFFERENT BY CHUNK ??
    
    // Pack lr buffers and send messages
    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        if(chunks[cc].neighbours[CHUNK_LEFT] != EXTERNAL_FACE)
        {
            int buffer_len = invoke_pack_or_unpack(&(chunks[cc]), settings, 
                    CHUNK_LEFT, depth, chunks[cc].y, true, 
                    chunks[cc].left_send);

            send_recv_message(settings, chunks[cc].left_send, 
                    chunks[cc].left_recv, buffer_len, 
                    chunks[cc].neighbours[CHUNK_LEFT], 0, 1,
                    &(requests[num_messages]), &(requests[num_messages+1]));

            num_messages += 2;
        }

        if(chunks[cc].neighbours[CHUNK_RIGHT] != EXTERNAL_FACE)
        {
            int buffer_len = invoke_pack_or_unpack(&(chunks[cc]), settings, 
                    CHUNK_RIGHT, depth, chunks[cc].y, true, 
                    chunks[cc].right_send);

            send_recv_message(settings, chunks[cc].right_send, 
                    chunks[cc].right_recv, buffer_len, 
                    chunks[cc].neighbours[CHUNK_RIGHT], 1, 0,
                    &(requests[num_messages]), &(requests[num_messages+1]));

            num_messages += 2;
        }
    }

    wait_for_requests(settings, num_messages, requests);

    // Unpack lr buffers
    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        if(chunks[cc].neighbours[CHUNK_LEFT] != EXTERNAL_FACE)
        {
            invoke_pack_or_unpack(&(chunks[cc]), settings, 
                    CHUNK_LEFT, depth, chunks[cc].y, false, 
                    chunks[cc].left_recv);
        }

        if(chunks[cc].neighbours[CHUNK_RIGHT] != EXTERNAL_FACE)
        {
            invoke_pack_or_unpack(&(chunks[cc]), settings, 
                    CHUNK_RIGHT, depth, chunks[cc].y, false, 
                    chunks[cc].right_recv);
        }
    }

    num_messages = 0;

    // Pack tb buffers and send messages
    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        if(chunks[cc].neighbours[CHUNK_BOTTOM] != EXTERNAL_FACE)
        {
            int buffer_len = invoke_pack_or_unpack(&(chunks[cc]), settings, 
                    CHUNK_BOTTOM, depth, chunks[cc].x, true, 
                    chunks[cc].bottom_send);

            send_recv_message(settings, chunks[cc].bottom_send, 
                    chunks[cc].bottom_recv, buffer_len, 
                    chunks[cc].neighbours[CHUNK_BOTTOM], 0, 1,
                    &(requests[num_messages]), &(requests[num_messages+1]));

            num_messages += 2;
        }

        if(chunks[cc].neighbours[CHUNK_TOP] != EXTERNAL_FACE)
        {
            int buffer_len = invoke_pack_or_unpack(&(chunks[cc]), settings, 
                    CHUNK_TOP, depth, chunks[cc].x, true, 
                    chunks[cc].top_send);

            send_recv_message(settings, chunks[cc].top_send, chunks[cc].top_recv, 
                    buffer_len, chunks[cc].neighbours[CHUNK_TOP], 1, 0,
                    &(requests[num_messages]), &(requests[num_messages+1]));

            num_messages += 2;
        }
    }

    wait_for_requests(settings, num_messages, requests);

    // Unpack tb buffers
    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        if(chunks[cc].neighbours[CHUNK_BOTTOM] != EXTERNAL_FACE)
        {
            invoke_pack_or_unpack(&(chunks[cc]), settings, 
                    CHUNK_BOTTOM, depth, chunks[cc].x, false, 
                    chunks[cc].bottom_recv);
        }

        if(chunks[cc].neighbours[CHUNK_TOP] != EXTERNAL_FACE)
        {
            invoke_pack_or_unpack(&(chunks[cc]), settings, 
                    CHUNK_TOP, depth, chunks[cc].x, false, 
                    chunks[cc].top_recv);
        }
    }

#endif
}

// Attempts to pack buffers
int invoke_pack_or_unpack(
        Chunk* chunk, Settings* settings, int face, int depth, int offset,
        bool pack, double* buffer)
{
    int buffer_len = 0;

    for(int ii = 0; ii < NUM_FIELDS; ++ii)
    {
        if(!settings->fields_to_exchange[ii]) 
        {
            continue;
        }

        FieldBufferType field;
        switch(ii)
        {
            case FIELD_DENSITY:
                field = chunk->density;
                break;
            case FIELD_ENERGY0:
                field = chunk->energy0;
                break;
            case FIELD_ENERGY1:
                field = chunk->energy;
                break;
            case FIELD_U:
                field = chunk->u;
                break;
            case FIELD_P:
                field = chunk->p;
                break;
            case FIELD_SD:
                field = chunk->sd;
                break;
            default:
                die(__LINE__,__FILE__, "Incorrect field provided: %d.\n", ii+1);
        }

        double* offset_buffer = buffer + buffer_len;
        buffer_len += depth*offset;

        if(settings->kernel_language == C)
        {
            run_pack_or_unpack(
                    chunk, settings, depth, face, pack, field, offset_buffer);
        }
        else if(settings->kernel_language == FORTRAN)
        {
        }
    }

    return buffer_len;
}


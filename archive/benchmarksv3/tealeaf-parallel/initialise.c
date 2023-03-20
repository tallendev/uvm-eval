#include <string.h>
#include <float.h>
#include <cuda_runtime.h>
#include "chunk.h"
#include "settings.h"
#include "application.h"
#include "drivers/drivers.h"

#include <map>

extern size_t UM_TOT_MEM;
extern size_t UM_MAX_MEM;
extern std::map<void*, size_t> MEMMAP;

void decompose_field(Settings* settings, Chunk* chunks);

// Initialise settings from input file
void initialise_application(Chunk** chunks, Settings* settings)
{
  State* states = NULL;
  read_config(settings, &states);

  //*chunks = (Chunk*)malloc(sizeof(Chunk)*settings->num_chunks_per_rank);
  cudaMallocManaged(chunks, sizeof(Chunk)*settings->num_chunks_per_rank);
    UM_TOT_MEM += sizeof(Chunk)*settings->num_chunks_per_rank;
    MEMMAP[chunks] = sizeof(Chunk)*settings->num_chunks_per_rank;
UM_MAX_MEM = UM_TOT_MEM > UM_MAX_MEM ? UM_TOT_MEM : UM_MAX_MEM;

  decompose_field(settings, *chunks);
  kernel_initialise_driver(*chunks, settings);
  set_chunk_data_driver(*chunks, settings);
  set_chunk_state_driver(*chunks, settings, states);

  // Prime the initial halo data
  reset_fields_to_exchange(settings);
  settings->fields_to_exchange[FIELD_DENSITY] = true;
  settings->fields_to_exchange[FIELD_ENERGY0] = true;
  settings->fields_to_exchange[FIELD_ENERGY1] = true;
  halo_update_driver(*chunks, settings, 2);

  store_energy_driver(*chunks, settings);
}

// Decomposes the field into multiple chunks
void decompose_field(Settings* settings, Chunk* chunks)
{
  // Calculates the num chunks field is to be decomposed into
  settings->num_chunks = settings->num_ranks * 
    settings->num_chunks_per_rank;

  int num_chunks = settings->num_chunks;

  double best_metric = DBL_MAX;
  double x_cells = (double)settings->grid_x_cells;
  double y_cells = (double)settings->grid_y_cells;
  int x_chunks = 0;
  int y_chunks = 0;

  // Decompose by minimal area to perimeter
  for(int xx = 1; xx <= num_chunks; ++xx)
  {
    if(num_chunks % xx) continue;

    // Calculate number of chunks grouped by x split
    int yy = num_chunks / xx;

    if(num_chunks % yy) continue;

    double perimeter = ((x_cells/xx)*(x_cells/xx) +
        (y_cells/yy)*(y_cells/yy)) * 2;
    double area = (x_cells/xx)*(y_cells/yy);

    double current_metric = perimeter / area;

    // Save improved decompositions
    if(current_metric < best_metric)
    {
      x_chunks = xx;
      y_chunks = yy;
      best_metric = current_metric;
    }
  }

  // Check that the decomposition didn't fail
  if(!x_chunks || !y_chunks)
  {
    die(__LINE__, __FILE__, 
        "Failed to decompose the field with given parameters.\n");
  }

  int dx = settings->grid_x_cells / x_chunks;
  int dy = settings->grid_y_cells / y_chunks;

  int mod_x = settings->grid_x_cells % x_chunks;
  int mod_y = settings->grid_y_cells % y_chunks;
  int add_x_prev = 0;
  int add_y_prev = 0;

  // Compute the full decomposition on all ranks
  for(int yy = 0; yy < y_chunks; ++yy)
  {
    int add_y = (yy < mod_y);

    for(int xx = 0; xx < x_chunks; ++xx)
    {
      int add_x = (xx < mod_x);

      for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
      {
        int chunk = xx + yy*x_chunks;
        int rank = cc + settings->rank*settings->num_chunks_per_rank;

        // Store the values for all chunks local to rank
        if(rank == chunk)
        {
          initialise_chunk(&(chunks[cc]), settings, dx+add_x, dy+add_y);

          // Set up the mesh ranges
          chunks[cc].left = xx*dx + add_x_prev;
          chunks[cc].right = chunks[cc].left + dx + add_x;
          chunks[cc].bottom = yy*dy + add_y_prev;
          chunks[cc].top = chunks[cc].bottom + dy + add_y;

          // Set up the chunk connectivity
          chunks[cc].neighbours[CHUNK_LEFT] = (xx == 0) 
            ? EXTERNAL_FACE 
            : chunk - 1;
          chunks[cc].neighbours[CHUNK_RIGHT] = (xx == x_chunks-1) 
            ? EXTERNAL_FACE 
            : chunk + 1;
          chunks[cc].neighbours[CHUNK_BOTTOM] = (yy == 0) 
            ? EXTERNAL_FACE : chunk - x_chunks;
          chunks[cc].neighbours[CHUNK_TOP] = (yy == y_chunks-1) 
            ? EXTERNAL_FACE 
            : chunk + x_chunks;
        }
      }

      // If chunks rounded up, maintain relative location
      add_x_prev += add_x;
    }
    add_x_prev = 0;
    add_y_prev += add_y;
  }
}

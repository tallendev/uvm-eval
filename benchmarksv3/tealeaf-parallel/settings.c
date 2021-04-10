#include "settings.h"
#include <string.h>

#define MAX_CHAR_LEN 256

void set_default_settings(Settings* settings)
{
  settings->test_problem_filename = 
    (char*)malloc(sizeof(char)*MAX_CHAR_LEN);
  strncpy(settings->test_problem_filename, 
      DEF_TEST_PROBLEM_FILENAME, MAX_CHAR_LEN);

  settings->tea_in_filename = 
    (char*)malloc(sizeof(char)*MAX_CHAR_LEN);
  strncpy(settings->tea_in_filename, 
      DEF_TEA_IN_FILENAME, MAX_CHAR_LEN);

  settings->tea_out_filename = 
    (char*)malloc(sizeof(char)*MAX_CHAR_LEN);
  strncpy(settings->tea_out_filename, 
      DEF_TEA_OUT_FILENAME, MAX_CHAR_LEN);

  settings->tea_out_fp = NULL;
  settings->grid_x_min = DEF_GRID_X_MIN;
  settings->grid_y_min = DEF_GRID_Y_MIN;
  settings->grid_x_max = DEF_GRID_X_MAX;
  settings->grid_y_max = DEF_GRID_Y_MAX;
  settings->grid_x_cells = DEF_GRID_X_CELLS;
  settings->grid_y_cells = DEF_GRID_Y_CELLS;
  settings->dt_init = DEF_DT_INIT;
  settings->max_iters = DEF_MAX_ITERS;
  settings->eps = DEF_EPS;
  settings->end_time = DEF_END_TIME;
  settings->end_step = DEF_END_STEP;
  settings->summary_frequency = DEF_SUMMARY_FREQUENCY;
  settings->solver = DEF_SOLVER;
  settings->coefficient = DEF_COEFFICIENT;
  settings->error_switch = DEF_ERROR_SWITCH;
  settings->presteps = DEF_PRESTEPS;
  settings->eps_lim = DEF_EPS_LIM;
  settings->check_result = DEF_CHECK_RESULT;
  settings->ppcg_inner_steps = DEF_PPCG_INNER_STEPS;
  settings->preconditioner = DEF_PRECONDITIONER;
  settings->num_states = DEF_NUM_STATES;
  settings->num_chunks = DEF_NUM_CHUNKS;
  settings->num_chunks_per_rank = DEF_NUM_CHUNKS_PER_RANK;
  settings->num_ranks = DEF_NUM_RANKS;
  settings->halo_depth = DEF_HALO_DEPTH;
  settings->is_offload = DEF_IS_OFFLOAD;
  settings->kernel_profile = 
    (struct Profile*)malloc(sizeof(struct Profile));
  settings->application_profile = 
    (struct Profile*)malloc(sizeof(struct Profile));
  settings->wallclock_profile = 
    (struct Profile*)malloc(sizeof(struct Profile));
  settings->fields_to_exchange =
    (bool*)malloc(sizeof(bool)*NUM_FIELDS);
  settings->solver_name = 
    (char*)malloc(sizeof(char)*MAX_CHAR_LEN);
}

// Resets all of the fields to be exchanged
void reset_fields_to_exchange(Settings* settings)
{
  for(int ii = 0; ii < NUM_FIELDS; ++ii)
  {
    settings->fields_to_exchange[ii] = false;
  }
}

// Checks if any of the fields are to be exchanged
bool is_fields_to_exchange(Settings* settings)
{
  for(int ii = 0; ii < NUM_FIELDS; ++ii)
  {
    if(settings->fields_to_exchange[ii])
    {
      return true;
    }
  }

  return false;
}


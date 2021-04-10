#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include "application.h"

int read_states(FILE* tea_in, Settings* settings, State** states);
void read_settings(FILE* tea_in, Settings* settings);
void read_value(const char* line, const char* word, char* value);
bool starts_with(const char* word, const char* line);
bool starts_get_double(const char* key, const char* line, char* word, double* value);
bool starts_get_int(const char* key, const char* line, char* word, int* value);

// Read configuration file
void read_config(Settings* settings, State** states)
{
  // Open the configuration file
  FILE* tea_in = fopen(settings->tea_in_filename, "r");
  if(!tea_in)
  {
    die(__LINE__, __FILE__, 
        "Could not open input file %s\n", 
        settings->tea_in_filename);
  }

  // Read all of the settings from the config
  read_settings(tea_in, settings);
  rewind(tea_in);

  // Read in the states
  settings->num_states = read_states(tea_in, settings, states);

  fclose(tea_in);

  initialise_log(settings);

  print_to_log(settings, "Solution Parameters:\n");
  print_to_log(settings, "\tdt_init = %f\n", settings->dt_init);
  print_to_log(settings, "\tend_time = %f\n", settings->end_time);
  print_to_log(settings, "\tend_step = %d\n", settings->end_step);
  print_to_log(settings, "\tgrid_x_min = %f\n", settings->grid_x_min);
  print_to_log(settings, "\tgrid_y_min = %f\n", settings->grid_y_min);
  print_to_log(settings, "\tgrid_x_max = %f\n", settings->grid_x_max);
  print_to_log(settings, "\tgrid_y_max = %f\n", settings->grid_y_max);
  print_to_log(settings, "\tgrid_x_cells = %d\n", settings->grid_x_cells);
  print_to_log(settings, "\tgrid_y_cells = %d\n", settings->grid_y_cells);
  print_to_log(settings, "\tpresteps = %d\n", settings->presteps);
  print_to_log(settings, "\tppcg_inner_steps = %d\n", settings->ppcg_inner_steps);
  print_to_log(settings, "\teps_lim = %f\n", settings->eps_lim);
  print_to_log(settings, "\tmax_iters = %d\n", settings->max_iters);
  print_to_log(settings, "\teps = %f\n", settings->eps);
  print_to_log(settings, "\thalo_depth = %d\n", settings->halo_depth);
  print_to_log(settings, "\tcheck_result = %d\n", settings->check_result);
  print_to_log(settings, "\tcoefficient = %d\n", settings->coefficient);
  print_to_log(settings, 
      "\tnum_chunks_per_rank = %d\n", settings->num_chunks_per_rank);
  print_to_log(settings, 
      "\tsummary_frequency = %d\n", settings->summary_frequency);

  for(int ss = 0; ss < settings->num_states; ++ss)
  {
    print_to_log(settings, "\t\nstate %d\n", ss);
    print_to_log(settings, "\tdensity = %.12E\n",  (*states)[ss].density);
    print_to_log(settings, "\tenergy= %.12E\n",  (*states)[ss].energy);
    if(ss > 0)
    {
      print_to_log(settings, "\tx_min = %.12E\n", (*states)[ss].x_min);
      print_to_log(settings, "\ty_min = %.12E\n", (*states)[ss].y_min);
      print_to_log(settings, "\tx_max = %.12E\n", (*states)[ss].x_max);
      print_to_log(settings, "\ty_max = %.12E\n", (*states)[ss].y_max);
      print_to_log(settings, "\tradius = %.12E\n", (*states)[ss].radius);
      print_to_log(settings, "\tgeometry = %d\n", (*states)[ss].geometry);
    }
  }
}

// Read all settings from the configuration file
void read_settings(FILE* tea_in, Settings* settings)
{
  size_t len = 0;
  char* line = NULL;

  // Get the number of states present in the config file
  while(getline(&line, &len, tea_in) != EOF)
  {
    char word[len];

    // Parse the key-value pairs
    if(starts_get_double("initial_timestep", line, word, &settings->dt_init))
      continue;
    if(starts_get_double("end_time", line, word, &settings->end_time))
      continue;
    if(starts_get_int("end_step", line, word, &settings->end_step))
      continue;
    if(starts_get_double("xmin", line, word, &settings->grid_x_min))
      continue;
    if(starts_get_double("ymin", line, word, &settings->grid_y_min))
      continue;
    if(starts_get_double("xmax", line, word, &settings->grid_x_max))
      continue;
    if(starts_get_double("ymax", line, word, &settings->grid_y_max))
      continue;
    if(settings->grid_x_cells == DEF_GRID_X_CELLS &&
        starts_get_int("x_cells", line, word, &settings->grid_x_cells))
      continue;
    if(settings->grid_y_cells == DEF_GRID_Y_CELLS &&
        starts_get_int("y_cells", line, word, &settings->grid_y_cells))
      continue;
    if(starts_get_int("summary_frequency", line, word, &settings->summary_frequency))
      continue;
    if(starts_get_int("presteps", line, word, &settings->presteps))
      continue;
    if(starts_get_int("ppcg_inner_steps", line, word, &settings->ppcg_inner_steps))
      continue;
    if(starts_get_double("epslim", line, word, &settings->eps_lim))
      continue;
    if(starts_get_int("max_iters", line, word, &settings->max_iters))
      continue;
    if(starts_get_double("eps", line, word, &settings->eps))
      continue;
    if(starts_get_int("num_chunks_per_rank", line, word, &settings->num_chunks_per_rank))
      continue;
    if(starts_get_int("halo_depth", line, word, &settings->halo_depth))
      continue;

    // Parse the switches
    if(starts_with("check_result", line))
    {
      settings->check_result = true;
      continue;
    }
    if(starts_with("errswitch", line))
    {
      settings->error_switch = true;
      continue;
    }
    if(starts_with("preconditioner_on", line))
    {
      settings->preconditioner = true;
      continue;
    }
    if(starts_with("use_fortran_kernels", line))
    {
      settings->kernel_language = FORTRAN;
      continue;
    }
    if(starts_with("use_c_kernels", line))
    {
      settings->kernel_language = C;
      continue;
    }
    if(starts_with("use_jacobi", line))
    {
      settings->solver = JACOBI_SOLVER;
      strcpy(settings->solver_name, "Jacobi");
      continue;
    }
    if(starts_with("use_cg", line))
    {
      settings->solver = CG_SOLVER;
      strcpy(settings->solver_name, "CG");
      continue;
    }
    if(starts_with("use_chebyshev", line))
    {
      settings->solver = CHEBY_SOLVER;
      strcpy(settings->solver_name, "Chebyshev");
      continue;
    }
    if(starts_with("use_ppcg", line))
    {
      settings->solver = PPCG_SOLVER;
      strcpy(settings->solver_name, "PPCG");
      continue;
    }
    if(starts_with("coefficient_density", line))
    {
      settings->coefficient = CONDUCTIVITY;
      continue;
    }
    if(starts_with("coefficient_inverse_density", line))
    {
      settings->coefficient = RECIP_CONDUCTIVITY;
      continue;
    }
  }

  // Set the cell widths now
  settings->dx = (settings->grid_x_max-settings->grid_x_min) / 
    (double)settings->grid_x_cells;
  settings->dy = (settings->grid_y_max-settings->grid_y_min) / 
    (double)settings->grid_y_cells;
}

// Read all of the states from the configuration file
int read_states(FILE* tea_in, Settings* settings, State** states)
{
  size_t len = 0;
  char* line = NULL;
  int num_states = 0;

  // First find the number of states
  while(getline(&line, &len, tea_in) != EOF)
  {
    int state_num = 0;
    char word[len];

    if(starts_get_int("state", line, word, &state_num))
    {
      num_states = MAX(num_states, state_num);
    }
  }

  rewind(tea_in);

  // Pre-initialise the set of states
  *states = (State*)malloc(sizeof(State)*num_states);
  for(int ss = 0; ss < num_states; ++ss)
  {
    (*states)[ss].defined = false;
  }

  // If a state boundary falls exactly on a cell boundary
  // then round off can cause the state to be put one cell
  // further than expected. This is compiler/system dependent.
  // To avoid this, a state boundary is reduced/increased by a
  // 100th of a cell width so it lies well within the intended
  // cell. Because a cell is either full or empty of a specified
  // state, this small modification to the state extents does
  // not change the answer.
  while(getline(&line, &len, tea_in) != EOF)
  {
    int state_num = 0;
    char word[len];

    // State found
    if(starts_get_int("state", line, word, &state_num))
    {
      State* state = &((*states)[state_num-1]);

      if(state->defined)
      {
        die(__LINE__, __FILE__, "State number %d defined twice.\n", state_num);
      }

      read_value(line, "density", word);
      state->density = atof(word);
      read_value(line, "energy", word);
      state->energy = atof(word);

      // State 1 is the default state so geometry irrelevant
      if(state_num > 1)
      {
        read_value(line, "xmin", word); 
        state->x_min = atof(word) + settings->dx/100.0;
        read_value(line, "ymin", word); 
        state->y_min = atof(word) + settings->dy/100.0;
        read_value(line, "xmax", word); 
        state->x_max = atof(word) - settings->dx/100.0;
        read_value(line, "ymax", word); 
        state->y_max = atof(word) - settings->dy/100.0;

        read_value(line, "geometry", word);

        if(strmatch(word, "rectangle"))
        {
          state->geometry = RECTANGULAR;
        }
        else if(strmatch(word, "circular"))
        {
          state->geometry = CIRCULAR;

          read_value(line, "radius", word); 
          state->radius = atof(word);
        }
        else if(strmatch(word, "point"))
        {
          state->geometry = POINT;
        }
      }

      state->defined = true;
    }
  }

  return num_states;
}

// Checks line starts with word
bool starts_with(const char* word, const char* line)
{
  int num_matched = 0;
  int word_len = strlen(word);

  for(int ll = 0; ll < (int)strlen(line); ++ll)
  {
    // Skip leading spaces
    if(!num_matched && isspace(line[ll]))
    {
      continue;
    }

    // Match the word
    if(line[ll] != word[num_matched])
    {
      return false;
    }
    else if(++num_matched == word_len)
    {
      return true;
    }
  }

  return false;
}

// Parses key-value pairs for state in configuration file
void read_value(const char* line, const char* word, char* value)
{
  int num_matched = 0;

  // Step through the line, find the token and parse value
  for(int ll = 0; ll < (int)strlen(line); ++ll)
  {
    if(num_matched == (int)strlen(word))
    {
      // Now match value, nest for correctness
      if(isalpha(line[ll]) || isdigit(line[ll]))
      {
        sscanf(&(line[ll]), "%s", value);
        return;
      }
    }
    else
    {
      num_matched = (line[ll] == word[num_matched]) ? num_matched+1 : 0;
    }
  }

  die(__LINE__, __FILE__, "Failed to find a value for key '%s'\n", word);
}

// Gets key value pair by checking that the line starts with key and getting value
bool starts_get_int(const char* key, const char* line, char* word, int* value)
{
  if(starts_with(key, line))
  {
    read_value(line, key, word);
    *value = atoi(word);
    return true;
  }

  return false;
}

// Gets key value pair by checking that the line starts with key and getting value
bool starts_get_double(const char* key, const char* line, char* word, double* value)
{
  if(starts_with(key, line))
  {
    read_value(line, key, word);
    *value = atof(word);
    return true;
  }

  return false;
}


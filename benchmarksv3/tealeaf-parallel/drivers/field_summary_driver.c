#include "../comms.h"
#include "../kernel_interface.h"
#include "../chunk.h"

void get_checking_value(Settings* settings, double* checking_value);

// Invokes the set chunk data kernel
void field_summary_driver(
    Chunk* chunks, Settings* settings, bool is_solve_finished)
{
  double vol = 0.0;
  double ie = 0.0;
  double temp = 0.0;
  double mass = 0.0;

  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    if(settings->kernel_language == C)
    {
      run_field_summary(&(chunks[cc]), settings, &vol, &mass, &ie, &temp);
    }
    else if(settings->kernel_language == FORTRAN)
    {
    }
  }

  // Bring all of the results to the master
  sum_over_ranks(settings, &vol);
  sum_over_ranks(settings, &mass);
  sum_over_ranks(settings, &ie);
  sum_over_ranks(settings, &temp);

  if(settings->rank == MASTER && settings->check_result && is_solve_finished)
  {
    print_and_log(settings, "\nChecking results...\n");

    double checking_value = 1.0;
    get_checking_value(settings, &checking_value);

    print_and_log(settings, "Expected %.15e\n", checking_value);
    print_and_log(settings, "Actual   %.15e\n", temp);

    double qa_diff = fabs(100.0*(temp/checking_value)-100.0);
    if(qa_diff < 0.001)
    {
      print_and_log(settings, "This run \033[32mPASSED\033[0m");
    }
    else
    {
      print_and_log(settings, "This run \033[31mFAILED\033[0m");
    }
    print_and_log(settings, " (Difference is within %.8lf%)\n", qa_diff); 
  }
}

// Fetches the checking value from the test problems file
void get_checking_value(Settings* settings, double* checking_value)
{
  FILE* test_problem_file = fopen(settings->test_problem_filename, "r");

  if(!test_problem_file)
  {
    print_and_log(settings,
        "\nWARNING: Could not open the test problem file.\n");
  }

  size_t len = 0;
  char* line = NULL;

  // Get the number of states present in the config file
  while(getline(&line, &len, test_problem_file) != EOF)
  {
    int x;
    int y;
    int num_steps;

    sscanf(line, "%d %d %d %lf", &x, &y, &num_steps, checking_value);

    // Found the problem in the file
    if(x == settings->grid_x_cells && y == settings->grid_y_cells &&
        num_steps == settings->end_step)
    {
      fclose(test_problem_file);
      return;
    }
  }

  *checking_value = 1.0;
  print_and_log(settings, 
      "\nWARNING: Problem was not found in the test problems file.\n");
  fclose(test_problem_file);
}

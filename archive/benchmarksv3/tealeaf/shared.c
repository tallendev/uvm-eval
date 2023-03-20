#include "comms.h"
#include "shared.h"

// Initialises the log file pointer
void initialise_log(
    struct Settings* settings)
{
  // Only write to log in master rank
  if(settings->rank != MASTER)
  {
    return;
  }

  printf("Opening %s as log file.\n", settings->tea_out_filename);
  settings->tea_out_fp = fopen(settings->tea_out_filename, "w");

  if(!settings->tea_out_fp)
  {
    die(__LINE__, __FILE__,
        "Could not open log %s\n", settings->tea_out_filename);
  }
}

// Prints to stdout and then logs message in log file
void print_and_log(
    struct Settings* settings, const char* format, ...)
{
  // Only master rank should print
  if(settings->rank != MASTER)
  {
    return;
  }

  va_list arglist;
  va_start(arglist, format);
  vprintf(format, arglist);
  va_end(arglist);

  if(!settings->tea_out_fp)
  {
    die(__LINE__, __FILE__, 
        "Attempted to write to log before it was initialised\n");
  }

  // Obtuse, but necessary
  va_list arglist2;
  va_start(arglist2, format);
  vfprintf(settings->tea_out_fp, format, arglist2);
  va_end(arglist2);
}

// Logs message in log file
void print_to_log(
    struct Settings* settings, const char* format, ...)
{
  // Only master rank should log
  if(settings->rank != MASTER)
  {
    return;
  }

  if(!settings->tea_out_fp)
  {
    die(__LINE__, __FILE__, 
        "Attempted to write to log before it was initialised\n");
  }

  va_list arglist;
  va_start(arglist, format);
  vfprintf(settings->tea_out_fp, format, arglist);
  va_end(arglist);
}

// Plots a two-dimensional dat file.
void plot_2d(int x, int y, double* buffer, const char* name)
{    
  // Open the plot file
  FILE* fp = fopen("plot2d.dat", "wb");
  if(!fp) { printf("Could not open plot file.\n"); }

  double b_sum = 0.0;

  for(int jj = 0; jj < y; ++jj)
  {
    for(int kk = 0; kk < x; ++kk)
    {
      double val = buffer[kk+jj*x];
      fprintf(fp, "%d %d %.12E\n", kk, jj, val);
      b_sum+=val;
    }
  }

  printf("%s: %.12E\n", name, b_sum);
  fclose(fp);
}

// Aborts the application.
void die(int lineNum, const char* file, const char* format, ...)
{
  // Print location of error
  printf("\x1b[31m");
  printf("\nError at line %d in %s:", lineNum, file);
  printf("\x1b[0m \n");

  va_list arglist;
  va_start(arglist, format);
  vprintf(format, arglist);
  va_end(arglist);

  abort_comms();
}

// Write out data for visualisation in visit
void write_to_visit(
    const int nx, const int ny, const int x_off, const int y_off, 
    const double* data, const char* name, const int step, const double time)
{
  char bovname[256];
  char datname[256];
  sprintf(bovname, "%s%d.bov", name, step);
  sprintf(datname, "%s%d.dat", name, step);

  FILE* bovfp = fopen(bovname, "w");

  if(!bovfp) {
    printf("Could not open file %s\n", bovname);
    exit(1);
  }

  fprintf(bovfp, "TIME: %.4f\n", time);
  fprintf(bovfp, "DATA_FILE: %s\n", datname);
  fprintf(bovfp, "DATA_SIZE: %d %d 1\n", nx, ny);
  fprintf(bovfp, "DATA_FORMAT: DOUBLE\n");
  fprintf(bovfp, "VARIABLE: density\n");
  fprintf(bovfp, "DATA_ENDIAN: LITTLE\n");
  fprintf(bovfp, "CENTERING: zone\n");
  fprintf(bovfp, "BRICK_ORIGIN: 0. 0. 0.\n");

  fprintf(bovfp, "BRICK_SIZE: %d %d 1\n", nx, ny);
  fclose(bovfp);

  FILE* datfp = fopen(datname, "wb");
  if(!datfp) {
    printf("Could not open file %s\n", datname);
    exit(1);
  }

  fwrite(data, sizeof(double), nx*ny, datfp);
  fclose(datfp);
}


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "profiler.h"

#define strmatch(a, b) (strcmp(a, b) == 0)

// Internally start the profiling timer
void profiler_start_timer(struct Profile* profile)
{
#ifdef __APPLE__
  profile->profiler_start = mach_absolute_time();
#else
  clock_gettime(CLOCK_MONOTONIC, &profile->profiler_start);
#endif
}

// Internally end the profiling timer and store results
void profiler_end_timer(struct Profile* profile, const char* entry_name)
{
#ifdef __APPLE__
  profile->profiler_end = mach_absolute_time();
#else
  clock_gettime(CLOCK_MONOTONIC, &profile->profiler_end);
#endif

  // Check if an entry exists
  int ii;
  for(ii = 0; ii < profile->profiler_entry_count; ++ii)
  {
    if(strmatch(profile->profiler_entries[ii].name, entry_name))
    {
      break;
    }
  }

  // Don't overrun
  if(ii >= PROFILER_MAX_ENTRIES)
  {
    printf("Attempted to profile too many entries, maximum is %d\n",
        PROFILER_MAX_ENTRIES);
    exit(1);
  }

  // Create new entry
  if(ii == profile->profiler_entry_count)
  {
    profile->profiler_entry_count++;
    strcpy(profile->profiler_entries[ii].name, entry_name);
  }

  // Update number of calls and time
#ifdef __APPLE__
  double elapsed = (profile->profiler_end-profile->profiler_start)*1.0E-9;
#else
  double elapsed = 
    (profile->profiler_end.tv_sec - profile->profiler_start.tv_sec) + 
    (profile->profiler_end.tv_nsec - profile->profiler_start.tv_nsec)*1.0E-9;
#endif

  profile->profiler_entries[ii].time += elapsed; 
  profile->profiler_entries[ii].calls++;
}

// Print the profiling results to output
void profiler_print_full_profile(struct Profile* profile)
{
  printf("\n-------------------------------------------------------------\n");
  printf("\nProfiling Results:\n\n");
  printf("%-30s%8s%20s\n", "Kernel Name", "Calls", "Runtime (s)");

  double total_elapsed_time = 0.0;
  for(int ii = 0; ii < profile->profiler_entry_count; ++ii)
  {
    total_elapsed_time += profile->profiler_entries[ii].time;
    printf("%-30s%8d%20.03F\n", profile->profiler_entries[ii].name, 
        profile->profiler_entries[ii].calls, 
        profile->profiler_entries[ii].time);
  }

  printf("\nTotal elapsed time: %.03Fs, entries * are excluded.\n", total_elapsed_time);
  printf("\n-------------------------------------------------------------\n\n");
}

// Prints profile without extra details
void profiler_print_simple_profile(struct Profile* profile)
{
  for(int ii = 0; ii < profile->profiler_entry_count; ++ii)
  {
    printf("\033[1m\033[30m%s\033[0m: %.3lfs (%d calls)\n", 
        profile->profiler_entries[ii].name, 
        profile->profiler_entries[ii].time,
        profile->profiler_entries[ii].calls);
  }
}

// Gets an individual profile entry
int profiler_get_profile_entry(struct Profile* profile, const char* entry_name)
{
  for(int ii = 0; ii < profile->profiler_entry_count; ++ii)
  {
    if(strmatch(profile->profiler_entries[ii].name, entry_name))
    {
      return ii;
    }
  }

  printf("Attempted to retrieve missing profile entry %s\n",
      entry_name);
  exit(1);
}


#pragma once 

#include "chunk.h"

void initialise_application(Chunk** chunks, Settings* settings);
void diffuse(Chunk* chunk, Settings* settings);
void read_config(Settings* settings, State** states);

#ifdef DIFFUSE_OVERLOAD
void diffuse_overload(Chunk* chunk, Settings* settings);
#endif

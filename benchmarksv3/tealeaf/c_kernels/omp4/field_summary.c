#include "../../shared.h"

/*
 * 		FIELD SUMMARY KERNEL
 * 		Calculates aggregates of values in field.
 */	

// The field summary kernel
void field_summary(
        const int x,
        const int y,
        const int halo_depth,
        double* volume,
        double* density,
        double* energy0,
        double* u,
        double* volOut,
        double* massOut,
        double* ieOut,
        double* tempOut)
{
    double vol = 0.0;
    double ie = 0.0;
    double temp = 0.0;
    double mass = 0.0;

    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            double cellVol = volume[index];
            double cellMass = cellVol*density[index];
            vol += cellVol;
            mass += cellMass;
            ie += cellMass*energy0[index];
            temp += cellMass*u[index];
        }
    }

    *volOut += vol;
    *ieOut += ie;
    *tempOut += temp;
    *massOut += mass;
}

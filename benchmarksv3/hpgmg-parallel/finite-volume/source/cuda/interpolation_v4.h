/*
 * # Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 * #
 * # Redistribution and use in source and binary forms, with or without
 * # modification, are permitted provided that the following conditions
 * # are met:
 * #  * Redistributions of source code must retain the above copyright
 * #    notice, this list of conditions and the following disclaimer.
 * #  * Redistributions in binary form must reproduce the above copyright
 * #    notice, this list of conditions and the following disclaimer in the
 * #    documentation and/or other materials provided with the distribution.
 * #  * Neither the name of NVIDIA CORPORATION nor the names of its
 * #    contributors may be used to endorse or promote products derived
 * #    from this software without specific prior written permission.
 * #
 * # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * */

#undef  READ
#define READ(i)	__ldg(&read[i])

template<int log_dim, int block_type>
__global__ void interpolation_v4_kernel(level_type level_f, int id_f, double prescale_f, level_type level_c, int id_c, communicator_type interpolation){
  // one CUDA thread block operates on one HPGMG tile/block
  blockCopy_type block = interpolation.blocks[block_type][blockIdx.z];

  // interpolate 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int write_dim_i   = block.dim.i<<1; // calculate the dimensions of the resultant fine block
  int write_dim_j   = block.dim.j<<1;
  int write_dim_k   = block.dim.k<<1;

  int i = (blockIdx.x*blockDim.x + threadIdx.x);
  int j = (blockIdx.y*blockDim.y + threadIdx.y)*2;
  if(i>=write_dim_i || j>=write_dim_j) return;

  int  read_i       = block.read.i;
  int  read_j       = block.read.j;
  int  read_k       = block.read.k;
  int  read_jStride = block.read.jStride;
  int  read_kStride = block.read.kStride;

  int write_i       = block.write.i;
  int write_j       = block.write.j;
  int write_k       = block.write.k;
  int write_jStride = block.write.jStride;
  int write_kStride = block.write.kStride;

  double * __restrict__  read = block.read.ptr;
  double * __restrict__ write = block.write.ptr;
  if(block.read.box >=0){
     read_jStride = level_c.my_boxes[block.read.box ].jStride;
     read_kStride = level_c.my_boxes[block.read.box ].kStride;
     read = level_c.my_boxes[ block.read.box].vectors[id_c] + level_c.my_boxes[ block.read.box].ghosts*(1+ read_jStride+ read_kStride);
  }
  if(block.write.box>=0){
    write_jStride = level_f.my_boxes[block.write.box].jStride;
    write_kStride = level_f.my_boxes[block.write.box].kStride;
    write = level_f.my_boxes[block.write.box].vectors[id_f] + level_f.my_boxes[block.write.box].ghosts*(1+write_jStride+write_kStride);
  }
 

//  #ifdef USE_NAIVE_INTERP
  // naive 125pt per fine grid cell
  //int i,j,k;
  double c2 = -3.0/128.0;
  double c1 = 22.0/128.0;
  int dj  =   read_jStride;
  int dk  =   read_kStride;
  int dj2 = 2*read_jStride;
  int dk2 = 2*read_kStride;
  for(int k=0;k<write_dim_k;k+=2){
    double sk1=c1,sk2=c2;//if(k&0x1){sk1=-c1;sk2=-c2;}
    double sj1=c1,sj2=c2;//if(j&0x1){sj1=-c1;sj2=-c2;}
    double si1=c1,si2=c2;if(i&0x1){si1=-c1;si2=-c2;}

    int write_ijk = ((i   )+write_i) + (((j   )+write_j)*write_jStride) + (((k   )+write_k)*write_kStride);
    int  read_ijk = ((i>>1)+ read_i) + (((j>>1)+ read_j)* read_jStride) + (((k>>1)+ read_k)* read_kStride);
    //
    // |   -3/128  |  +22/128  |    1.0    |  -22/128  |   +3/128  | coarse grid
    // |-----+-----|-----+-----|-----+-----|-----+-----|-----+-----|
    // |     |     |     |     |?????|     |     |     |     |     | fine grid
    //

    double r1  = si2*READ(read_ijk-2-dj2-dk2) + si1*READ(read_ijk-1-dj2-dk2) + READ(read_ijk-dj2-dk2) - si1*READ(read_ijk+1-dj2-dk2) - si2*READ(read_ijk+2-dj2-dk2);
    double r2  = si2*READ(read_ijk-2-dj -dk2) + si1*READ(read_ijk-1-dj -dk2) + READ(read_ijk-dj -dk2) - si1*READ(read_ijk+1-dj -dk2) - si2*READ(read_ijk+2-dj -dk2);
    double r3  = si2*READ(read_ijk-2    -dk2) + si1*READ(read_ijk-1    -dk2) + READ(read_ijk    -dk2) - si1*READ(read_ijk+1    -dk2) - si2*READ(read_ijk+2    -dk2);
    double r4  = si2*READ(read_ijk-2+dj -dk2) + si1*READ(read_ijk-1+dj -dk2) + READ(read_ijk+dj -dk2) - si1*READ(read_ijk+1+dj -dk2) - si2*READ(read_ijk+2+dj -dk2);
    double r5  = si2*READ(read_ijk-2+dj2-dk2) + si1*READ(read_ijk-1+dj2-dk2) + READ(read_ijk+dj2-dk2) - si1*READ(read_ijk+1+dj2-dk2) - si2*READ(read_ijk+2+dj2-dk2);
    double r6  = si2*READ(read_ijk-2-dj2-dk ) + si1*READ(read_ijk-1-dj2-dk ) + READ(read_ijk-dj2-dk ) - si1*READ(read_ijk+1-dj2-dk ) - si2*READ(read_ijk+2-dj2-dk );
    double r7  = si2*READ(read_ijk-2-dj -dk ) + si1*READ(read_ijk-1-dj -dk ) + READ(read_ijk-dj -dk ) - si1*READ(read_ijk+1-dj -dk ) - si2*READ(read_ijk+2-dj -dk );
    double r8  = si2*READ(read_ijk-2    -dk ) + si1*READ(read_ijk-1    -dk ) + READ(read_ijk    -dk ) - si1*READ(read_ijk+1    -dk ) - si2*READ(read_ijk+2    -dk );
    double r9  = si2*READ(read_ijk-2+dj -dk ) + si1*READ(read_ijk-1+dj -dk ) + READ(read_ijk+dj -dk ) - si1*READ(read_ijk+1+dj -dk ) - si2*READ(read_ijk+2+dj -dk );
    double r10 = si2*READ(read_ijk-2+dj2-dk ) + si1*READ(read_ijk-1+dj2-dk ) + READ(read_ijk+dj2-dk ) - si1*READ(read_ijk+1+dj2-dk ) - si2*READ(read_ijk+2+dj2-dk );
    double r11 = si2*READ(read_ijk-2-dj2    ) + si1*READ(read_ijk-1-dj2    ) + READ(read_ijk-dj2    ) - si1*READ(read_ijk+1-dj2    ) - si2*READ(read_ijk+2-dj2    );
    double r12 = si2*READ(read_ijk-2-dj     ) + si1*READ(read_ijk-1-dj     ) + READ(read_ijk-dj     ) - si1*READ(read_ijk+1-dj     ) - si2*READ(read_ijk+2-dj     );
    double r13 = si2*READ(read_ijk-2        ) + si1*READ(read_ijk-1        ) + READ(read_ijk        ) - si1*READ(read_ijk+1        ) - si2*READ(read_ijk+2        );
    double r14 = si2*READ(read_ijk-2+dj     ) + si1*READ(read_ijk-1+dj     ) + READ(read_ijk+dj     ) - si1*READ(read_ijk+1+dj     ) - si2*READ(read_ijk+2+dj     );
    double r15 = si2*READ(read_ijk-2+dj2    ) + si1*READ(read_ijk-1+dj2    ) + READ(read_ijk+dj2    ) - si1*READ(read_ijk+1+dj2    ) - si2*READ(read_ijk+2+dj2    );
    double r16 = si2*READ(read_ijk-2-dj2+dk ) + si1*READ(read_ijk-1-dj2+dk ) + READ(read_ijk-dj2+dk ) - si1*READ(read_ijk+1-dj2+dk ) - si2*READ(read_ijk+2-dj2+dk );
    double r17 = si2*READ(read_ijk-2-dj +dk ) + si1*READ(read_ijk-1-dj +dk ) + READ(read_ijk-dj +dk ) - si1*READ(read_ijk+1-dj +dk ) - si2*READ(read_ijk+2-dj +dk );
    double r18 = si2*READ(read_ijk-2    +dk ) + si1*READ(read_ijk-1    +dk ) + READ(read_ijk    +dk ) - si1*READ(read_ijk+1    +dk ) - si2*READ(read_ijk+2    +dk );
    double r19 = si2*READ(read_ijk-2+dj +dk ) + si1*READ(read_ijk-1+dj +dk ) + READ(read_ijk+dj +dk ) - si1*READ(read_ijk+1+dj +dk ) - si2*READ(read_ijk+2+dj +dk );
    double r20 = si2*READ(read_ijk-2+dj2+dk ) + si1*READ(read_ijk-1+dj2+dk ) + READ(read_ijk+dj2+dk ) - si1*READ(read_ijk+1+dj2+dk ) - si2*READ(read_ijk+2+dj2+dk );
    double r21 = si2*READ(read_ijk-2-dj2+dk2) + si1*READ(read_ijk-1-dj2+dk2) + READ(read_ijk-dj2+dk2) - si1*READ(read_ijk+1-dj2+dk2) - si2*READ(read_ijk+2-dj2+dk2);
    double r22 = si2*READ(read_ijk-2-dj +dk2) + si1*READ(read_ijk-1-dj +dk2) + READ(read_ijk-dj +dk2) - si1*READ(read_ijk+1-dj +dk2) - si2*READ(read_ijk+2-dj +dk2);
    double r23 = si2*READ(read_ijk-2    +dk2) + si1*READ(read_ijk-1    +dk2) + READ(read_ijk    +dk2) - si1*READ(read_ijk+1    +dk2) - si2*READ(read_ijk+2    +dk2);
    double r24 = si2*READ(read_ijk-2+dj +dk2) + si1*READ(read_ijk-1+dj +dk2) + READ(read_ijk+dj +dk2) - si1*READ(read_ijk+1+dj +dk2) - si2*READ(read_ijk+2+dj +dk2);
    double r25 = si2*READ(read_ijk-2+dj2+dk2) + si1*READ(read_ijk-1+dj2+dk2) + READ(read_ijk+dj2+dk2) - si1*READ(read_ijk+1+dj2+dk2) - si2*READ(read_ijk+2+dj2+dk2);

    // i  j  k
    write[write_ijk] = prescale_f*write[write_ijk] +
                       + sk2*( + sj2*r1  + sj1*r2  + r3  - sj1*r4  - sj2*r5  )
                       + sk1*( + sj2*r6  + sj1*r7  + r8  - sj1*r9  - sj2*r10 )
                       +     ( + sj2*r11 + sj1*r12 + r13 - sj1*r14 - sj2*r15 )
                       - sk1*( + sj2*r16 + sj1*r17 + r18 - sj1*r19 - sj2*r20 )
                       - sk2*( + sj2*r21 + sj1*r22 + r23 - sj1*r24 - sj2*r25 );

    // i  j  k+1
    write_ijk = ((i  )+write_i) + (((j  )+write_j)*write_jStride) + (((k+1)+write_k)*write_kStride);
    sk1=-c1;sk2=-c2;
    sj1=c1,sj2=c2;
    write[write_ijk] = prescale_f*write[write_ijk] +
                       + sk2*( + sj2*r1  + sj1*r2  + r3  - sj1*r4  - sj2*r5  )
                       + sk1*( + sj2*r6  + sj1*r7  + r8  - sj1*r9  - sj2*r10 )
                       +     ( + sj2*r11 + sj1*r12 + r13 - sj1*r14 - sj2*r15 )
                       - sk1*( + sj2*r16 + sj1*r17 + r18 - sj1*r19 - sj2*r20 )
                       - sk2*( + sj2*r21 + sj1*r22 + r23 - sj1*r24 - sj2*r25 );

    // i  j+1  k
    write_ijk = ((i  )+write_i) + (((j+1)+write_j)*write_jStride) + (((k  )+write_k)*write_kStride);
    sk1=c1;sk2=c2;
    sj1=-c1,sj2=-c2;
    write[write_ijk] = prescale_f*write[write_ijk] +
                       + sk2*( + sj2*r1  + sj1*r2  + r3  - sj1*r4  - sj2*r5  )
                       + sk1*( + sj2*r6  + sj1*r7  + r8  - sj1*r9  - sj2*r10 )
                       +     ( + sj2*r11 + sj1*r12 + r13 - sj1*r14 - sj2*r15 )
                       - sk1*( + sj2*r16 + sj1*r17 + r18 - sj1*r19 - sj2*r20 )
                       - sk2*( + sj2*r21 + sj1*r22 + r23 - sj1*r24 - sj2*r25 );

    // i  j+1  k+1
    write_ijk = ((i  )+write_i) + (((j+1)+write_j)*write_jStride) + (((k+1)+write_k)*write_kStride);
    sk1=-c1;sk2=-c2;
    sj1=-c1,sj2=-c2;
    write[write_ijk] = prescale_f*write[write_ijk] +
                       + sk2*( + sj2*r1  + sj1*r2  + r3  - sj1*r4  - sj2*r5  )
                       + sk1*( + sj2*r6  + sj1*r7  + r8  - sj1*r9  - sj2*r10 )
                       +     ( + sj2*r11 + sj1*r12 + r13 - sj1*r14 - sj2*r15 )
                       - sk1*( + sj2*r16 + sj1*r17 + r18 - sj1*r19 - sj2*r20 )
                       - sk2*( + sj2*r21 + sj1*r22 + r23 - sj1*r24 - sj2*r25 );


    /*write[write_ijk] = prescale_f*write[write_ijk] +
                       + sk2*( + sj2*( r1 )
                               + sj1*( r2 )
                               +     ( r3 )
                               - sj1*( r4 )
                               - sj2*( r5 ) )
                       + sk1*( + sj2*( r6 )
                               + sj1*( r7 )
                               +     ( r8 )
                               - sj1*( r9 )
                               - sj2*( r10 ) )
                       +     ( + sj2*( r11 )
                               + sj1*( r12 )
                               +     ( r13 )
                               - sj1*( r14 )
                               - sj2*( r15 ) )
                       - sk1*( + sj2*( r16 )
                               + sj1*( r17 )
                               +     ( r18 )
                               - sj1*( r19 )
                               - sj2*( r20 ) )
                       - sk2*( + sj2*( r21 )
                               + sj1*( r22 )
                               +     ( r23 )
                               - sj1*( r24 )
                               - sj2*( r25 ) );*/

    /*write[write_ijk] = prescale_f*write[write_ijk] +
                       + sk2*( + sj2*( si2*READ(read_ijk-2-dj2-dk2) + si1*READ(read_ijk-1-dj2-dk2) + READ(read_ijk-dj2-dk2) - si1*READ(read_ijk+1-dj2-dk2) - si2*READ(read_ijk+2-dj2-dk2) )
                               + sj1*( si2*READ(read_ijk-2-dj -dk2) + si1*READ(read_ijk-1-dj -dk2) + READ(read_ijk-dj -dk2) - si1*READ(read_ijk+1-dj -dk2) - si2*READ(read_ijk+2-dj -dk2) )
                               +     ( si2*READ(read_ijk-2    -dk2) + si1*READ(read_ijk-1    -dk2) + READ(read_ijk    -dk2) - si1*READ(read_ijk+1    -dk2) - si2*READ(read_ijk+2    -dk2) )
                               - sj1*( si2*READ(read_ijk-2+dj -dk2) + si1*READ(read_ijk-1+dj -dk2) + READ(read_ijk+dj -dk2) - si1*READ(read_ijk+1+dj -dk2) - si2*READ(read_ijk+2+dj -dk2) )
                               - sj2*( si2*READ(read_ijk-2+dj2-dk2) + si1*READ(read_ijk-1+dj2-dk2) + READ(read_ijk+dj2-dk2) - si1*READ(read_ijk+1+dj2-dk2) - si2*READ(read_ijk+2+dj2-dk2) ) )
                       + sk1*( + sj2*( si2*READ(read_ijk-2-dj2-dk ) + si1*READ(read_ijk-1-dj2-dk ) + READ(read_ijk-dj2-dk ) - si1*READ(read_ijk+1-dj2-dk ) - si2*READ(read_ijk+2-dj2-dk ) )
                               + sj1*( si2*READ(read_ijk-2-dj -dk ) + si1*READ(read_ijk-1-dj -dk ) + READ(read_ijk-dj -dk ) - si1*READ(read_ijk+1-dj -dk ) - si2*READ(read_ijk+2-dj -dk ) )
                               +     ( si2*READ(read_ijk-2    -dk ) + si1*READ(read_ijk-1    -dk ) + READ(read_ijk    -dk ) - si1*READ(read_ijk+1    -dk ) - si2*READ(read_ijk+2    -dk ) )
                               - sj1*( si2*READ(read_ijk-2+dj -dk ) + si1*READ(read_ijk-1+dj -dk ) + READ(read_ijk+dj -dk ) - si1*READ(read_ijk+1+dj -dk ) - si2*READ(read_ijk+2+dj -dk ) )
                               - sj2*( si2*READ(read_ijk-2+dj2-dk ) + si1*READ(read_ijk-1+dj2-dk ) + READ(read_ijk+dj2-dk ) - si1*READ(read_ijk+1+dj2-dk ) - si2*READ(read_ijk+2+dj2-dk ) ) )
                       +     ( + sj2*( si2*READ(read_ijk-2-dj2    ) + si1*READ(read_ijk-1-dj2    ) + READ(read_ijk-dj2    ) - si1*READ(read_ijk+1-dj2    ) - si2*READ(read_ijk+2-dj2    ) )
                               + sj1*( si2*READ(read_ijk-2-dj     ) + si1*READ(read_ijk-1-dj     ) + READ(read_ijk-dj     ) - si1*READ(read_ijk+1-dj     ) - si2*READ(read_ijk+2-dj     ) )
                               +     ( si2*READ(read_ijk-2        ) + si1*READ(read_ijk-1        ) + READ(read_ijk        ) - si1*READ(read_ijk+1        ) - si2*READ(read_ijk+2        ) )
                               - sj1*( si2*READ(read_ijk-2+dj     ) + si1*READ(read_ijk-1+dj     ) + READ(read_ijk+dj     ) - si1*READ(read_ijk+1+dj     ) - si2*READ(read_ijk+2+dj     ) )
                               - sj2*( si2*READ(read_ijk-2+dj2    ) + si1*READ(read_ijk-1+dj2    ) + READ(read_ijk+dj2    ) - si1*READ(read_ijk+1+dj2    ) - si2*READ(read_ijk+2+dj2    ) ) )
                       - sk1*( + sj2*( si2*READ(read_ijk-2-dj2+dk ) + si1*READ(read_ijk-1-dj2+dk ) + READ(read_ijk-dj2+dk ) - si1*READ(read_ijk+1-dj2+dk ) - si2*READ(read_ijk+2-dj2+dk ) )
                               + sj1*( si2*READ(read_ijk-2-dj +dk ) + si1*READ(read_ijk-1-dj +dk ) + READ(read_ijk-dj +dk ) - si1*READ(read_ijk+1-dj +dk ) - si2*READ(read_ijk+2-dj +dk ) )
                               +     ( si2*READ(read_ijk-2    +dk ) + si1*READ(read_ijk-1    +dk ) + READ(read_ijk    +dk ) - si1*READ(read_ijk+1    +dk ) - si2*READ(read_ijk+2    +dk ) )
                               - sj1*( si2*READ(read_ijk-2+dj +dk ) + si1*READ(read_ijk-1+dj +dk ) + READ(read_ijk+dj +dk ) - si1*READ(read_ijk+1+dj +dk ) - si2*READ(read_ijk+2+dj +dk ) )
                               - sj2*( si2*READ(read_ijk-2+dj2+dk ) + si1*READ(read_ijk-1+dj2+dk ) + READ(read_ijk+dj2+dk ) - si1*READ(read_ijk+1+dj2+dk ) - si2*READ(read_ijk+2+dj2+dk ) ) )
                       - sk2*( + sj2*( si2*READ(read_ijk-2-dj2+dk2) + si1*READ(read_ijk-1-dj2+dk2) + READ(read_ijk-dj2+dk2) - si1*READ(read_ijk+1-dj2+dk2) - si2*READ(read_ijk+2-dj2+dk2) )
                               + sj1*( si2*READ(read_ijk-2-dj +dk2) + si1*READ(read_ijk-1-dj +dk2) + READ(read_ijk-dj +dk2) - si1*READ(read_ijk+1-dj +dk2) - si2*READ(read_ijk+2-dj +dk2) )
                               +     ( si2*READ(read_ijk-2    +dk2) + si1*READ(read_ijk-1    +dk2) + READ(read_ijk    +dk2) - si1*READ(read_ijk+1    +dk2) - si2*READ(read_ijk+2    +dk2) )
                               - sj1*( si2*READ(read_ijk-2+dj +dk2) + si1*READ(read_ijk-1+dj +dk2) + READ(read_ijk+dj +dk2) - si1*READ(read_ijk+1+dj +dk2) - si2*READ(read_ijk+2+dj +dk2) )
                               - sj2*( si2*READ(read_ijk-2+dj2+dk2) + si1*READ(read_ijk-1+dj2+dk2) + READ(read_ijk+dj2+dk2) - si1*READ(read_ijk+1+dj2+dk2) - si2*READ(read_ijk+2+dj2+dk2) ) );*/
  }
}
#undef  KERNEL
#define KERNEL(log_dim, block_type) \
  interpolation_v4_kernel<log_dim,block_type><<<grid,block>>>(level_f,id_f,prescale_f,level_c,id_c,interpolation);

extern "C"
void cuda_interpolation_v4(level_type level_f, int id_f, double prescale_f, level_type level_c, int id_c, communicator_type interpolation, int block_type)
{
  int num_blocks = interpolation.num_blocks[block_type]; if(num_blocks<=0) return;
  dim3 block = dim3(min(level_f.box_dim,BLOCKCOPY_TILE_I), BLOCKCOPY_TILE_J, 1);
  dim3 grid = dim3((level_f.box_dim+block.x-1)/block.x, 1, num_blocks);

  int log_dim = (int)log2((double)level_f.dim.i);
  switch(block_type){
    case 0: KERNEL_LEVEL(log_dim,0); CUDA_ERROR break;
    case 1: KERNEL_LEVEL(log_dim,1); CUDA_ERROR break;
    default: printf("CUDA ERROR: incorrect block type, %i\n", block_type);
  }
}

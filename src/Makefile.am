CXXLD=$(NVCC)
AM_CPPFLAGS=$(mumps_CPPFLAGS)
AM_LDFLAGS=$(mumps_LIBS) $(magma_LIBS) $(lapack_LIBS) $(cuda_LIBS) -lm

CPPFLAGS+=-pedantic -W -Wall -O2
NVCC_CPPFLAGS=-I. -I../include $(cuda_CPPFLAGS) $(magma_CPPFLAGS)

bin_PROGRAMS = bspline_2d

bspline_2d_SOURCES = main.cc mumps_solver.cc kernels.cu

.cu.o:
	$(NVCC) $(NVCCFLAGS) $(NVCC_CPPFLAGS) -c -o $@ $<

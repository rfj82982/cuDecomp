test_fft_c2c_x: test_fft_c2c_x.f90
	${MPIF90} -cuda -cpp ${BUILD_INCLUDES} -o $@ $< ${BUILD_FLIBS}

clean:
	rm -f test_fft_c2c_x

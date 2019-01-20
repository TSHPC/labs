all: cpu omp cuda cuda_opt

cpu:
	gcc -O3 -o matrix_cpu matrix_cpu.c

omp:
	gcc -O3 -fopenmp -o matrix_cpu_omp matrix_cpu_omp.c

cuda:
	nvcc -O3 -o matrix_cuda matrix_cuda.cu -lcuda -Xcompiler -fopenmp

cuda_opt:
	nvcc -O3 -o matrix_cuda_opt matrix_cuda_opt.cu -lcuda -Xcompiler -fopenmp

clean:
	rm -rf matrix_cpu matrix_cpu_omp matrix_cuda matrix_cuda_opt
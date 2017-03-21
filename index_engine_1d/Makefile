
# the compiler: gcc for C program, define as g++ for C++
CC = gcc

# compiler flags:
#  -Wall turns on most, but not all, compiler warnings
CFLAGS_OPENCL = -Wall -lglut -lGL -lm -O3 -fopenmp -lOpenCL -fno-stack-protector 
CFLAGS_OPENMP = -Wall -lglut -lGL -lm -O3 -fopenmp -fno-stack-protector 
CFLAGS_ORIG = -Wall -lglut -lGL -lm -O3 -fno-stack-protector 

# the build target executable:
TARGET_OPENCL = parallel
TARGET_OPENMP = parallel_openmp
TARGET_ORIG = parallel_orig


all:
	$(RM) $(TARGET_OPENCL)
	$(RM) $(TARGET_OPENMP)
	$(RM) $(TARGET_OPENCL)
	$(CC) -std=c99 -o $(TARGET_OPENCL) $(TARGET_OPENCL).c $(CFLAGS_OPENCL)
	$(CC) -std=c99 -o $(TARGET_OPENMP) $(TARGET_OPENMP).c $(CFLAGS_OPENMP)
	$(CC) -std=c99 -o $(TARGET_ORIG) $(TARGET_ORIG).c $(CFLAGS_ORIG)
	
clean:
	$(RM) $(TARGET_OPENCL)
	$(RM) $(TARGET_OPENMP)
	$(RM) $(TARGET_OPENCL)





	


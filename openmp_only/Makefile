
# the compiler: gcc for C program, define as g++ for C++
CC = gcc

# compiler flags:
#  -Wall turns on most, but not all, compiler warnings
CFLAGS = -Wall -lglut -lGL -lm -O3 -fopenmp

# the build target executable:
TARGET = parallel_openmp

all:
	$(CC) -std=c99 -o $(TARGET) $(TARGET).c $(CFLAGS)

run:
	./$(TARGET)

clean:
	$(RM) $(TARGET)
	
hitandrun: clean all	run

	


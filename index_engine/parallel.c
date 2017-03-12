/* TIE-51257 Parallelization Excercise 2017
   Copyright (c) 2016 Matias Koskela matias.koskela@tut.fi
                      Heikki Kultala heikki.kultala@tut.fi
*/

// Example compilation on linux
// no optimization:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -fno-stack-protector
// full optimization: gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O3 -fno-stack-protector
// prev and OpenMP:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O3 -fopenmp -fno-stack-protector
// prev and OpenCL:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O3 -fopenmp -lOpenCL -fno-stack-protector
 
// Example compilation on macos X
// no optimization:   gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL
// full optimization: gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL -O3

#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h> // printf
#include <math.h> // INFINI TY
#include <stdlib.h> 

//own variables
#include <sys/time.h>  
//#include <cmath>
#include <CL/cl.h>
#define __CL_ENABLE_EXCEPTIONS

// Window handling includes
#ifndef __APPLE__
#include <GL/gl.h>
#include <GL/glut.h>
#else
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#endif
// These are used to decide the window size
#define WINDOW_HEIGHT 1024
#define WINDOW_WIDTH  1024

// The number of satelites can be changed to see how it affects performance
#define SATELITE_COUNT 35


#define LOCAL_ITEM_SIZE_X 2
#define LOCAL_ITEM_SIZE_Y 2

// These are used to control the satelite movement
#define SATELITE_RADIUS 3.16f
#define MAX_VELOCITY 0.1f
#define GRAVITY 1.0f

// Some helpers to window size variables
#define SIZE WINDOW_HEIGHT*WINDOW_HEIGHT
#define HORIZONTAL_CENTER (WINDOW_WIDTH / 2)
#define VERTICAL_CENTER (WINDOW_HEIGHT / 2)

//Create OpenCL constant variables by using macro
#define TEXTIFY(A) #A
#define _OPTION_CREATOR(WIDTH, HEIGHT, RAD, CNT)	\
				" -D WINDOW_WIDTH=" TEXTIFY(WIDTH) 		\
				" -D WINDOW_HEIGHT=" TEXTIFY(HEIGHT)	\
 				" -D SAT_RADIUS=" TEXTIFY(RAD)  			\
				" -D SAT_COUNT=" TEXTIFY(CNT)
#define CL_OPTIONS _OPTION_CREATOR(WINDOW_WIDTH, WINDOW_HEIGHT, SATELITE_RADIUS, SATELITE_COUNT)

// Is used to find out frame times
int previousFrameTimeSinceStart = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;

// Stores 2D data like the coordinates
typedef struct{
   float x;
   float y;
} vector;

// Stores rendered colors. Each float may vary from 0.0f ... 1.0f
typedef struct{
   float red;
   float green;
   float blue;
} color;

// Stores the satelite data, which fly around black hole in the space
typedef struct{
   color identifier;
   vector position;
   vector velocity;
} satelite;


typedef struct DeviceDesc{
	cl_device_id    deviceId;
	cl_device_type  deviceType;
	char*           deviceTypeString;
	char*           deviceName;
} DeviceDesc;

typedef struct ClDevice{
	cl_mem satelite_id_gpu;
	cl_mem satelite_data_gpu;
	cl_mem pixels_gpu;
	cl_mem pixel_start_offset_y;
	cl_program program;
	cl_kernel kernel;
	cl_command_queue command_queue;
	cl_context context;
	cl_device_id device_id;
   cl_platform_id platform_id;
   cl_device_type type;
   int global_start_x;
   int global_stop_x;
   int global_start_y;
   int global_stop_y;
   size_t *global_size;
   size_t *local_size;
   cl_event evnt;
   char* pixel_ids;
   int pixel_arr_size;
   
} ClDevice;

// Pixel buffer which is rendered to the screen
color* pixels;

// Pixel buffer which is used for error checking
color* correctPixels;

// Buffer for all satelites in the space
satelite* satelites;

//All found and accepted OpenCL devices
ClDevice* cl_devices;
int num_of_cldevices = 0;


// ## You may add your own variables here ##

#define MAX_CL_DEVICES 5
#define MAX_SOURCE_SIZE (0x100000)

#define DEBUG_FILENAME "file.csv"

float render_avg = 55.0f;
size_t *global_size = NULL;
size_t *local_size = NULL;

int itemsize_x = 0;
int itemsize_y = 1;
int testloops = 100; //Init this to switch interval

//char* pixel_ids = NULL;

int best_frame_time = 99999;
int best_coloring_time = 9999;
int best_moving_time = 9999;
float coloring_avg = 999.99f;
double coloring_avg_cpu = 9999999.99f;
double coloring_avg_gpu = 9999999.99f;
double memory_avg_cpu = 9999999.99f;
double memory_avg_gpu = 9999999.99f;




const char *getErrorString(cl_int error, int device_number){
	int len;
	clGetProgramBuildInfo(cl_devices[device_number].program, cl_devices[device_number].device_id, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
	char *log = (char*)malloc(sizeof(char)*len);
	clGetProgramBuildInfo(cl_devices[device_number].program, cl_devices[device_number].device_id, CL_PROGRAM_BUILD_LOG, len, log, NULL);
	fprintf(stderr, log);
	free(log);
	
	switch(error){
		 // run-time and JIT compiler errors
		 case 0: return "CL_SUCCESS";
		 case -1: return "CL_DEVICE_NOT_FOUND";
		 case -2: return "CL_DEVICE_NOT_AVAILABLE";
		 case -3: return "CL_COMPILER_NOT_AVAILABLE";
		 case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		 case -5: return "CL_OUT_OF_RESOURCES";
		 case -6: return "CL_OUT_OF_HOST_MEMORY";
		 case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		 case -8: return "CL_MEM_COPY_OVERLAP";
		 case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		 case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		 case -11: return "CL_BUILD_PROGRAM_FAILURE";
		 case -12: return "CL_MAP_FAILURE";
		 case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		 case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		 case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		 case -16: return "CL_LINKER_NOT_AVAILABLE";
		 case -17: return "CL_LINK_PROGRAM_FAILURE";
		 case -18: return "CL_DEVICE_PARTITION_FAILED";
		 case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		 // compile-time errors
		 case -30: return "CL_INVALID_VALUE";
		 case -31: return "CL_INVALID_DEVICE_TYPE";
		 case -32: return "CL_INVALID_PLATFORM";
		 case -33: return "CL_INVALID_DEVICE";
		 case -34: return "CL_INVALID_CONTEXT";
		 case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		 case -36: return "CL_INVALID_COMMAND_QUEUE";
		 case -37: return "CL_INVALID_HOST_PTR";
		 case -38: return "CL_INVALID_MEM_OBJECT";
		 case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		 case -40: return "CL_INVALID_IMAGE_SIZE";
		 case -41: return "CL_INVALID_SAMPLER";
		 case -42: return "CL_INVALID_BINARY";
		 case -43: return "CL_INVALID_BUILD_OPTIONS";
		 case -44: return "CL_INVALID_PROGRAM";
		 case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		 case -46: return "CL_INVALID_KERNEL_NAME";
		 case -47: return "CL_INVALID_KERNEL_DEFINITION";
		 case -48: return "CL_INVALID_KERNEL";
		 case -49: return "CL_INVALID_ARG_INDEX";
		 case -50: return "CL_INVALID_ARG_VALUE";
		 case -51: return "CL_INVALID_ARG_SIZE";
		 case -52: return "CL_INVALID_KERNEL_ARGS";
		 case -53: return "CL_INVALID_WORK_DIMENSION";
		 case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		 case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		 case -56: return "CL_INVALID_GLOBAL_OFFSET";
		 case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		 case -58: return "CL_INVALID_EVENT";
		 case -59: return "CL_INVALID_OPERATION";
		 case -60: return "CL_INVALID_GL_OBJECT";
		 case -61: return "CL_INVALID_BUFFER_SIZE";
		 case -62: return "CL_INVALID_MIP_LEVEL";
		 case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		 case -64: return "CL_INVALID_PROPERTY";
		 case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		 case -66: return "CL_INVALID_COMPILER_OPTIONS";
		 case -67: return "CL_INVALID_LINKER_OPTIONS";
		 case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		 // extension errors
		 case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		 case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		 case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		 case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		 case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		 case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		 default: return "Unknown OpenCL error";
    }
}

/*
	Error handler for CL_ Calls
*/
__attribute__((always_inline)) void checkAndHandleErr(cl_int ret, int device, char *additional, int linenum){
	if (ret != CL_SUCCESS){
		fprintf(stderr, "Called from linenumber:%d\n",linenum);
		fprintf(stderr, additional);
		fprintf(stderr, getErrorString(ret, device));
		exit(0);
	}
}

int get_platforms_and_devices(ClDevice *CLdevices){
	int              i;
	cl_int              maxDevices = 5;
	cl_device_id*       deviceIDs = (cl_device_id*)malloc(maxDevices*sizeof(cl_device_id));
	cl_platform_id*     platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id));
	cl_int              err;
	cl_uint             num_entries = 2;
	cl_uint             available;
	cl_uint             numDevices;
	DeviceDesc*         devices;

	int end_devices = 0;
	cl_int result = clGetPlatformIDs(num_entries, platforms, &available);
	for(int j= 0; j<available; j++){
		err = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, maxDevices, deviceIDs, &numDevices);

		devices = (DeviceDesc*)malloc(numDevices*sizeof(DeviceDesc));

		for(i=0 ; i<numDevices ; i++)		{
			devices[i].deviceId = deviceIDs[i];
			size_t actualSize;

			//Getting the device type (processor, graphics card, accelerator)
			result = clGetDeviceInfo(
				deviceIDs[i], 
				CL_DEVICE_TYPE, 
				sizeof(cl_device_type), 
				&devices[i].deviceType, 
				&actualSize);

			//Getting the human readable device type
			switch(devices[i].deviceType)	{
				 case CL_DEVICE_TYPE_CPU:               
					devices[i].deviceTypeString = "Processor"; 
					break;
				 case CL_DEVICE_TYPE_GPU:               
					devices[i].deviceTypeString = "Graphics card"; 
					break;
				 case CL_DEVICE_TYPE_ACCELERATOR:       
					devices[i].deviceTypeString = "Accelerator"; 
					break;
				 default:                               
					devices[i].deviceTypeString = "NONE"; 
					break;
			}

			//Getting the device name
			size_t deviceNameLength = 4096;
			char* tempDeviceName = (char*)malloc(4096);
			result |= clGetDeviceInfo(
				deviceIDs[i], 
				CL_DEVICE_NAME, 
				deviceNameLength, 
				tempDeviceName, 
				&actualSize);
			if(result == CL_SUCCESS){
				devices[i].deviceName = (char*)malloc(actualSize);
				memcpy(devices[i].deviceName, tempDeviceName, actualSize);
				free(tempDeviceName);
			}
			//If an error occured
			if(result != CL_SUCCESS){
				printf("Error while getting device info\n");
				return 0;
			}
		}

		//And finally we print the information we wanted to have
		for(i=0 ; i<numDevices ; i++){
			if (devices[i].deviceType == CL_DEVICE_TYPE_GPU){
				printf("Device %s is of type %s\n", devices[i].deviceName, devices[i].deviceTypeString);
				CLdevices[end_devices].device_id = deviceIDs[i];
				CLdevices[end_devices].platform_id = platforms[j];
				CLdevices[end_devices].type = devices[i].deviceType;
				num_of_cldevices ++;
				end_devices ++;
			}
		}
	}
	free(deviceIDs);
	free(platforms);
	return end_devices;
}

// ## You may add your own initialization routines here ##
void init(){
	
	FILE *fp2 = fopen(DEBUG_FILENAME, "w");
	if (fp2 == NULL)	{
		 printf("Error opening file!\n");
		 exit(1);
	}
	fprintf(fp2, "TotalFrameTime;moving;coloring;average;color_cpu_part;color_gpu_part;mem_cpu;mem_gpu;x_size;y_size;comment\n");
	fclose(fp2);
	fprintf(stdout, "init starts\n");
	// Load the kernel source code into the array source_str
	
	fprintf(stdout, "Reading openCL kernel from file\n");
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("parallel_graphics_engine.cl", "r");
	if (!fp) {
	  fprintf(stderr, "Failed to load kernel.\n");
	  exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
	
	//Allocate memory for possible CL devices
	cl_devices = (ClDevice*) malloc(sizeof(ClDevice)*MAX_CL_DEVICES);

	int found_devices = get_platforms_and_devices(cl_devices);
	printf("found_devices:%d\n", found_devices);

	cl_int ret = CL_SUCCESS;
	//Share workload equally between devices
	for (int i=0;i<found_devices;i++){
		fprintf(stdout, "\nInitializin device, type: ");
		switch(cl_devices[i].type)	{
			 case CL_DEVICE_TYPE_CPU:               
				fprintf(stdout,"Processor\n"); 
				break;
			 case CL_DEVICE_TYPE_GPU:               
				fprintf(stdout,"Graphics card\n");
				break;
			 case CL_DEVICE_TYPE_ACCELERATOR:      
				fprintf(stdout,"Accelerator\n");
				break;
			 default:                              
				fprintf(stdout,"NONE\n");
				break;
		}
		int division_ratio = 64; //FIXME, now stupid 2 device hardcoded ratio
		if (found_devices == 1){
			cl_devices[i].global_start_x = 0;
			cl_devices[i].global_stop_x = WINDOW_WIDTH;
			cl_devices[i].global_start_y = 0;
			cl_devices[i].global_stop_y = WINDOW_HEIGHT;
			cl_devices[i].pixel_ids = (char*)malloc(sizeof(char) * (SIZE)); //TODO fix this to be only this device render area. now we are using x. times too much memory
			cl_devices[i].pixel_arr_size = (SIZE);
		}
		else{	
			if (i==1){
				cl_devices[i].global_start_x = 0;
				cl_devices[i].global_stop_x = WINDOW_WIDTH;
				cl_devices[i].global_start_y = (WINDOW_HEIGHT/division_ratio)*0;
				cl_devices[i].global_stop_y = (WINDOW_HEIGHT/division_ratio)*(1);
				cl_devices[i].pixel_ids = (char*)malloc(sizeof(char) * (SIZE/division_ratio*(1))); //TODO fix this to be only this device render area. now we are using x. times too much memory
				cl_devices[i].pixel_arr_size = (SIZE/division_ratio)*(1);
			}
			else{
				cl_devices[i].global_start_x = 0;
				cl_devices[i].global_stop_x = WINDOW_WIDTH;
				cl_devices[i].global_start_y = (WINDOW_HEIGHT/division_ratio)*1;
				cl_devices[i].global_stop_y = (WINDOW_HEIGHT/division_ratio)*(division_ratio);
				cl_devices[i].pixel_ids = (char*)malloc(sizeof(char) * (SIZE/division_ratio)*(division_ratio-1)); //TODO fix this to be only this device render area. now we are using x. times too much memory
				cl_devices[i].pixel_arr_size = (SIZE/division_ratio)*(division_ratio-1);
			}
		}
		
		//Allocate memory for global and local item sizes
		cl_devices[i].global_size = (size_t*) malloc(sizeof(size_t)*2);
		cl_devices[i].local_size = (size_t*) malloc(sizeof(size_t)*2);	
		if (cl_devices[i].global_size == NULL || cl_devices[i].local_size == NULL){
			fprintf(stderr, "memory allocation failed\n");
			exit(1);
		}

		// Create an OpenCL context
		cl_devices[i].context = clCreateContext( NULL, 1, &cl_devices[i].device_id, NULL, NULL, &ret);
		checkAndHandleErr(ret, i, "ERROR clCreateContext\n",__LINE__);

		// Create a command queue
		cl_devices[i].command_queue = clCreateCommandQueue(cl_devices[i].context, cl_devices[i].device_id, 0, &ret);
		checkAndHandleErr(ret, i, "ERROR clCreateCommandQueue\n",__LINE__);
		
		
		fprintf(stdout, "Creating OpenCL buffers\n");
		
		cl_devices[i].satelite_data_gpu = clCreateBuffer(cl_devices[i].context, CL_MEM_READ_ONLY,  SATELITE_COUNT * sizeof(satelite), NULL, &ret);
		checkAndHandleErr(ret, i, "ERROR clCreateBuffer satelite_data_gpu\n",__LINE__);
		fprintf(stdout, "satelite_data_gpu size:%ld\n",SATELITE_COUNT * sizeof(satelite));
		fprintf(stdout, "satelite_data_gpu id:%d\n",cl_devices[i].satelite_data_gpu);
	
	
		cl_devices[i].satelite_id_gpu = clCreateBuffer(cl_devices[i].context, CL_MEM_WRITE_ONLY,  sizeof(char)*cl_devices[i].pixel_arr_size, NULL, &ret);
		checkAndHandleErr(ret, i, "ERROR clCreateBuffer satelite_id_gpu\n",__LINE__);
		fprintf(stdout, "pixel_arr_size size:%ld\n", sizeof(char)*cl_devices[i].pixel_arr_size);
		fprintf(stdout, "pixel_arr_size id:%d\n",cl_devices[i].satelite_id_gpu);
	
	
		cl_devices[i].pixel_start_offset_y = clCreateBuffer(cl_devices[i].context, CL_MEM_READ_ONLY,  sizeof(int), NULL, &ret);
		checkAndHandleErr(ret, i, "ERROR clCreateBuffer pixel_start_offset_y\n",__LINE__);

	
		// Create a program from the kernel source
		cl_devices[i].program = clCreateProgramWithSource(cl_devices[i].context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
		checkAndHandleErr(ret, i, "ERROR clCreateProgramWithSource\n",__LINE__);

		// Build the program
		char *opts = CL_OPTIONS;
		printf("CL-kernel options: %s\n", CL_OPTIONS);

		fprintf(stdout, "Bulding OpenCL kernel\n");
		ret = clBuildProgram(cl_devices[i].program, 1, &cl_devices[i].device_id, opts, NULL, NULL);
		checkAndHandleErr(ret, i, "ERROR clBuildProgram\n", __LINE__);

		// Create the OpenCL kernel
		cl_devices[i].kernel = clCreateKernel(cl_devices[i].program, "render", &ret);
		checkAndHandleErr(ret, i, "ERROR clCreateKernel\n", __LINE__);
		

	
		fprintf(stdout, "Setting kernel argument locations\n");
		
		ret = clSetKernelArg(cl_devices[i].kernel, 0, sizeof(cl_mem), (void *)&cl_devices[i].satelite_data_gpu);
		checkAndHandleErr(ret, i, "ERROR clSetKernelArg 0 \n", __LINE__);
	
		ret = clSetKernelArg(cl_devices[i].kernel, 1, sizeof(cl_mem), (void *)&cl_devices[i].satelite_id_gpu);
		checkAndHandleErr(ret, i, "ERROR clSetKernelArg 1\n", __LINE__);
		
		ret = clSetKernelArg(cl_devices[i].kernel, 2, sizeof(cl_mem), (void *)&cl_devices[i].pixel_start_offset_y);
		checkAndHandleErr(ret, i, "ERROR clSetKernelArg 2\n", __LINE__);
		
		//Set pixel offsets for different devices
		
		fprintf(stdout, "cl_devices[i].global_start_y:%d\n",cl_devices[i].global_start_y);
		ret = clEnqueueWriteBuffer(cl_devices[i].command_queue, cl_devices[i].pixel_start_offset_y, CL_TRUE, 0, sizeof(int), &cl_devices[i].global_start_y, 0, NULL, NULL);
		checkAndHandleErr(ret, i, "ERROR clEnqueueWriteBuffer\n", __LINE__);
	}
	fprintf(stdout, "init ends\n");
}


// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine) 
// Moves the satelites based on gravity
// This is done multiple times in a frame because the Euler integration 
// is not accurate enough to be done only once
void parallelPhysicsEngine(int deltaTime){
   const int physicsUpdatesInOneFrame = 10000;
	#pragma omp parallel for
   for(int i = 0; i < SATELITE_COUNT; ++i){
      // Distance to the blackhole (bit ugly code because C-struct cannot have member functions)
      vector positionToBlackHole = {.x = satelites[i].position.x -
         HORIZONTAL_CENTER, .y = satelites[i].position.y - VERTICAL_CENTER};
      float distToBlackHoleSquared = 
         positionToBlackHole.x * positionToBlackHole.x +
         positionToBlackHole.y * positionToBlackHole.y;
      float distToBlackHole = sqrt(distToBlackHoleSquared);

      // Gravity force
      vector normalizedDirection = { 
         .x = positionToBlackHole.x / distToBlackHole,
         .y = positionToBlackHole.y / distToBlackHole};
      float accumulation = GRAVITY / distToBlackHoleSquared;

      // Delta time is used to make velocity same despite different FPS
      // Update velocity based on force
      for(int physicsUpdateIndex = 0; physicsUpdateIndex < physicsUpdatesInOneFrame; ++physicsUpdateIndex){
	      satelites[i].velocity.x -= accumulation * normalizedDirection.x * deltaTime / physicsUpdatesInOneFrame;
	      satelites[i].velocity.y -= accumulation * normalizedDirection.y * deltaTime / physicsUpdatesInOneFrame;

	      // Update position based on velocity
	      satelites[i].position.x = satelites[i].position.x + satelites[i].velocity.x * deltaTime / physicsUpdatesInOneFrame;
	      satelites[i].position.y = satelites[i].position.y + satelites[i].velocity.y * deltaTime / physicsUpdatesInOneFrame;
      }
   }
}

// ## You are asked to make this code parallel ##
// Rendering loop (This is called once a frame after physics engine) 
// Decides the color for each pixel.
void parallelGraphicsEngine(){
	struct timeval t1, t2,t3_sub1,t3_sub2, t3, t4, t5, t6, t4_sub1, t4_sub2;
	gettimeofday(&t1, NULL);
	
	
	cl_int ret = 0;
	//Copy Satellite positions to all Available devices
	for (int i = 0; i< num_of_cldevices;i++){
		ret = clEnqueueWriteBuffer(cl_devices[i].command_queue, cl_devices[i].satelite_data_gpu, CL_TRUE, 0, SATELITE_COUNT * sizeof(satelite), satelites, 0, NULL, &cl_devices[i].evnt);
		checkAndHandleErr(ret, i, "ERROR clEnqueueWriteBuffer\n", __LINE__);
	}
	for (int i = 0; i< num_of_cldevices;i++){
		ret = clWaitForEvents(1, &cl_devices[i].evnt);
		checkAndHandleErr(ret, i, "ERROR clEnqueueWriteBuffer\n", __LINE__);
		clReleaseEvent(cl_devices[i].evnt);
	}
	gettimeofday(&t2, NULL);
		
		
	while(1){
		if (testloops == 100){
			if (best_frame_time < 99999){
				FILE *fp = fopen(DEBUG_FILENAME, "a");   
				fprintf(fp, "%d;%d;%d;%.2f;%.0f;%.0f;%.0f;%.0f;%d;%d;%dx%d\n",best_frame_time, best_moving_time, best_coloring_time, coloring_avg, coloring_avg_cpu,coloring_avg_gpu,memory_avg_cpu,memory_avg_gpu, itemsize_x, itemsize_y,itemsize_x, itemsize_y);   
				fclose(fp);
			}
			testloops = 0;
			best_frame_time = 99999;
			best_coloring_time = 9999;
			best_moving_time = 9999;
			coloring_avg = 99.9f;
			coloring_avg_cpu = 9999999999.9f;
			coloring_avg_gpu = 9999999999.9f;
			memory_avg_cpu = 999999999.9f;
			memory_avg_gpu = 999999999.9f;
			render_avg = 55.0f;
			
			itemsize_x++;
			if (itemsize_x > 256){
				itemsize_y++;
				itemsize_x = 1;
			}
			if (itemsize_y > 128){
				itemsize_y = 1;
				exit(0);
			}
			
			for (int i = 0; i< num_of_cldevices;i++){
				cl_devices[i].global_size[0] = WINDOW_WIDTH/itemsize_x; cl_devices[i].global_size[1] = (cl_devices[i].global_stop_y - cl_devices[i].global_start_y) / itemsize_y;
				cl_devices[i].local_size[0] = itemsize_x; cl_devices[i].local_size[1] = itemsize_y;
			}
			fprintf(stdout, "itemsize_x:%d, itemsize_y:%d\n",itemsize_x,itemsize_y);
		}
	
		testloops ++;
		
		
		int size_ok = 0;
		// wait event synchronization handle used by OpenCL API
		for (int i = num_of_cldevices-1;i>-1; i--){
			//fprintf(stdout,"clEnqueueNDRangeKernel starts_ device:%d global:%d local:%d\n",i,cl_devices[i].global_size, cl_devices[i].local_size);
			ret = clEnqueueNDRangeKernel(cl_devices[i].command_queue, cl_devices[i].kernel, 2, NULL, cl_devices[i].global_size, cl_devices[i].local_size, 0, NULL, &cl_devices[i].evnt);
			if (ret != CL_SUCCESS){
				//Handle only unexpected errors, work_group size errors are fine.
				if (ret != CL_INVALID_WORK_GROUP_SIZE && ret != CL_INVALID_GLOBAL_WORK_SIZE){
					checkAndHandleErr(ret, i, "ERROR clEnqueueNDRangeKernel\n", __LINE__);
				}
			}
			else{	
				size_ok ++;
			}
		}
		if (size_ok == num_of_cldevices){
			break;
		}			
	}
	for (int i = num_of_cldevices-1;i>-1; i--){
		ret = clWaitForEvents(1, &cl_devices[i].evnt);
		checkAndHandleErr(ret, i, "ERROR clEnqueueWriteBuffer\n", __LINE__);
		clReleaseEvent(cl_devices[i].evnt);
		if (i == 0){
			gettimeofday(&t3_sub1, NULL);
		}
		else{
			gettimeofday(&t3_sub2, NULL);			
		}
		ret = clEnqueueReadBuffer(	cl_devices[i].command_queue, 
											cl_devices[i].satelite_id_gpu, 
											CL_TRUE, 
											0, 
											cl_devices[i].pixel_arr_size * sizeof(char), 
											cl_devices[i].pixel_ids, 
											0, 
											NULL, 
											&cl_devices[i].evnt);
											
		checkAndHandleErr(ret, i, "ERROR clEnqueueWriteBuffer\n", __LINE__);
	}
	
	gettimeofday(&t3, NULL);	
	//for (int i = 0; i< num_of_cldevices;i++){
		
	//}
	
	gettimeofday(&t4, NULL);
	
	gettimeofday(&t5, NULL);
	
	color default_cl = {.red = 1.0f, .green= 1.0f, .blue=1.0f};
	
	for (int d = 0; d < num_of_cldevices; d++){
		//WAit for buffer read to be ready
		ret = clWaitForEvents(1, &cl_devices[d].evnt);
		checkAndHandleErr(ret, d, "ERROR clEnqueueWriteBuffer\n", __LINE__);
		clReleaseEvent(cl_devices[d].evnt);
		if (d == 0){
			gettimeofday(&t4_sub1, NULL);
		}
		else{
			gettimeofday(&t4_sub2, NULL);
		}
		
		int offset_start = (cl_devices[d].global_start_y * WINDOW_WIDTH);
		int offset_stop = (cl_devices[d].global_stop_y * WINDOW_WIDTH);
		char *pixel_ids = cl_devices[d].pixel_ids;
		#pragma omp parallel for
		for (int i = offset_start; i < offset_stop; i++){
			char id = pixel_ids[i-offset_start];
			if (id == -1){
				pixels[i] = default_cl;
			}
			else{
				color cl = satelites[id].identifier;
				pixels[i] = cl;
			}
		}
	}
	gettimeofday(&t6, NULL);
	
	double elapsedTime_2 = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec);      // sec to ms
	double elapsedTime_3 = (t3.tv_sec - t2.tv_sec) * 1000.0 + (t3.tv_usec - t2.tv_usec);      // sec to ms
	double elapsedTime_3s1 = (t3_sub1.tv_sec - t2.tv_sec) * 1000.0 + (t3_sub1.tv_usec - t2.tv_usec);      // sec to ms
	double elapsedTime_3s2 = (t3_sub2.tv_sec - t2.tv_sec) * 1000.0 + (t3_sub2.tv_usec - t2.tv_usec);      // sec to ms
	double elapsedTime_4 = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec);      // sec to ms
	double elapsedTime_4s1 = (t4_sub1.tv_sec - t3.tv_sec) * 1000.0 + (t4_sub1.tv_usec - t3.tv_usec);      // sec to ms
	double elapsedTime_4s2 = (t4_sub2.tv_sec - t3.tv_sec) * 1000.0 + (t4_sub2.tv_usec - t3.tv_usec);      // sec to ms
	double elapsedTime_5 = (t4_sub2.tv_sec - t4.tv_sec) * 1000.0 + (t4_sub2.tv_usec - t4.tv_usec);      // sec to ms
	double elapsedTime_6 = (t6.tv_sec - t5.tv_sec) * 1000.0 + (t6.tv_usec - t4_sub2.tv_usec);      // sec to ms
	double total = (t6.tv_sec - t1.tv_sec) * 1000.0 + (t6.tv_usec - t1.tv_usec);      // sec to ms
	fprintf(stdout,"total:%.2lf t2:%.2lf t3:%.2lf t3s1:%.2lf t3s2:%.2lf t4:%.2lf t4s1:%.2lf t4s2:%.2lf t5:%.2lf t6:%.2lf\n",total,elapsedTime_2,elapsedTime_3,elapsedTime_3s1,elapsedTime_3s2,elapsedTime_4,elapsedTime_4s1,elapsedTime_4s2,elapsedTime_5,elapsedTime_6);
	
	if (elapsedTime_3s1 > 0 && elapsedTime_3s2 > 0){
		if (coloring_avg_cpu > elapsedTime_3s1){
			coloring_avg_cpu = elapsedTime_3s1;
		} 
		if (coloring_avg_gpu > elapsedTime_3s2){
			coloring_avg_gpu = elapsedTime_3s2;
		} 
		if (memory_avg_cpu > elapsedTime_4s1){
			memory_avg_cpu = elapsedTime_4s1;
		} 
		if (memory_avg_gpu > elapsedTime_4s2){
			memory_avg_gpu = elapsedTime_4s2;
		} 
	}
	
}

// ## You may add your own destrcution routines here ##
void destroy(){
	for (int i = 0; i< num_of_cldevices;i++){
		cl_int ret = clFlush(cl_devices[i].command_queue);
		ret = clFinish(cl_devices[i].command_queue);
		ret = clReleaseKernel(cl_devices[i].kernel);
		ret = clReleaseProgram(cl_devices[i].program);
		ret = clReleaseMemObject(cl_devices[i].satelite_data_gpu);
		ret = clReleaseMemObject(cl_devices[i].pixels_gpu);
		ret = clReleaseMemObject(cl_devices[i].pixel_start_offset_y);
		ret = clReleaseCommandQueue(cl_devices[i].command_queue);
		ret = clReleaseContext(cl_devices[i].context);
   	free(cl_devices[i].global_size);
   	free(cl_devices[i].local_size);
   	free(cl_devices[i].pixel_ids);
	}
	free(cl_devices);
	//Clean the size holders
	//free(global_size);
	//free(local_size);

	// Clean up
	//free(pixel_ids);
}



////////////////////////////////////////////////
// ¤¤ TO NOT EDIT ANYTHING AFTER THIS LINE ¤¤ //
////////////////////////////////////////////////

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Sequential rendering loop used for findign errors
void sequentialGraphicsEngine(){
   for(int i=0; i < SIZE; ++i) {

      // Row wise ordering
      vector pixel = {.x = i % WINDOW_WIDTH, .y = i / WINDOW_WIDTH};

      // This color is used for coloring the pixel
      color renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};

      // Find closest satelite
      float shortestDistance = INFINITY;
      for(int j = 0; j < SATELITE_COUNT; ++j){
         vector difference = {.x = pixel.x - satelites[j].position.x,
                              .y = pixel.y - satelites[j].position.y};
         float distance = sqrt(difference.x * difference.x + 
            difference.y * difference.y);
         if(distance < shortestDistance){
            shortestDistance = distance;
            renderColor = satelites[j].identifier;
         }
         // Display satelites themselves with white
         if(distance < SATELITE_RADIUS){
            renderColor.red = 1.0f;
            renderColor.green = 1.0f;
            renderColor.blue = 1.0f;
         }
      }

      correctPixels[i] = renderColor;
   }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void errorCheck(){
   for(int i=0; i < SIZE; ++i) {
      if(correctPixels[i].red != pixels[i].red ||
         correctPixels[i].green != pixels[i].green ||
         correctPixels[i].blue != pixels[i].blue){ 
			printf("cp_r:%.6f cp_g:%.6f cp_b:%.6f   px_r:%.6f px_g:%.6f px_b:%.6f\n", 	correctPixels[i].red,correctPixels[i].green,
																							 					correctPixels[i].blue,pixels[i].red,
																							 					pixels[i].green,pixels[i].blue);
         printf("Buggy pixel at (x=%i, y=%i). Press enter to continue.\n", i % WINDOW_WIDTH, i / WINDOW_WIDTH);
         getchar();
         return;
       }
   }
   printf("Error check passed!\n");
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void compute(void){
   int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
   int deltaTime = timeSinceStart - previousFrameTimeSinceStart;
   previousFrameTimeSinceStart = timeSinceStart;

   // Moves satelites in the space
   parallelPhysicsEngine(deltaTime);

   int sateliteMovementTime = glutGet(GLUT_ELAPSED_TIME) - timeSinceStart;

   // Decides the colors for the pixels
   parallelGraphicsEngine();

   // Sequential code is used to check possible errors in the parallel version
   if(frameNumber < 2){
      sequentialGraphicsEngine();
      errorCheck();
   }

   // Print timings
   int pixelColoringTime = glutGet(GLUT_ELAPSED_TIME) - timeSinceStart;
   render_avg = (render_avg*9+pixelColoringTime)/10;
   printf("Total frametime: %ims, satelite moving: %ims, space coloring: %ims. avg:%fms is_x:%d is_y:%d\n",
      deltaTime, sateliteMovementTime, pixelColoringTime, render_avg, itemsize_x, itemsize_y);
   if (best_frame_time > deltaTime){
   	best_frame_time = deltaTime;
   }
   if (best_moving_time > sateliteMovementTime){
   	best_moving_time = sateliteMovementTime;
   }
   if (best_coloring_time > pixelColoringTime){
   	best_coloring_time = pixelColoringTime;
   }
   if (coloring_avg > render_avg){
		coloring_avg = render_avg;
	}   
	
	
   
   // Render the frame
   glutPostRedisplay();
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Probably not the best random number generator
float random(float min, float max){
   return (rand() * (max - min) / RAND_MAX) + min;
}

// DO NOT EDIT THIS FUNCTION
void fixedInit(unsigned int seed){

   if(seed != 0){
     srand(seed);
   }

   // Init pixel buffer which is rendered to the widow
   pixels = (color*)malloc(sizeof(color) * SIZE);

   // Init pixel buffer which is used for error checking
   correctPixels = (color*)malloc(sizeof(color) * SIZE);

   // Init satelites buffer which are moving in the space
   satelites = (satelite*)malloc(sizeof(satelite) * SATELITE_COUNT);

   // Create random satelites
   for(int i = 0; i < SATELITE_COUNT; ++i){

      // Random reddish color
      color id = {.red = random(0.f, 0.5f) + 0.5f,
                  .green = random(0.f, 0.5f) + 0.0f,
                  .blue = random(0.f, 0.5f) + 0.0f};
    
      // Random position with margins to borders
      vector initialPosition = {.x = HORIZONTAL_CENTER - random(50, 300),
                              .y = VERTICAL_CENTER - random(50, 300) };
      initialPosition.x = (i / 2 % 2 == 0) ?
         initialPosition.x : WINDOW_WIDTH - initialPosition.x;
      initialPosition.y = (i < SATELITE_COUNT / 2) ?
         initialPosition.y : WINDOW_HEIGHT - initialPosition.y;

      // Randomize velocity tangential to the balck hole
      vector positionToBlackHole = {.x = initialPosition.x - HORIZONTAL_CENTER,
                                    .y = initialPosition.y - VERTICAL_CENTER};
      float distance = (0.06 + random(-0.01f, 0.01f))/ 
        sqrt(positionToBlackHole.x * positionToBlackHole.x + 
          positionToBlackHole.y * positionToBlackHole.y);
      vector initialVelocity = {.x = distance * -positionToBlackHole.y,
                                .y = distance * positionToBlackHole.x};

      // Every other orbits clockwise
      if(i % 2 == 0){
         initialVelocity.x = -initialVelocity.x;
         initialVelocity.y = -initialVelocity.y;
      }

      satelite tmpSatelite = {.identifier = id, .position = initialPosition,
                              .velocity = initialVelocity};
      satelites[i] = tmpSatelite;
   }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void fixedDestroy(void){
   destroy();

   free(pixels);
   free(correctPixels);
   free(satelites);

   if(seed != 0){
     printf("Used seed: %i\n", seed);
   }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Renders pixels-buffer to the window 
void render(void){
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_FLOAT, pixels);
   glutSwapBuffers();
   frameNumber++;
}

// DO NOT EDIT THIS FUNCTION
// Inits glut and start mainloop
int main(int argc, char** argv){

   if(argc > 1){
     seed = atoi(argv[1]);
     printf("Using seed: %i\n", seed);
   }

   // Init glut window
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
   glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
   glutCreateWindow("Parallelization excercise");
   glutDisplayFunc(render);
   atexit(fixedDestroy);
   previousFrameTimeSinceStart = glutGet(GLUT_ELAPSED_TIME);
   glEnable(GL_DEPTH_TEST);
   glClearColor(0.0, 0.0, 0.0, 1.0);
   fixedInit(seed);
   init();

   // compute-function is called when everythin from last frame is ready
   glutIdleFunc(compute);

   // Start main loop
   glutMainLoop();
}

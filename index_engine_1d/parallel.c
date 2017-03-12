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
#include <string.h>

//own variables
#include <sys/time.h>  
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
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
#define SATELITE_COUNT 1500

// These are used to control the satelite movement
#define SATELITE_RADIUS 3.16f
#define MAX_VELOCITY 0.1f
#define GRAVITY 1.0f

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

// Pixel buffer which is rendered to the screen
color* pixels;

// Pixel buffer which is used for error checking
color* correctPixels;

// Buffer for all satelites in the space
satelite* satelites;

// ## You may add your own variables here ##
#define LOCAL_ITEM_SIZE 32
#define MAX_CL_DEVICES 5
#define MAX_SOURCE_SIZE (0x100000)


//Multi OpenCL device configuration:
//Define what devices are taken into account
#define GPU 1
#define CPU 2
#define ALL 3
#define CL_DEVICE_TYPES ALL // CPU,GPU, ALL, IE: define CPU if you want to use CPU:s OpenCL capabilities, GPU if only GPU, All= Both

//Define share between devices, Change device_ratio based on number of available devices
int device_ratio[2] = {14, 2}; //Share between opencl Devices, first device gets sum/ratio.
//You need to change this based on device positions

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

typedef struct DeviceDesc{
	cl_device_id    deviceId;
	cl_device_type  deviceType;
	char*           deviceTypeString;
	char*           deviceName;
} DeviceDesc;

//Define struct for OpenCL devices, In case of multiple devices these are really neen
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
   int global_start_y;
   int global_stop_y;
   size_t global_size;
   size_t local_size;
   cl_event evnt;
#if SATELITE_COUNT < 255
   uint8_t* pixel_ids;
#else
	int* pixel_ids;
#endif
   int pixel_arr_size;
   
} ClDevice;


//All found and accepted OpenCL devices
ClDevice* cl_devices;
int num_of_cldevices = 0;


/*
	OpenCL commands error handler
*/
const char *getErrorString(cl_int error, int device_number){
	size_t len;
	clGetProgramBuildInfo(cl_devices[device_number].program, cl_devices[device_number].device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
	char *log = (char*)malloc(sizeof(char)*len);
	clGetProgramBuildInfo(cl_devices[device_number].program, cl_devices[device_number].device_id, CL_PROGRAM_BUILD_LOG, len, log, NULL);
	fprintf(stderr,"%s\n",log);
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
//__attribute__((always_inline)) 
void checkAndHandleErr(cl_int ret, int device, char *additional, int linenum){
	if (ret != CL_SUCCESS){
		fprintf(stderr, "Called from linenumber:%d\n",linenum);
		fprintf(stderr, "%s\n", additional);
		fprintf(stderr, "%s\n", getErrorString(ret, device));
		exit(0);
	}
}

int get_platforms_and_devices(ClDevice *CLdevices){
	int              i;
	cl_int              maxDevices = 5;
	cl_device_id*       deviceIDs = (cl_device_id*)malloc(maxDevices*sizeof(cl_device_id));
	cl_platform_id*     platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id));
	cl_uint             num_entries = 2;
	cl_uint             available;
	cl_uint             numDevices;
	DeviceDesc*         devices;

	int end_devices = 0;
	cl_int result = clGetPlatformIDs(num_entries, platforms, &available);
	for(int j= 0; j<available; j++){
		cl_int err = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, maxDevices, deviceIDs, &numDevices);
		if (err == CL_SUCCESS){
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
				if ((CL_DEVICE_TYPES == GPU && devices[i].deviceType == CL_DEVICE_TYPE_GPU) ||
					 (CL_DEVICE_TYPES == CPU && devices[i].deviceType == CL_DEVICE_TYPE_CPU) ||
					 (CL_DEVICE_TYPES == ALL)){
					printf("Device %s is of type %s\n", devices[i].deviceName, devices[i].deviceTypeString);
					CLdevices[end_devices].device_id = deviceIDs[i];
					CLdevices[end_devices].platform_id = platforms[j];
					CLdevices[end_devices].type = devices[i].deviceType;
					num_of_cldevices ++;
					end_devices ++;
				}
			}
			free(devices);
		}
	}
	
	free(deviceIDs);
	free(platforms);
	return end_devices;
}

// ## You may add your own initialization routines here ##
void init(){

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
	printf("found_devices:%d\n\n\n", found_devices);

	cl_int ret = CL_SUCCESS;
	
	//Share workload between available devices
	for (int i=0;i<found_devices;i++){
		
		
		
		if (found_devices == 1){
			fprintf(stdout, "\nFound only one device, using it then. no divisior used\n");
			cl_devices[i].global_start_y = 0;
			cl_devices[i].global_stop_y = SIZE;
			#if SATELITE_COUNT < 255
				cl_devices[i].pixel_ids = (uint8_t*)malloc(sizeof(uint8_t) * (SIZE));
			#else
				fprintf(stdout, "Over 254 satellites => using int indexes\n");
				cl_devices[i].pixel_ids = (int*)malloc(sizeof(int) * (SIZE));
			#endif
			
			cl_devices[i].pixel_arr_size = (SIZE);
			cl_devices[i].global_size = (SIZE)/(LOCAL_ITEM_SIZE); 
			cl_devices[i].local_size = LOCAL_ITEM_SIZE;
		}
		else{		
			int division_ratio = 0;
			for (int r=0;r<sizeof(device_ratio)/sizeof(device_ratio[0]);r++){
				division_ratio += device_ratio[r];
			}
			fprintf(stdout,"division_ratio:%d\n",division_ratio);
			 
			if (i==0){
				cl_devices[i].global_start_y = 0;
				cl_devices[i].global_stop_y = SIZE/division_ratio*device_ratio[i];
				#if SATELITE_COUNT < 255
					cl_devices[i].pixel_ids = (uint8_t*)malloc(sizeof(uint8_t) * (SIZE/division_ratio*device_ratio[i]));
				#else
					fprintf(stdout, "Over 254 satellites => using int indexes\n");
					cl_devices[i].pixel_ids = (int*)malloc(sizeof(int) * (SIZE/division_ratio*device_ratio[i]));
				#endif
				
				cl_devices[i].pixel_arr_size = (SIZE/division_ratio*device_ratio[i]);
				cl_devices[i].global_size = (SIZE/division_ratio*device_ratio[i])/(LOCAL_ITEM_SIZE); 
				cl_devices[i].local_size = LOCAL_ITEM_SIZE;
			}
			else{
				cl_devices[i].global_start_y = SIZE/division_ratio*(device_ratio[i-1]);
				cl_devices[i].global_stop_y = SIZE;
				#if SATELITE_COUNT < 255
					cl_devices[i].pixel_ids = (uint8_t*)malloc(sizeof(uint8_t) * (SIZE/division_ratio*device_ratio[i]));
				#else
					fprintf(stdout, "Over 254 satellites => using int indexes\n");
					cl_devices[i].pixel_ids = (int*)malloc(sizeof(int) * (SIZE/division_ratio*device_ratio[i]));
				#endif
				cl_devices[i].pixel_arr_size = (SIZE/division_ratio*device_ratio[i]);
				cl_devices[i].global_size = (SIZE/division_ratio*device_ratio[i])/(LOCAL_ITEM_SIZE); 
				cl_devices[i].local_size = LOCAL_ITEM_SIZE;
			}
		}
		fprintf(stdout, "kernel calculation zones:\n");
		fprintf(stdout, "cl_devices[i].global_start_y = %d\n",cl_devices[i].global_start_y);
		fprintf(stdout, "cl_devices[i].global_stop_y = %d\n",cl_devices[i].global_stop_y);
		fprintf(stdout, "cl_devices[i].pixel_arr_size = %d\n",cl_devices[i].pixel_arr_size);
		fprintf(stdout, "cl_devices[i].global_size = %ld\n",cl_devices[i].global_size);
		fprintf(stdout, "cl_devices[i].local_size =%ld\n",cl_devices[i].local_size);
		
		
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
		fprintf(stdout, "satelite_data_gpu id:%ld\n",(long)cl_devices[i].satelite_data_gpu);
	
		#if SATELITE_COUNT < 255
		cl_devices[i].satelite_id_gpu = clCreateBuffer(cl_devices[i].context, CL_MEM_WRITE_ONLY,  sizeof(uint8_t)*cl_devices[i].pixel_arr_size, NULL, &ret);
		#else
		cl_devices[i].satelite_id_gpu = clCreateBuffer(cl_devices[i].context, CL_MEM_WRITE_ONLY,  sizeof(int)*cl_devices[i].pixel_arr_size, NULL, &ret);
		#endif
		
		checkAndHandleErr(ret, i, "ERROR clCreateBuffer satelite_id_gpu\n",__LINE__);
		fprintf(stdout, "pixel_arr_size size:%ld\n", sizeof(uint8_t)*cl_devices[i].pixel_arr_size);
		fprintf(stdout, "pixel_arr_size id:%ld\n",(long)cl_devices[i].satelite_id_gpu);
	
	
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
		
		fprintf(stdout, "cl_devices[i].global_start_y:%d\n\n\n",cl_devices[i].global_start_y);
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

	cl_int ret = 0;
	//Copy Satellite positions to all Available devices
	for (int i = 0; i< num_of_cldevices;i++){
		ret = clEnqueueWriteBuffer(cl_devices[i].command_queue, cl_devices[i].satelite_data_gpu, CL_TRUE, 0, SATELITE_COUNT * sizeof(satelite), satelites, 0, NULL, &cl_devices[i].evnt);
		checkAndHandleErr(ret, i, "ERROR clEnqueueWriteBuffer\n", __LINE__);
	}
	
	//Wait for Copy ready event
	for (int i = 0; i< num_of_cldevices;i++){
		ret = clWaitForEvents(1, &cl_devices[i].evnt);
		checkAndHandleErr(ret, i, "ERROR clEnqueueWriteBuffer\n", __LINE__);
		clReleaseEvent(cl_devices[i].evnt);
	}
	
	//Do calculation
	for (int i = 0;i<num_of_cldevices; i++){
		ret = clEnqueueNDRangeKernel(cl_devices[i].command_queue, cl_devices[i].kernel, 1, NULL, &cl_devices[i].global_size, &cl_devices[i].local_size, 0, NULL, &cl_devices[i].evnt);
		if (ret != CL_SUCCESS){
			checkAndHandleErr(ret, i, "ERROR clEnqueueNDRangeKernel\n", __LINE__);	
		}
	}
	
	//Wait calculation to finished and transfer results
	for (int i = 0;i<num_of_cldevices; i++){
		ret = clWaitForEvents(1, &cl_devices[i].evnt);
		checkAndHandleErr(ret, i, "ERROR clEnqueueWriteBuffer\n", __LINE__);
		clReleaseEvent(cl_devices[i].evnt);

		ret = clEnqueueReadBuffer(	cl_devices[i].command_queue, 
											cl_devices[i].satelite_id_gpu, 
											CL_TRUE, 
											0, 
											#if SATELITE_COUNT < 255
												cl_devices[i].pixel_arr_size * sizeof(uint8_t), 
											#else
												cl_devices[i].pixel_arr_size * sizeof(int), 
											#endif
											cl_devices[i].pixel_ids, 
											0, 
											NULL, 
											&cl_devices[i].evnt);
											
		checkAndHandleErr(ret, i, "ERROR clEnqueueWriteBuffer\n", __LINE__);
	}
	color default_cl = {.red = 1.0f, .green= 1.0f, .blue=1.0f};
	
	//Wait data transfer to complete before continue
	for (int d = 0; d < num_of_cldevices; d++){
		//Wait for buffer read to be ready
		ret = clWaitForEvents(1, &cl_devices[d].evnt);
		checkAndHandleErr(ret, d, "ERROR clEnqueueWriteBuffer\n", __LINE__);
		clReleaseEvent(cl_devices[d].evnt);
		
		//Loop through all satelite ID:s and assign final colors to pixel -array
		int offset_start = (cl_devices[d].global_start_y);
		int offset_stop = (cl_devices[d].global_stop_y);

		for (int i = offset_start; i < offset_stop; i++){
			#if SATELITE_COUNT < 255
			uint8_t id = cl_devices[d].pixel_ids[i-offset_start];
				if (id == 0xFF){
			#else
			int id = cl_devices[d].pixel_ids[i-offset_start];
				if (id == 0xFFFF){
			#endif
				pixels[i] = default_cl;
			}
			else{
				#if SATELITE_COUNT < 255
				color cl = satelites[(uint8_t)id].identifier;
				#else
				color cl = satelites[(int)id].identifier;
				#endif
				pixels[i] = cl;
			}
		}
	}

}

// ## You may add your own destrcution routines here ##
void destroy(){
	for (int i = 0; i< num_of_cldevices;i++){
		cl_int ret = clFlush(cl_devices[i].command_queue);
		checkAndHandleErr(ret, i, "ERROR clFlush\n", __LINE__);
		ret = clFinish(cl_devices[i].command_queue);
		checkAndHandleErr(ret, i, "ERROR clFinish\n", __LINE__);
		ret = clReleaseKernel(cl_devices[i].kernel);
		checkAndHandleErr(ret, i, "ERROR clReleaseKernel\n", __LINE__);
		ret = clReleaseProgram(cl_devices[i].program);
		checkAndHandleErr(ret, i, "ERROR clReleaseProgram\n", __LINE__);
		ret = clReleaseMemObject(cl_devices[i].satelite_data_gpu);
		checkAndHandleErr(ret, i, "ERROR clReleaseMemObject\n", __LINE__);
		ret = clReleaseMemObject(cl_devices[i].pixels_gpu);
		checkAndHandleErr(ret, i, "ERROR clReleaseMemObject\n", __LINE__);
		ret = clReleaseMemObject(cl_devices[i].pixel_start_offset_y);
		checkAndHandleErr(ret, i, "ERROR clReleaseMemObject\n", __LINE__);
		ret = clReleaseCommandQueue(cl_devices[i].command_queue);
		checkAndHandleErr(ret, i, "ERROR clReleaseCommandQueue\n", __LINE__);
		ret = clReleaseContext(cl_devices[i].context);
		checkAndHandleErr(ret, i, "ERROR clReleaseContext\n", __LINE__);
   	free(cl_devices[i].pixel_ids);
	}
	free(cl_devices);
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
   printf("Total frametime: %ims, satelite moving: %ims, space coloring: %ims.\n",
      deltaTime, sateliteMovementTime, pixelColoringTime);
   
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

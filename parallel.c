/* TIE-51257 Parallelization Excercise 2017
   Copyright (c) 2016 Matias Koskela matias.koskela@tut.fi
                      Heikki Kultala heikki.kultala@tut.fi
*/

// Example compilation on linux
// no optimization:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm
// full optimization: gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O3
// prev and OpenMP:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O3 -fopenmp
// prev and OpenCL:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O3 -fopenmp -lOpenCL

// Example compilation on macos X
// no optimization:   gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL
// full optimization: gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL -O3

#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h> // printf
#include <math.h> // INFINITY
#include <stdlib.h>

//own variables
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
#define SATELITE_COUNT 125

// Define number of OpenCL kernels
#define LOCAL_ITEM_SIZE 32

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

typedef struct{
	int window_height;
	int window_width;
	int satelite_count;
	float satelite_radius;
} render_parameters;

// Pixel buffer which is rendered to the screen
color* pixels;

char* pixel_ids;

// Pixel buffer which is used for error checking
color* correctPixels;

// Buffer for all satelites in the space
satelite* satelites;

float render_avg = 0;


render_parameters* render_param = NULL;

#define MAX_SOURCE_SIZE (0x100000)
#define MAX_OPTIONS_SIZE (0x100000)

cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;   
cl_uint ret_num_devices;
cl_uint ret_num_platforms;

// ## You may add your own variables here ##

cl_mem satelite_id_gpu = NULL;
cl_mem satelite_data_gpu = NULL;
cl_mem pixels_gpu = NULL;
cl_mem render_parameters_gpu = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
command_queue = NULL;
cl_context context = NULL;




const char *getErrorString(cl_int error){
	int len;
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, NULL, NULL, &len);
	char *log = (char*)malloc(sizeof(char)*len);
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, log, NULL);
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


// ## You may add your own initialization routines here ##
void init(){

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
	
	pixel_ids = (char*)malloc(sizeof(char) * SIZE);

	render_param = (render_parameters*)malloc(sizeof(render_parameters));
	render_param->window_height = WINDOW_HEIGHT;
	render_param->window_width = WINDOW_WIDTH;
	render_param->satelite_count = SATELITE_COUNT;
	render_param->satelite_radius = SATELITE_RADIUS;

	fprintf(stdout, "Init platform\n");
	// Get platform and device information
	cl_int ret = clGetPlatformIDs(5, &platform_id, &ret_num_platforms);
	printf("\nDetected OpenCL platforms: %d\n", ret_num_platforms);
	if (ret != CL_SUCCESS){
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}
	
	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,  &device_id, &ret_num_devices);
	if (ret != CL_SUCCESS){
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}
	
	// Create an OpenCL context
	context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
   
   fprintf(stdout, "Creating buffers\n");
	// Create memory buffers on the device for each vector 
	satelite_data_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY,  SATELITE_COUNT * sizeof(satelite), NULL, &ret);
	fprintf(stdout, "satelite_data_gpu size:%ld\n",SATELITE_COUNT * sizeof(satelite));
	fprintf(stdout, "satelite_data_gpu id:%d\n",satelite_data_gpu);
	
	
	render_parameters_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(render_parameters), NULL, &ret);
	fprintf(stdout, "render_parameters size:%ld\n", sizeof(render_parameters));
	fprintf(stdout, "render_parameters id:%d\n",render_parameters_gpu);
	
	
	/*pixels_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(color), NULL, &ret);
	fprintf(stdout, "pixels_gpu size:%ld\n",SIZE * sizeof(color));
	fprintf(stdout, "pixels_gpu id:%d\n",pixels_gpu);
	*/
	
	satelite_id_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  sizeof(int)*SIZE, NULL, &ret);
	fprintf(stdout, "render_parameters size:%ld\n", sizeof(int));
	fprintf(stdout, "render_parameters id:%d\n",satelite_id_gpu);
	
	// Create a program from the kernel source
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	char *opts = CL_OPTIONS;
	printf("CL-kernel options: %s\n", CL_OPTIONS);

	fprintf(stdout, "Bulding OpenCL kernel\n");
	ret = clBuildProgram(program, 1, &device_id, opts, NULL, NULL);
	if (ret != CL_SUCCESS){
		fprintf(stderr, "ERROR clBuildProgram\n");
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}
	// Create the OpenCL kernel
	kernel = clCreateKernel(program, "render", &ret);
	if (ret != CL_SUCCESS){
		fprintf(stderr, "ERROR clCreateKernel\n");
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}
	
	
	fprintf(stdout, "Setting kernel argument locations\n");
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&satelite_data_gpu);
	if (ret != CL_SUCCESS){
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}
	
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&render_parameters_gpu);
	if (ret != CL_SUCCESS){
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}
	/*ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&pixels_gpu);
	if (ret != CL_SUCCESS){
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}*/
	
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&satelite_id_gpu);
	if (ret != CL_SUCCESS){
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}
		
	
	fprintf(stdout, "Setting kernel render parameters\n");
	ret = clEnqueueWriteBuffer(command_queue, render_parameters_gpu, CL_TRUE, 0, sizeof(render_parameters), render_param, 0, NULL, NULL);
	if (ret != CL_SUCCESS){
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}
	
	
	fprintf(stdout, "init ends\n");
	//printf( "init ends");
   //c = getchar( );
}

// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine) 
// Moves the satelites based on gravity
// This is done multiple times in a frame because the Euler integration 
// is not accurate enough to be done only once
void parallelPhysicsEngine(int deltaTime){
	//fprintf(stdout, "parallelPhysicsEngine starts\n");
   const int physicsUpdatesInOneFrame = 10000;
   #pragma omp parallel for
   for(int physicsUpdateIndex = 0; 
      physicsUpdateIndex < physicsUpdatesInOneFrame; 
      ++physicsUpdateIndex){

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
         satelites[i].velocity.x -= accumulation * normalizedDirection.x *
            deltaTime / physicsUpdatesInOneFrame;
         satelites[i].velocity.y -= accumulation * normalizedDirection.y *
            deltaTime / physicsUpdatesInOneFrame;

         // Update position based on velocity
         satelites[i].position.x = satelites[i].position.x +
            satelites[i].velocity.x * deltaTime / physicsUpdatesInOneFrame;
         satelites[i].position.y = satelites[i].position.y +
            satelites[i].velocity.y * deltaTime / physicsUpdatesInOneFrame;
      }
   }
   //fprintf(stdout, "parallelPhysicsEngine ends\n");
}

// ## You are asked to make this code parallel ##
// Rendering loop (This is called once a frame after physics engine) 
// Decides the color for each pixel.
int first = 0;
void parallelGraphicsEngine(){

	cl_int ret = 0;
	// Copy the lists A and B to their respective memory buffers

	ret = clEnqueueWriteBuffer(command_queue, satelite_data_gpu, CL_TRUE, 0, SATELITE_COUNT * sizeof(satelite), satelites, 0, NULL, NULL);
	if (ret != CL_SUCCESS){
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}
	
	
	// Execute the OpenCL kernel on the list
	size_t global_item_size = SIZE/LOCAL_ITEM_SIZE;//LOCAL_ITEM_SIZE; // Process the entire lists
	size_t local_item_size = LOCAL_ITEM_SIZE; // Divide work items into groups of 64
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
	if (ret != CL_SUCCESS){
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}
	
	/*
	ret = clEnqueueReadBuffer(command_queue, pixels_gpu, CL_TRUE, 0, SIZE * sizeof(color), pixels, 0, NULL, NULL);
	if (ret != CL_SUCCESS){
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}*/
	
	ret = clEnqueueReadBuffer(command_queue, satelite_id_gpu, CL_TRUE, 0, SIZE * sizeof(char), pixel_ids, 0, NULL, NULL);
	if (ret != CL_SUCCESS){
		fprintf(stderr, getErrorString(ret));
		exit(0);
	}
	
	color default_cl = {.red = 1.0f, .green= 1.0f, .blue=1.0f};
	for (int i = 0; i < SIZE; i++){
		char id = pixel_ids[i];
		if (id == -1){
			pixels[i] = default_cl;
		}
		else{
			color cl = satelites[id].identifier;
			pixels[i] = cl;
		}
	}

}

// ## You may add your own destrcution routines here ##
void destroy(){
	// Clean up
	free(pixel_ids);
	free(render_param);
	cl_int ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(satelite_data_gpu);
	ret = clReleaseMemObject(pixels_gpu);
	ret = clReleaseMemObject(render_parameters_gpu);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
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
   printf("Total frametime: %ims, satelite moving: %ims, space coloring: %ims. avg:%fms\n",
      deltaTime, sateliteMovementTime, pixelColoringTime, render_avg);

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

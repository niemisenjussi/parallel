/* TIE-51257 Parallelization Excercise 2017
   Copyright (c) 2016 Matias Koskela matias.koskela@tut.fi
                      Heikki Kultala heikki.kultala@tut.fi
*/

// Example compilation on linux
// no optimization:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm
// full optimization: gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O3
// prev and OpenMP:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O3 -fopenmp
// prev and OpenCL:   gcc -o parallel_orig parallel_orig.c -std=c99 -lglut -lGL -lm -O3 -fopenmp -lOpenCL

// Example compilation on macos X
// no optimization:   gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL
// full optimization: gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL -O3

#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h> // printf
#include <math.h> // INFINITY
#include <stdlib.h>

// Window handling includes
#ifndef __APPLE__
#include <GL/gl.h>
#include <GL/glut.h>
#else
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#endif
// These are used to 8 the window size
#define WINDOW_HEIGHT 80
#define WINDOW_WIDTH  80

// The number of satelites can be changed to see how it affects performance
#define SATELITE_COUNT 35

// These are used to control the satelite movement
#define SATELITE_RADIUS 3.16f
#define MAX_VELOCITY 0.1f
#define GRAVITY 1.0f

// Some helpers to window size variables
#define SIZE WINDOW_HEIGHT*WINDOW_HEIGHT
#define HORIZONTAL_CENTER (WINDOW_WIDTH / 2)
#define VERTICAL_CENTER (WINDOW_HEIGHT / 2)

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




// ## You may add your own initialization routines here ##
void init(){


}

// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine) 
// Moves the satelites based on gravity
// This is done multiple times in a frame because the Euler integration 
// is not accurate enough to be done only once
void parallelPhysicsEngine(int deltaTime){

   const int physicsUpdatesInOneFrame = 10000;
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
}

// ## You are asked to make this code parallel ##
// Rendering loop (This is called once a frame after physics engine) 
// Decides the color for each pixel.
void parallelGraphicsEngine(){
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

      pixels[i] = renderColor;
   }
}

// ## You may add your own destrcution routines here ##
void destroy(){


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

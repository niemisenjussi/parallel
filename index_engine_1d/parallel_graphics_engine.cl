
typedef struct{
   float x;
   float y;
} vector;

typedef struct{
   float red;
   float green;
   float blue;
} color;

typedef struct{
   color identifier;
   vector position;
   vector velocity;
} satelite;

#if SAT_COUNT < 255
	__kernel void render(__global const satelite *satelites,	__global unsigned char *sat_ids,	 __global const int *offset_start) {
#else
	__kernel void render(__global const satelite *satelites,	__global int *sat_ids,	 __global const int *offset_start) {
#endif
 //http://stackoverflow.com/questions/23535040/opencl-size-of-local-memory-has-impact-on-speed
 //https://software.intel.com/en-us/articles/using-opencl-20-work-group-functions
 
	// Get the index of the current element to be processed
	int global_id = get_global_id(0);
	int local_size = get_local_size(0);
	
	for (int i = 0; i < local_size; i++){	
		int position = (offset_start[0]+global_id*local_size+i); //Offset is used if multiple OpenCL devices are in use
		
		int x = (position) % WINDOW_WIDTH;
		int y = (position) / WINDOW_HEIGHT;
		float shortestDistance = INFINITY;

	#if SAT_COUNT < 255
		for(unsigned char j = 0; j < SAT_COUNT; ++j){
	#else
		for(int j = 0; j < SAT_COUNT; ++j){
	#endif
			vector difference = {.x = x - satelites[j].position.x, .y = y - satelites[j].position.y};
	
			float dist = sqrt(difference.x * difference.x + difference.y * difference.y);
			if(dist < shortestDistance){
				shortestDistance = dist;
				sat_ids[position-offset_start[0]] = j;
			}
	
			// Display satelites themselves with white
			if(dist < SAT_RADIUS){
				#if SAT_COUNT < 255
					sat_ids[position-offset_start[0]] = 0xFF;
				#else
					sat_ids[position-offset_start[0]] = 0xFFFF;
				#endif
			}
		}
	}

}

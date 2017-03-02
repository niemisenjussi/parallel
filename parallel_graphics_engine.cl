
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

typedef struct{
	int window_height;
	int window_width;
	int satelite_count;
	float satelite_radius;
} render_parameters;

__kernel void render(__global const satelite *satelites,
							__global const render_parameters *render_param, 
							//__global color *pixels,
							__global char *sat_ids) {
 //http://stackoverflow.com/questions/23535040/opencl-size-of-local-memory-has-impact-on-speed
 //https://software.intel.com/en-us/articles/using-opencl-20-work-group-functions
 
	// Get the index of the current element to be processed
	int global_id = get_global_id(0);
	int local_size = get_local_size(0);
	int group_id = get_group_id(0);
	int local_id = get_local_id(0);
  	int global_size = get_global_size(0);


  	//barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < local_size; i++){
		int position = global_id*local_size+i;
		vector pixel = {.x = (position) % WINDOW_WIDTH, .y = (position) / WINDOW_HEIGHT};
		float shortestDistance = INFINITY;

		for(int j = 0; j < SAT_COUNT; ++j){
			vector difference = {.x = pixel.x - satelites[j].position.x, .y = pixel.y - satelites[j].position.y};
		
			float dist = sqrt(difference.x * difference.x + difference.y * difference.y);
			if(dist < shortestDistance){
				shortestDistance = dist;
				sat_ids[position] = j;
			}
		
			// Display satelites themselves with white
			if(dist < SAT_RADIUS){
				sat_ids[position] = -1;
			}
		}
	}
}






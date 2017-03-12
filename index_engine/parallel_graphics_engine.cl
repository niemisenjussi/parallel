
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

__kernel void render(__global const satelite *satelites,	__global char *sat_ids,	
						   __global const int *offset_start) {
 //http://stackoverflow.com/questions/23535040/opencl-size-of-local-memory-has-impact-on-speed
 //https://software.intel.com/en-us/articles/using-opencl-20-work-group-functions
 
	// Get the index of the current element to be processed
	int global_id_x = get_global_id(0);
	int global_id_y = get_global_id(1);
	int local_size_x = get_local_size(0);
	int local_size_y = get_local_size(1);
	//int group_id = get_group_id(0);
	//int local_id = get_local_id(0);
  	//int global_size = get_global_size(0);


  	//barrier(CLK_GLOBAL_MEM_FENCE);
	int ypos = global_id_y * local_size_y;
	
	for (int k = 0; k < local_size_y; k++){ 
		int position_y = (ypos + k)+offset_start[0]; //calc row
		for (int i = 0; i < local_size_x; i++){	
			int position_x = global_id_x * local_size_x + i; //calc column
			//vector pixel = {.x = (position_x) % WINDOW_WIDTH, .y = (position) / WINDOW_HEIGHT};
			float shortestDistance = INFINITY;

			int pixel_dest = ((ypos + k)) * WINDOW_WIDTH + (position_x);
			for(char j = 0; j < SAT_COUNT; ++j){
				vector difference = {.x = position_x - satelites[j].position.x, .y = position_y - satelites[j].position.y};
		
				float dist = sqrt(difference.x * difference.x + difference.y * difference.y);
				if(dist < shortestDistance){
					shortestDistance = dist;
					sat_ids[pixel_dest] = j;
				}
		
				// Display satelites themselves with white
				if(dist < SAT_RADIUS){
					sat_ids[pixel_dest] = -1;
				}
			}
		}
	}
}






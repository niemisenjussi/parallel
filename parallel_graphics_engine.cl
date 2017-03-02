
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
							__global color *pixels) {
 //http://stackoverflow.com/questions/23535040/opencl-size-of-local-memory-has-impact-on-speed
 
 //https://software.intel.com/en-us/articles/using-opencl-20-work-group-functions
	// Get the index of the current element to be processed
	int global_id = get_global_id(0);
	int local_size = get_local_size(0);
	int group_id = get_group_id(0);
  	int global_size = get_global_size(0);
  	
  	int i = global_id;
	for (int i = 0; i< 8; i++){
		vector pixel = {.x = (global_id+i) % render_param->window_width, .y = (global_id+i) / render_param->window_height};
		color renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};
		float shortestDistance = INFINITY;

		for(int j = 0; j < render_param->satelite_count; ++j){
			vector difference = {.x = pixel.x - satelites[j].position.x, .y = pixel.y - satelites[j].position.y};
		
			float dist = sqrt(difference.x * difference.x + difference.y * difference.y);
			if(dist < shortestDistance){
				shortestDistance = dist;
				renderColor.red = satelites[j].identifier.red;
				renderColor.green = satelites[j].identifier.green;
				renderColor.blue = satelites[j].identifier.blue;
			}
		
			// Display satelites themselves with white
			if(dist < render_param->satelite_radius){
				renderColor.red = 1.0f;
				renderColor.green = 1.0f;
				renderColor.blue = 1.0f;
			}
		}
		pixels[global_id+i] = renderColor;
	}
}

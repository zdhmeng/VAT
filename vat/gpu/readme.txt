install opencv3，CUDA first

compile：nvcc pvat.cu -o pvat `pkg-config --cflags --libs opencv`
 
run：./pvat filename dimension
all:
	g++ -o main main.cpp && ./main

tri:
	g++ -o tri tridiagonal.cpp && ./tri
	rm tri
iteration:
	g++ -o iteration iteration.cpp && ./iteration
	rm iteration
eigen:
	g++ -o eigen eigen.cpp && ./eigen
	rm eigen
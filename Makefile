all:
	g++ -o main main.cpp && ./main

tri:
	g++ -o tri tridiagonal.cpp && ./tri
	rm tri
zeidel:
	g++ -o iteration iteration.cpp && ./iteration
	rm iteration
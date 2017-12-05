all:
	cython --embed test15.pyx
	g++ -g -O2 -c test15.c -o test15.o `python3-config --includes`
	g++ -g -O2 -o test test15.o `python3-config --libs`

clean:
	rm -f *.o test15.c tes15

CFLAGS = -Wall -Wshadow -O3 -g # -march=native
#LDLIBS = -lm -lopenblas -L/usr/local/opt/openblas/lib  -fopenmp
LDLIBS = -lm -fopenmp -lopenblas -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib
#-lgomp

CC=gcc

all: main

clean:
	$(RM) main

.PHONY: clean

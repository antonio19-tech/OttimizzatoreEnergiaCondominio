CC=gcc
CFLAGS=-Wall -Wextra -O2 -std=c11
LIBS=-lglpk -lm

all: main

main: main.c Incertezza.c NeuralNetwork.c PL_Scheduler.c
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f main

CC=gcc
CFLAGS=-Wall -Wextra -O2 -std=c11
LIBS=-lglpk -lm

all: main

main: src/main.c src/Incertezza.c src/NeuralNetwork.c src/PL_Scheduler.c
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f main

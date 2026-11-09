CC=clang
FLAGS=-O2 -g
UTIL=util/util.c

SRCS = $(filter-out access_size_fix (2).c,$(wildcard *.c))
PROGS = $(patsubst %.c,%.out,$(SRCS))

.PHONY: all clean

all: $(PROGS)

%.out: %.c $(UTIL) util/util.h
	$(CC) $(FLAGS) $< $(UTIL) -o $@

clean:
	rm -f *.out

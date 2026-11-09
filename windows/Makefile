# Windows-only (MinGW-w64). Build with mingw32-make from a MinGW/MSYS shell or
# cmd. Self-contained: no dependency on the Linux tree one level up.
CC    = gcc
FLAGS = -O2 -g
UTIL  = util/util.c

PROGS = DMP_chase_aop.exe DMP_chase_linked_list_rand.exe

.PHONY: all clean

all: $(PROGS)

%.exe: %.c $(UTIL) util/util.h
	$(CC) $(FLAGS) $< $(UTIL) -o $@

clean:
	del /Q *.exe 2>nul

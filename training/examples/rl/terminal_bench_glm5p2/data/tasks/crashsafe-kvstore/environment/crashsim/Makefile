CC = gcc
CFLAGS = -shared -fPIC -Wall -O2
LDFLAGS = -ldl

all: libcrashsim.so

libcrashsim.so: crashsim.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f libcrashsim.so

.PHONY: all clean

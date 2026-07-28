CC = gcc
CFLAGS = -O2 -Wall

PROGRAMS = azure amber crimson onyx jade

all: $(PROGRAMS)

azure: azure.c
	$(CC) $(CFLAGS) -o $@ $<

amber: amber.c
	$(CC) $(CFLAGS) -o $@ $<

crimson: crimson.c
	$(CC) $(CFLAGS) -o $@ $<

onyx: onyx.c
	$(CC) $(CFLAGS) -o $@ $<

jade: jade.c
	$(CC) $(CFLAGS) -o $@ $<

strip: all
	strip $(PROGRAMS)

clean:
	rm -f $(PROGRAMS)

.PHONY: all strip clean

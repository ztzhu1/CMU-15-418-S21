
CXXFLAGS += -Wall -Wextra -pthread -fopenmp -g -O3 -DNDEBUG

.phony: all wsp release

all: release

release: wsp.cpp
	g++ wsp.cpp -o wsp $(CXXFLAGS)

clean:
	rm -f ./wsp
	rm -f ./wsp

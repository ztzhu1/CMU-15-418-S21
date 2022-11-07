TARGET = wsp-mpi
MPICC = mpicc
MPICXX = mpic++
CFLAGS += -Wall -Werror -O2
LDFLAGS += 
LIBS += 

all: $(TARGET)

$(TARGET) : main.o $(TARGET).o
	$(MPICXX) $(CFLAGS) $(LDFLAGS) $^ $(LIBS) -o $@

main.o: main.cpp $(TARGET).h
	$(MPICXX) $(CFLAGS) -c $< -o $@

$(TARGET).o: $(TARGET).cpp $(TARGET).h
	$(MPICXX) $(CFLAGS) -c $< -o $@

.PHONY: clean run

clean:
	rm main.o $(TARGET).o

run: $(TARGET)
	mpirun -np $(np) ./$(TARGET) input/$(input)
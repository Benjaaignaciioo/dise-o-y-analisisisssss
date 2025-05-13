CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3 -I/usr/include/eigen3 -I/usr/include/jsoncpp -I./include -Wno-unused-result
LDFLAGS = -ljsoncpp

SRCS = src/experiment.cpp
TARGET = experiment

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
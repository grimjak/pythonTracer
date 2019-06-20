SHELL = /bin/sh
CC = gcc
CXX=g++

TARGET = result
SRC_DIRS ?= ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(addsuffix .o,$(basename $(SRCS)))
DEPS := $(OBJS:.o=.d)

INCLUDES = -I/app/include -I/amqpcpp/include
LIBS = -L/app/embree/build -ltbb -Wl,-Bstatic -lamqpcpp -lrabbitmq -lboost_chrono -lboost_system -lrt -lssl -lcrypto -Wl,-Bdynamic -ldl -lpthread

CPPFLAGS = -std=c++11 -W -Wall -Wextra -pedantic $(INCLUDES)
LDFLAGS =

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(LIBS)

%.cpp.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@


clean:
	rm -f $(OBJECTS)

distclean: clean
	rm -f *~ .depend


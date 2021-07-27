PALISADE=/pkg/he/palisade-1.11.3fix

CXX=g++
OPT=-O3 -std=c++17 -fopenmp
INCLUDE=-I${PALISADE}/include -I${PALISADE}/include/palisade \
        -I${PALISADE}/include/palisade/pke \
        -I${PALISADE}/include/palisade/core  \
        -I${PALISADE}/include/palisade/third-party/include
LIBS=-L/usr/lib64 -lhdf5 -lhdf5_cpp \
     -L${PALISADE}/lib -Wl,-rpath=${PALISADE}/lib -lPALISADEpke -lPALISADEcore
OBJS=henn.o

all: henn

${OBJS}: layer.h utils.h ccTest.h

henn: ${OBJS}
	$(CXX) $(OPT) $? -o $@ $(LIBS)

gothru: gothru.o
	$(CXX) $(OPT) $? -o $@ $(LIBS)


.cpp.o:
	$(CXX) $(OPT) $(INCLUDE) $< -c -o $@ $(DEBUG)
clean:
	rm *.o -f ${TARGET}


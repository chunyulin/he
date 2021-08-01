PALISADE=/pkg/he/palisade-1.11.3fix

CXX=g++
OPT=-O3 -std=c++17 -fopenmp
INCLUDE=-I${PALISADE}/include -I${PALISADE}/include/palisade \
        -I${PALISADE}/include/palisade/pke \
        -I${PALISADE}/include/palisade/core  \
        -I${PALISADE}/include/palisade/third-party/include
LIBS=-L/usr/lib64 -lhdf5 -lhdf5_cpp \
     -L${PALISADE}/lib -Wl,-rpath=${PALISADE}/lib -lPALISADEpke -lPALISADEcore


OBJS=preprocess.o heinfer.o main.o


all :main
main: $(OBJS)
	$(CXX) $(OPT) $(OBJS) -o $@ $(LIBS)

.cpp.o:
	$(CXX) $(OPT) $(INCLUDE) $< -c -o $@ $(DEBUG)
clean:
	rm -f  *.o main


preprocess.o: heinfer.h 

heinfer.o: heinfer.h

main.o: utils.h heinfer.h


##################################
gothru: gothru.o
	$(CXX) $(OPT) $? -o $@ $(LIBS)



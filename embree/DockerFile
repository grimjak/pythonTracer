FROM ubuntu:xenial as builder
#FROM embreedocker/centos:7.4 as builder


# Get g++ for compiling, wget to download Boost, git to clone source code repo,
# and make to automate program compilation with Makefile provided
 
RUN apt-get update \
  && apt-get install -y build-essential \
                        pkg-config \
                        git \
                        g++ \
                        make \
                        cmake \
                        openssl \
                        libssl-dev \
                        libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev \
                        libtbb-dev \
                        cmake-curses-gui \
                        freeglut3-dev \
                        libxmu-dev libxi-dev \
                        libcpprest-dev \
                        linux-tools-generic

RUN \
	rm -rf rabbitmq-c && \
	git clone https://github.com/alanxz/rabbitmq-c.git && \
	cd rabbitmq-c && \
	git checkout -b v0.8.0 && \
	mkdir -p build && \
	cd build && cmake .. && \
	cmake --build . && \
	cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
	make && make install && \
	cd / && rm -rf rabbitmq-c

RUN \
  rm -rf amqpcpp && \
  git clone https://github.com/akalend/amqpcpp.git && \
  cd amqpcpp && \
  make && \
  cp libamqpcpp.a /usr/local/lib

RUN \
  rm -rf msgpack-c && \
  git clone https://github.com/msgpack/msgpack-c.git && \
  cd msgpack-c && \
  cmake -DMSGPACK_CXX11=ON . && \
  make install && \
  cd / && rm -rf msgpack-c

USER root
WORKDIR /app

RUN \
  git clone https://github.com/embree/embree.git

RUN apt-get update && apt-get install -y \
            xz-utils \
            build-essential \
            curl \
            && rm -rf /var/lib/apt/lists/* \
            && curl -SL http://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz \
            | tar -xJC . && \
            mv clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04 /app/clang_6.0.0 && \
            echo 'export PATH=/app/clang_6.0.0/bin:$PATH' >> ~/.bashrc && \
            echo 'export LD_LIBRARY_PATH=/app/clang_6.0.0/lib:LD_LIBRARY_PATH' >> ~/.bashrc


RUN \
  export PATH=/app/clang_6.0.0/bin:$PATH && cd embree && mkdir build && cd build && cmake -DEMBREE_ISPC_SUPPORT=OFF -DEMBREE_TUTORIALS=OFF -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang EMBREE_MAX_ISA=AVX .. && make

RUN \
  git clone https://github.com/orca-zhang/influxdb-cpp.git && \
  ls && \
  cp influxdb-cpp/influxdb.hpp /usr/local/include



COPY ./embree /app
COPY ./common /common

#RUN /bin/bash -c "source embree/embree-vars.sh && make VERBOSE=1"
WORKDIR /app
RUN make VERBOSE=1

# FROM alpine:latest
FROM ubuntu:xenial
WORKDIR /app
ENV LD_LIBRARY_PATH=.
COPY --from=builder /app/result ./result
COPY --from=builder /app/embree/build/libembree3* ./
COPY --from=builder /usr/lib/x86_64-linux-gnu/libtbb* ./
#RUN apt-get update && apt-get install -y libtbb-dev
CMD ["/app/result"]
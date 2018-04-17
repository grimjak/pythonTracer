FROM ubuntu:xenial as builder
#FROM embreedocker/centos:7.4 as builder


# Get g++ for compiling, wget to download Boost, git to clone source code repo,
# and make to automate program compilation with Makefile provided
RUN apt-get update \
  && apt-get install -y git \
                        g++ \
                        make \
                        cmake \
                        openssl \
                        libssl-dev \
                        libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev \
                        wget \
                        libtbb-dev \
                        cmake-curses-gui \
                        freeglut3-dev \
                        libxmu-dev libxi-dev

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


USER root
WORKDIR /app

RUN \
  git clone https://github.com/embree/embree.git

RUN \
  cd embree && mkdir build && cd build && cmake -DEMBREE_ISPC_SUPPORT=OFF .. && make

RUN \
  mkdir tmp && \
  cd tmp && \ 
  git clone https://github.com/Tencent/rapidjson.git && \
  cp -rf rapidjson/include /app/include

COPY . /app

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
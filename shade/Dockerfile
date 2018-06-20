FROM ubuntu:xenial as builder
#FROM embreedocker/centos:7.4 as builder


# Get g++ for compiling, wget to download Boost, git to clone source code repo,
# and make to automate program compilation with Makefile provided
RUN apt-get update \
  && apt-get install -y build-essential \
                        git \
                        cmake \
                        openssl libssl-dev \
                        libboost-all-dev \
                        flex bison \
                        libtbb-dev \
                        libpugixml-dev \
                        libilmbase-dev \
                        libopenimageio-dev \
                        llvm  \
                        clang-3.5

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
  mkdir tmp && \
  cd tmp && \ 
  git clone https://github.com/Tencent/rapidjson.git && \
  cp -rf rapidjson/include /app/include

COPY . /app

WORKDIR /app
RUN make VERBOSE=1

# FROM alpine:latest
FROM ubuntu:xenial
WORKDIR /app
ENV LD_LIBRARY_PATH=.
COPY --from=builder /app/result ./result
COPY --from=builder /usr/lib/x86_64-linux-gnu/libtbb* ./

CMD ["/app/result"]
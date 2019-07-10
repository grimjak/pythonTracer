#include <msgpack.hpp>

#ifdef EMBREE
//#include <embree3/rtcore.h> 
//#include<tutorials/common/math/math.h>
//#include<tutorials/common/math/vec.h>
#include<tutorials/common/core/ray.h>
using namespace embree;
#endif

#ifdef IMATH
#include <OpenEXR/ImathVec.h>
typedef float Float;
typedef Imath_2_2::Vec3<Float>     Vec3f;
typedef Imath_2_2::Matrix33<Float> Matrix33;
typedef Imath_2_2::Matrix44<Float> Matrix44;
typedef Imath_2_2::Color3<Float>   Color3;
typedef Imath_2_2::Vec2<Float>     Vec2;
#endif


typedef struct PixelSample { 
  int o, w, i, j; Vec3f t;
  PixelSample(){};
  PixelSample(int offset, int weight, int ii,int jj, Vec3f throughput):o(offset),w(weight),i(ii),j(jj),t(throughput){}
} PixelSample;

typedef struct TRay {float pdf; 
                    int depth; 
                    Vec3f o, d;
                    TRay(){};
                    TRay(Vec3f origin, Vec3f direction):pdf(1.0),depth(0),o(origin),d(direction){};
                    TRay(float p, int d, Vec3f origin, Vec3f direction):pdf(p),depth(d),o(origin),d(direction){};

} TRay;

// Routines to convert to and from messagepacks
// User defined class template specialization
namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
namespace adaptor {

template<>
struct convert<PixelSample> {
    msgpack::object const& operator()(msgpack::object const& o, PixelSample& v) const {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        if (o.via.array.size != 7) throw msgpack::type_error();
        v = PixelSample(
            o.via.array.ptr[0].as<int>(),
            o.via.array.ptr[1].as<int>(),
            o.via.array.ptr[2].as<int>(),
            o.via.array.ptr[3].as<int>(),
            Vec3f(
              o.via.array.ptr[4].as<float>(),
              o.via.array.ptr[5].as<float>(),
              o.via.array.ptr[6].as<float>()           
            )
            );

        return o;
    }
};

template<>
struct pack<PixelSample> {
    template <typename Stream>
    packer<Stream>& operator()(msgpack::packer<Stream>& o, PixelSample const& v) const {
        // packing member variables as an array.
        o.pack_array(7);
        o.pack(v.o);o.pack(v.w);o.pack(v.i);o.pack(v.j);
        o.pack(v.t.x);o.pack(v.t.y);o.pack(v.t.z);
        return o;
    }
};

template<>
struct convert<TRay> {
    msgpack::object const& operator()(msgpack::object const& o, TRay& v) const {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        if (o.via.array.size != 8) throw msgpack::type_error();
        v = TRay(
            o.via.array.ptr[0].as<float>(),
            o.via.array.ptr[1].as<int>(),
            Vec3f(
              o.via.array.ptr[2].as<float>(),
              o.via.array.ptr[3].as<float>(),
              o.via.array.ptr[4].as<float>()           
            ),
            Vec3f(
              o.via.array.ptr[5].as<float>(),
              o.via.array.ptr[6].as<float>(),
              o.via.array.ptr[7].as<float>()           
            )
            );

        return o;
    }
};

template<>
struct pack<TRay> {
    template <typename Stream>
    packer<Stream>& operator()(msgpack::packer<Stream>& o, TRay const& v) const {
        // packing member variables as an array.
        o.pack_array(8);
        o.pack(v.pdf);o.pack(v.depth);
        o.pack(v.o.x);o.pack(v.o.y);o.pack(v.o.z);
        o.pack(v.d.x);o.pack(v.d.y);o.pack(v.d.z);

        return o;
    }
};

template<>
struct convert<Vec3f> {
    msgpack::object const& operator()(msgpack::object const& o, Vec3f& v) const {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        if (o.via.array.size != 3) throw msgpack::type_error();
        v = Vec3f(
              o.via.array.ptr[0].as<float>(),
              o.via.array.ptr[1].as<float>(),
              o.via.array.ptr[2].as<float>()           
            );

        return o;
    }
};

template<>
struct pack<Vec3f> {
    template <typename Stream>
    packer<Stream>& operator()(msgpack::packer<Stream>& o, Vec3f const& v) const {
        // packing member variables as an array.
        o.pack_array(3);
        o.pack(v.x);o.pack(v.y);o.pack(v.z);

        return o;
    }
};

}
}
}
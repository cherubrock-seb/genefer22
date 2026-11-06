// Synthetic OpenCL endpoint: record host dispatch and inject admission failures.
#define CL_TARGET_OPENCL_VERSION 110
#include <CL/cl.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
struct _cl_program { bool aux; };
struct _cl_kernel { std::string name; bool aux; cl_mem data = nullptr; cl_mem source = nullptr; cl_uint offset = 0; };
struct _cl_mem { int id; };
static int nextBuffer = 0;
static const std::string setting(const char* key) { const char* value=std::getenv(key); return value ? value : ""; }
template<class T> static cl_int put(T value, size_t bytes, void *out) {
    if (bytes < sizeof(T)) return CL_INVALID_VALUE;
    std::memcpy(out,&value,sizeof(T)); return CL_SUCCESS;
}
extern "C" {
cl_int clGetPlatformInfo(cl_platform_id,cl_platform_info,size_t n,void *out,size_t*) { std::memset(out,0,n); return CL_SUCCESS; }
cl_int clGetDeviceInfo(cl_device_id,cl_device_info key,size_t n,void *out,size_t *returned) {
    if (key==CL_DEVICE_MAX_WORK_ITEM_SIZES) {
        if (setting("FAIL")=="query") return CL_INVALID_VALUE;
        size_t values[3]={setting("FAIL")=="dim"?32ul:1024ul,1024,64}; if (returned) *returned=sizeof(values);
        if(out) { if(n<sizeof(values)) return CL_INVALID_VALUE; std::memcpy(out,values,sizeof(values)); } return CL_SUCCESS;
    }
    if(key==CL_DEVICE_MAX_WORK_GROUP_SIZE) return put<size_t>(setting("FAIL")=="device"?128:1024,n,out);
    if(key==CL_DEVICE_LOCAL_MEM_SIZE) return put<cl_ulong>(setting("FAIL")=="local"?8192:65536,n,out);
    if(out) std::memset(out,0,n);
    return CL_SUCCESS;
}
cl_context clCreateContext(const cl_context_properties*,cl_uint,const cl_device_id*,void(CL_CALLBACK *)(const char*, const void*,size_t,void*),void*,cl_int*e) { *e=0;return reinterpret_cast<cl_context>(1); }
cl_command_queue clCreateCommandQueue(cl_context,cl_device_id,cl_command_queue_properties,cl_int*e){*e=0;return reinterpret_cast<cl_command_queue>(1);}
cl_program clCreateProgramWithSource(cl_context,cl_uint,const char**s,const size_t*,cl_int*e){*e=0;return new _cl_program{std::strncmp(s[0],"#define GENEFER",15)==0};}
cl_int clBuildProgram(cl_program p,cl_uint,const cl_device_id*,const char*,void(CL_CALLBACK *)(cl_program,void*),void*){return p->aux&&setting("FAIL")=="build"?CL_BUILD_PROGRAM_FAILURE:CL_SUCCESS;}
cl_mem clCreateBuffer(cl_context,cl_mem_flags,size_t,void*,cl_int*e){*e=0;return new _cl_mem{++nextBuffer};}
cl_int clEnqueueWriteBuffer(cl_command_queue,cl_mem,cl_bool,size_t,size_t,const void*,cl_uint,const cl_event*,cl_event*){return 0;}
cl_kernel clCreateKernel(cl_program p,const char*n,cl_int*e){if(p->aux&&setting("FAIL")=="missing"){*e=CL_INVALID_KERNEL_NAME;return nullptr;} *e=0;return new _cl_kernel{n,p->aux};}
cl_int clSetKernelArg(cl_kernel k,cl_uint i,size_t n,const void*v){if(k->aux&&setting("FAIL")=="arg")return CL_INVALID_ARG_INDEX; if(k->aux && i==2) { if(setting("FAIL")=="source") return CL_INVALID_ARG_VALUE; std::memcpy(&k->source,v,n); } if(k->aux && i==3) { if(setting("FAIL")=="offset") return CL_INVALID_ARG_VALUE; std::memcpy(&k->offset,v,n); } if(i==0&&n==sizeof(cl_mem))std::memcpy(&k->data,v,n);return 0;}
cl_int clGetKernelWorkGroupInfo(cl_kernel k,cl_device_id,cl_kernel_work_group_info key,size_t n,void*out,size_t*){
    if(key==CL_KERNEL_WORK_GROUP_SIZE)return put<size_t>(setting("DENY")==k->name?31:1024,n,out);
    if(key==CL_KERNEL_LOCAL_MEM_SIZE)return put<cl_ulong>(k->name=="square8192"?32768:(k->name=="forward1024_0" || k->name=="prepare256_0" || k->name=="prepare1024_0")?16384:4096,n,out);
    return CL_INVALID_VALUE;
}
cl_int clEnqueueNDRangeKernel(cl_command_queue,cl_kernel k,cl_uint,const size_t*,const size_t*,const size_t*,cl_uint,const cl_event*,cl_event*) {
    std::cout<<k->name<<'@'<<(k->data?k->data->id:0)<<' ';
    if(k->aux) std::cout<<"source@"<<(k->source?k->source->id:0)<<":"<<k->offset<<" ";
    return setting("BAD_ENQUEUE")==k->name?CL_OUT_OF_RESOURCES:CL_SUCCESS;
}
cl_int clReleaseKernel(cl_kernel k){delete k;return 0;}
cl_int clReleaseProgram(cl_program p){delete p;return 0;}
cl_int clReleaseMemObject(cl_mem m){delete m;return 0;}
cl_int clFinish(cl_command_queue){return 0;}
cl_int clReleaseCommandQueue(cl_command_queue){return 0;}
cl_int clReleaseContext(cl_context){return 0;}
}

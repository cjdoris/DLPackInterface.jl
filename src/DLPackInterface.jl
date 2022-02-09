module DLPackInterface

import Requires: @require
import ArrayInterface

export DLDeviceType, DLDevice, DLDataTypeCode, DLDataType, DLTensor
export DL_CPU, DL_CUDA, DL_CUDAHost, DL_OpenCL, DL_Vulkan, DL_Metal, DL_VPI, DL_ROCM, DL_ROCMHost, DL_ExtDev, DL_CUDAManaged, DL_OneAPI, DL_WebGPU, DL_Hexagon
export DL_Int, DL_UInt, DL_Float, DL_OpaqueHandle, DL_Bfloat, DL_Complex
export isdleltype, isdltensor, dltensor, dldata, dldataoffset, dldevice, dldatatype, dlndim, dlshape, dlstride, dlstrides, dlsizeof, dlinfo, dllength

### Types
#
# Note: These are similar to but NOT the same as the C structs from dlpack.h.

@enum DLDeviceType begin
    DL_CPU = 1
    DL_CUDA = 2
    DL_CUDAHost = 3
    DL_OpenCL = 4
    DL_Vulkan = 7
    DL_Metal = 8
    DL_VPI = 9
    DL_ROCM = 10
    DL_ROCMHost = 11
    DL_ExtDev = 12
    DL_CUDAManaged = 13
    DL_OneAPI = 14
    DL_WebGPU = 15
    DL_Hexagon = 16
end

"""
    DLDevice(type::DLDeviceType, id=0)

A device of the given type and id.
"""
struct DLDevice
    type::DLDeviceType
    id::Int
end
DLDevice(t) = DLDevice(t, 0)

@enum DLDataTypeCode begin
    DL_Int = 0
    DL_UInt = 1
    DL_Float = 2
    DL_OpaqueHandle = 3
    DL_Bfloat = 4
    DL_Complex = 5
end

"""
    DLDataType(code::DLDataTypeCode, bits, lanes=1)
    DLDataType(T::Type, lanes=1)

A datatype with the specified number of bits.
"""
struct DLDataType
    code::DLDataTypeCode
    bits::Int
    lanes::Int
end
DLDataType(code, bits) = DLDataType(code, bits, 1)

const TYPE_TO_DATATYPE = Dict(
    Int8 => DLDataType(DL_Int, 8),
    Int16 => DLDataType(DL_Int, 16),
    Int32 => DLDataType(DL_Int, 32),
    Int64 => DLDataType(DL_Int, 64),
    UInt8 => DLDataType(DL_UInt, 8),
    UInt16 => DLDataType(DL_UInt, 16),
    UInt32 => DLDataType(DL_UInt, 32),
    UInt64 => DLDataType(DL_UInt, 64),
    Float16 => DLDataType(DL_Float, 16),
    Float32 => DLDataType(DL_Float, 32),
    Float64 => DLDataType(DL_Float, 64),
    ComplexF16 => DLDataType(DL_Complex, 32),
    ComplexF32 => DLDataType(DL_Complex, 64),
    ComplexF64 => DLDataType(DL_Complex, 128),
)

DLDataType(::Type, lanes=1) = error("not a valid DLPack data type")
for (T, t) in TYPE_TO_DATATYPE
    @eval DLDataType(::Type{$T}, lanes=1) = DLDataType($(t.code), $(t.bits), lanes)
end


### Interface functions

"""
    isdleltype(T::Type)

True if `T` is a valid eltype for a DLTensor.
"""
function isdleltype end
@eval isdleltype(::Type{T}) where {T} = T in $(Tuple(keys(TYPE_TO_DATATYPE)))

"""
    isdltensor(T::Type)
    isdltensor(x::T)

True if objects of type `T` satisfy the DLPack interface.

You can call `t = dltensor(x)` on such an object, then call `dldata(t)`, `dldevice(t)`,
etc. to determine information about the tensor.
"""
isdltensor(::Type{T}) where {T} = false
isdltensor(::Type{T}) where {T<:AbstractArray} = isdleltype(eltype(T)) && ArrayInterface.defines_strides(T) && ArrayInterface.device(T) === ArrayInterface.CPUPointer()
isdltensor(::Type{T}) where {T<:StridedArray} = isdleltype(eltype(T))
isdltensor(x) = isdltensor(typeof(x))

"""
    dltensor(x)

An object `t` which can be passed to `dldata`, `dldevice`, etc. to determine information
about the tensor `x`.

As long as `t` is not garbarge collected, the pointer `dldata(t)` remains valid.
"""
dltensor(x) = isdltensor(x) ? x : error("not a tensor")

"""
    dldata(t) :: Ptr{Cvoid}

A pointer to the data in tensor `t`.
"""
dldata(t) = convert(Ptr{Cvoid}, pointer(t))

"""
    dldataoffset(t) :: Int

The data in tensor `t` starts at `dldata(t) + dldataoffset(t)`.
"""
dldataoffset(t) = 0

"""
    dldevice(t) :: DLDevice

The device hosting the data in tensor `t`.
"""
dldevice(t) = DLDevice(DL_CPU)

"""
    dldatatype(t) :: DLDataType

The type of data in tensor `t`.
"""
dldatatype(t) = DLDataType(eltype(t))

"""
    dlndim(t) :: Int

The number of dimensions of tensor `t`.
"""
dlndim(t) = Int(ndims(t))

"""
    dlshape(t, i) :: Int

The size of tensor `t` along the `i`th dimension.
"""
dlshape(t, i) = Int(size(t, i))

"""
    dlstride(t, i) :: Int

The distance in memory (in elements, not bytes) between elements of `t` along the `i`th
dimension.
"""
dlstride(t, i) = Int(stride(t, i))


### Utility functions

"""
    dlshape(t) :: NTuple{N,Int}

The size of tensor `t` in each dimension.
"""
dlshape(t) = ntuple(i->dlshape(t, i)::Int, dlndim(t)::Int)

"""
    dlstrides(t) :: NTuple{N,Int}

The distance in memory (in elements, not bytes) between elements of `t` along each
dimension.
"""
dlstrides(t) = ntuple(i->dlstride(t, i)::Int, dlndim(t)::Int)

"""
    dllength(t) :: Int

The number of elements in the tensor `t`.
"""
dllength(t) = prod(dlshape(t))::Int

"""
    dlsizeof(t::DLDataType) :: Int

The size in bytes of an element of this type.
"""
dlsizeof(t::DLDataType) = cld(Int(t.bits) * Int(t.lanes), 8)::Int

"""
    dlsizeof(t) :: Int

The size in bytes of the tensor `t`, if it were stored contiguously.
"""
dlsizeof(t) = dllength(t) * dlsizeof(dldatatype(t)::DLDataType)

"""
    dlinfo(x) :: Union{Nothing,NamedTuple}

If `x` satisfies the DLPack interface, return a NamedTuple of information about it.
Otherwise, return `nothing`.

It has fields `tensor`, `pointer`, `pointeroffset`, `device`, `datatype`, `ndim`, `shape`,
`strides` containing the outputs of the corresponding `dl*` functions.
"""
function dlinfo(x)
    if isdltensor(x)
        t = dltensor(x)
        (
            tensor = t,
            pointer = dldata(t)::Ptr{Cvoid},
            pointeroffset = dldataoffset(t)::Int,
            device = dldevice(t)::DLDevice,
            datatype = dldatatype(t)::DLDataType,
            ndim = dlndim(t)::Int,
            shape = dlshape(t)::Tuple{Vararg{Int}},
            strides = dlstrides(t)::Tuple{Vararg{Int}},
        )
    end
end


### Integrations

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" @eval begin
        isdltensor(::CUDA.CuArray) = true
        dldata(x::CUDA.CuArray) = reinterpret(Ptr{Cvoid}, pointer(x))
        dldevice(x::CUDA.CuArray) = DLDevice(CUDA.device(x))
        DLDevice(d::CUDA.CuDevice) = DLDevice(DL_CUDA, d.handle)
    end
end

end # module

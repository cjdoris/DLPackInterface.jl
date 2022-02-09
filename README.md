# DLPackInterface.jl

Provides a common interface for declaring that a tensor type is compatible with `DLPack`.

This includes `Array` and `CUDA.CuArray`.

## Install

```
pkg> add DLPackInterface
```

## Usage

Call `dlinfo(x)`. It returns a NamedTuple of information about `x` if it
satisfies the DLPack interface. Otherwise, it returns `nothing`.

Alternatively you can call `t = dltensor(x)`. If `x` is a tensor, this yields a view of `x`
which can be queried for information (otherwise returns `nothing`):
- `dldata(t)`
- `dldataoffset(t)`
- `dldevice(t)`
- `dldatatype(t)`
- `dlndim(t)`
- `dlshape(t, i)` and `dlshape(t)`
- `dlstride(t, i)` and `dlstrides(t)`
- `dllength(t)`
- `dlsizeof(t)`

## Implementing the interface

To implement the interface for type `T`, you must define the following functions:
- `dltensor(x::T) -> t` or `nothing` if `x` is not a tensor; it is reasonable to return `x`
- `dldata(t)::Ptr{Cvoid}` (defaults to `convert(Ptr{Cvoid}, pointer(t))`)
- `dldataoffset(t)::Int` (defaults to 0)
- `dldevice(t)::DLDevice` (defaults to `DLDevice(DL_CPU)`)
- `dldatatype(t)::DLDataType` (defaults to `DLDataType(eltype(t))`)
- `dlndim(t)::Int` (defaults to `Int(ndims(t))`)
- `dlshape(t, i)::Int` (defaults to `Int(size(t, i))`)
- `dlstride(t, i)::Int` (defaults to `Int(stride(t, i))`)

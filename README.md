# DLPackInterface.jl

Provides a common interface for declaring that a tensor type is compatible with `DLPack`.

This includes `Array` and `CUDA.CuArray`.

## Install

```
pkg> add DLPackInterface
```

## Usage

Easiest option: call `dlinfo(x)`. It returns a NamedTuple of information about `x` if it
satisfies the DLPack interface. Otherwise, it returns `nothing`.

Fine-level option: call `t = dltensor(x)`. This yields a view of `x` which can be queried
for information:
- `dldata(t)`
- `dldataoffset(t)`
- `dldevice(t)`
- `dldatatype(t)`
- `dlndim(t)`
- `dlshape(t, i)` and `dlshape(t)`
- `dlstride(t, i)` and `dlstrides(t)`
- `dllength(t)`
- `dlsizeof(t)`

You can check if `x` satisfies the interface with `isdltensor(x)`.
If this returns false then `dltensor(x)` will throw and `dlinfo(x)` will return `nothing`.

## Implementing the interface

To implement the interface for type `T`, you must define the following functions:
- `isdltensor(::Type{T})::Bool = true`
- `dltensor(x::T)` (defaults to `x`)
- `dldata(t)::Ptr{Cvoid}` (defaults to `convert(Ptr{Cvoid}, pointer(t))`)
- `dldataoffset(t)::Int` (defaults to 0)
- `dldevice(t)::DLDevice` (defaults to `DLDevice(DL_CPU)`)
- `dldatatype(t)::DLDataType` (defaults to `DLDataType(eltype(t))`)
- `dlndim(t)::Int` (defaults to `Int(ndims(t))`)
- `dlshape(t, i)::Int` (defaults to `Int(size(t, i))`)
- `dlstride(t, i)::Int` (defaults to `Int(stride(t, i))`)

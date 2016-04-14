# Cache some stuff so we don't need to recreate them in every function call

################
# Displacement #
################
type U_Buffers{T}
    NNx::Matrix{T}
    NNy::Matrix{T}
    BBx::Matrix{T}
    BBy::Matrix{T}
    g::Vector{T}
    ε::Vector{T}
    ε_m::Vector{T} # ε for a mode
    dNdx::Matrix{Float64}
    dNdy::Matrix{Float64}
    Nx::Vector{Float64}
    Ny::Vector{Float64}
end

function U_Buffers{T}(Tv::Type{T})
    g = zeros(T,8) # Not general

    Nx = zeros(2) # Shape functions
    Ny = zeros(2)

    dNdx = zeros(1,2) # Derivatives
    dNdy = zeros(1,2)

    NNx = zeros(T,2,4) # N matrix for force vector
    NNy = zeros(T,2,4)

    BBx = zeros(T,3,4) # B matrix for stiffness
    BBy = zeros(T,3,4)

    ε = zeros(T, 3)
    ε_m = zeros(T, 3)

    return U_Buffers(NNx, NNy, BBx, BBy, g, ε, ε_m, dNdx, dNdy, Nx, Ny)
end

type U_BufferCollection{Q, T}
    buff_grad::U_Buffers{Q}
    buff_float::U_Buffers{T}
end

function U_BufferCollection{Q, T}(Tq::Type{Q}, Tt::Type{T})
    U_BufferCollection(U_Buffers(Tq), U_Buffers(Tt))
end

import ForwardDiff.GradientNumber
# The 8 here is the "chunk size" used
U_Tgrad = ForwardDiff.GradientNumber{8, Float64, NTuple{8, Float64}}
get_buffer{T <: GradientNumber}(buff_coll::U_BufferCollection, ::Type{T}) = buff_coll.buff_grad
get_buffer{T}(buff_coll::U_BufferCollection, ::Type{T}) = buff_coll.buff_float


const U_buff_colls = U_BufferCollection(U_Tgrad, Float64)

######################################
# Displacement as function of damage #
######################################
type UD_Buffers{T}
    NNx::Matrix{T}
    NNy::Matrix{T}
    BBx::Matrix{T}
    BBy::Matrix{T}
    g::Vector{T}
    ε::Vector{T}
    ε_m::Vector{T} # ε for a mode
    dNdx::Matrix{Float64}
    dNdy::Matrix{Float64}
    Nx::Vector{Float64}
    Ny::Vector{Float64}
end

function UD_Buffers{T}(Tv::Type{T})
    g = zeros(T,8) # Not general

    Nx = zeros(2) # Shape functions
    Ny = zeros(2)

    dNdx = zeros(1,2) # Derivatives
    dNdy = zeros(1,2)

    NNx = zeros(T,2,4) # N matrix for force vector
    NNy = zeros(T,2,4)

    BBx = zeros(T,3,4) # B matrix for stiffness
    BBy = zeros(T,3,4)

    ε = zeros(T, 3)
    ε_m = zeros(T, 3)

    return UD_Buffers(NNx, NNy, BBx, BBy, g, ε, ε_m, dNdx, dNdy, Nx, Ny)
end

type UD_BufferCollection{Q, T}
    buff_grad::UD_Buffers{Q}
    buff_float::UD_Buffers{T}
end

function UD_BufferCollection{Q, T}(Tq::Type{Q}, Tt::Type{T})
    UD_BufferCollection(UD_Buffers(Tq), UD_Buffers(Tt))
end

# import ForwardDiff.GradientNumber
# The 8 here is the "chunk size" used
UD_Tgrad = ForwardDiff.GradientNumber{8, Float64, NTuple{8, Float64}}
get_buffer{T <: GradientNumber}(buff_coll::UD_BufferCollection, ::Type{T}) = buff_coll.buff_grad
get_buffer{T}(buff_coll::UD_BufferCollection, ::Type{T}) = buff_coll.buff_float


const UD_buff_colls = UD_BufferCollection(UD_Tgrad, Float64)

######################################
# Damage as function of displacement #
######################################
type DU_Buffers{T}
    NNx::Matrix{T}
    NNy::Matrix{T}
    BBx::Matrix{T}
    BBy::Matrix{T}
    g::Vector{T}
    ε::Vector{T}
    ε_m::Vector{T} # ε for a mode
    dNdx::Matrix{Float64}
    dNdy::Matrix{Float64}
    Nx::Vector{Float64}
    Ny::Vector{Float64}
end

function DU_Buffers{T}(Tv::Type{T})
    g = zeros(T,4) # Not general

    Nx = zeros(2) # Shape functions
    Ny = zeros(2)

    dNdx = zeros(1,2) # Derivatives
    dNdy = zeros(1,2)

    NNx = zeros(T,1,2) # N matrix for force vector
    NNy = zeros(T,1,2)

    BBx = zeros(T,2,2) # B matrix for stiffness
    BBy = zeros(T,2,2)

    ε = zeros(T, 3)
    ε_m = zeros(T, 3)

    return DU_Buffers(NNx, NNy, BBx, BBy, g, ε, ε_m, dNdx, dNdy, Nx, Ny)
end

type DU_BufferCollection{Q, T}
    buff_grad::DU_Buffers{Q}
    buff_float::DU_Buffers{T}
end

function DU_BufferCollection{Q, T}(Tq::Type{Q}, Tt::Type{T})
    DU_BufferCollection(DU_Buffers(Tq), DU_Buffers(Tt))
end

# import ForwardDiff.GradientNumber
# The 8 here is the "chunk size" used
DU_Tgrad = ForwardDiff.GradientNumber{4, Float64, NTuple{4, Float64}}
get_buffer{T <: GradientNumber}(buff_coll::DU_BufferCollection, ::Type{T}) = buff_coll.buff_grad
get_buffer{T}(buff_coll::DU_BufferCollection, ::Type{T}) = buff_coll.buff_float


const DU_buff_colls = DU_BufferCollection(DU_Tgrad, Float64)

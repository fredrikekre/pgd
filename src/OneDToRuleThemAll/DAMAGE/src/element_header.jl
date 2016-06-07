immutable d_shapefunctions{dim,T}
    Nx::Vector{Vector{T}}
    Ny::Vector{Vector{T}}
    Nz::Vector{Vector{T}}
    dNx::Vector{Vector{Tensor{1,dim,T,dim}}}
    dNy::Vector{Vector{Tensor{1,dim,T,dim}}}
    dNz::Vector{Vector{Tensor{1,dim,T,dim}}}
end

immutable u_shapefunctions{dim,T,M}
    Nx::Vector{Vector{Tensor{1,dim,T,dim}}}
    Ny::Vector{Vector{Tensor{1,dim,T,dim}}}
    Nz::Vector{Vector{Tensor{1,dim,T,dim}}}
    dNx::Vector{Vector{Tensor{2,dim,T,M}}}
    dNy::Vector{Vector{Tensor{2,dim,T,M}}}
    dNz::Vector{Vector{Tensor{2,dim,T,M}}}
end
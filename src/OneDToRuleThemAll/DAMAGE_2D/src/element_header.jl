immutable d_shapefunctions{dim,T}
    Nx::Vector{Vector{T}}
    Ny::Vector{Vector{T}}
    dNx::Vector{Vector{Tensor{1,dim,T,dim}}}
    dNy::Vector{Vector{Tensor{1,dim,T,dim}}}
end

immutable u_shapefunctions{dim,T,M}
    Nx::Vector{Vector{Tensor{1,dim,T,dim}}}
    Ny::Vector{Vector{Tensor{1,dim,T,dim}}}
    dNx::Vector{Vector{Tensor{2,dim,T,M}}}
    dNy::Vector{Vector{Tensor{2,dim,T,M}}}
end
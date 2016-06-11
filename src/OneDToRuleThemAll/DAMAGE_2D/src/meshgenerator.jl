abstract mesh
abstract spaceMesh <: mesh

type mesh1D <: spaceMesh
    ex::Vector{Vector{Vec{1,Float64,1}}} # element coordinates
    x::Vector{Vec{1,Float64,1}} # coordinates
    edof::Matrix{Int} # element dofs
    nEl::Int # number of elements
    nDofs::Int # number of dofs
    nElDofs::Int # number of dofs per element
    nElNodes::Int # number of nodes per element (2 for linear, 3 for quadratic)
    uniform::Bool # uniform mesh? If true: Speeds up integration
end


function create_mesh1D(sCoord,eCoord,nEl,nElNodes,nNodeDofs)

    nNodes = (nElNodes-1)*nEl + 1
    nDofs = nNodes*nNodeDofs
    edof = zeros(Int,nElNodes*nNodeDofs,nEl)

    x = linspace(sCoord,eCoord,nNodes)
    x = reinterpret(Vec{1,Float64},collect(x),(nNodes,))

    ex = Vector{Vec{1,Float64}}[]
    if nElNodes == 2
        for i in 1:nEl
            push!(ex,typeof(x[i])[x[i], x[i+1]])
        end
        mesh = [1:nNodes-1 2:nNodes]'
    elseif nElNodes == 3
        for i in 1:nEl
            push!(ex,typeof(x[i])[x[3*i-2], x[3*i-1], x[x[3*i]]])
        end
        mesh = [1:2:nNodes-2 2:2:nNodes-1 3:2:nNodes]'
    end

    for i = 1:nNodeDofs
        edof[i:nNodeDofs:((nElNodes-1)*nNodeDofs+i),:] = mesh*nNodeDofs-(nNodeDofs-i)
    end

    return mesh1D(ex,x,edof,nEl,nDofs,nElNodes*nNodeDofs,nElNodes,true)
end

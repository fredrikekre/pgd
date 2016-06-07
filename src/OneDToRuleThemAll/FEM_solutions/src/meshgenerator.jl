
type FEmesh{dim,T,M}
    x::Vector{Tensor{1,dim,T,M}}
    topology::Matrix{Int}
    edof::Matrix{Int}
    side_nodes::Vector{Vector{Int}}
    side_dofs::Vector{Vector{Vector{Int}}}
    nEl::Int
    nElx::Int
    nEly::Int
    nElz::Int
    nnodes::Int
end

# ##################
# # 2D square mesh #
# ##################
# function generate_mesh{T}(::Dim{2}, xs::Vector{T}, xe::Vector{T}, nel::Vector{Int})

#     xs_x = xs[1]; xs_y = xs[2]
#     xe_x = xe[1]; xe_y = xe[2]
#     nel_x = nel[1]; nel_y = nel[2]
#     nel = nel_x * nel_y
#     n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
#     n_nodes = n_nodes_x * n_nodes_y

#     coords_x = linspace(xs_x, xe_x, n_nodes_x)
#     coords_y = linspace(xs_y, xe_y, n_nodes_y)

#     coords = T[]
#     for j in 1:n_nodes_y, i in 1:n_nodes_x
#         ccoords = [coords_x[i], coords_y[j]]
#         append!(coords,ccoords)
#     end
#     coords = reshape(coords,(2,n_nodes))

#     tensor_coords = reinterpret(Vec{2, Float64}, coords, (size(coords,2),))

#     nodes = reshape(collect(1:n_nodes),(n_nodes_x,n_nodes_y))

#     topology = Int[]
#     for j in 1:nel_y, i in 1:nel_x
#         ctopology = [nodes[i:i+1,j]; nodes[i+1:-1:i,j+1]]
#         append!(topology,ctopology)
#     end
#     topology = reshape(topology,(4,nel))

#     # Edges
#     edge_topology = Vector{Int}[
#                     nodes[:,1][:], nodes[end,:][:], nodes[:,end][:], nodes[1,:][:]]

#     b1D = FEBoundary{1}[]

#     for edge in 1:length(edge_topology)
#         nel_bound = length(edge_topology[edge])-1
#         b1Dtopology = Int[]
#         for i in 1:nel_bound
#             append!(b1Dtopology,edge_topology[edge][i:i+1])
#         end
#         b1Dtopology = reshape(b1Dtopology,(2,nel_bound))
#         push!(b1D,FEBoundary{1}(b1Dtopology))
#     end

#     # Corners
#     b0D = [FEBoundary{0}([nodes[1,1]]),
#            FEBoundary{0}([nodes[end,1]]),
#            FEBoundary{0}([nodes[1,end]]),
#            FEBoundary{0}([nodes[end,end]])]


#     boundary = Vector[b1D,b0D]

#     return FEMesh(tensor_coords,topology,boundary)
# end

##################
# 3D square mesh #
##################
function generate_mesh{T}(xs::Tensor{1,3,T}, xe::Tensor{1,3,T}, nel::Vector{Int})

    xs_x = xs[1]; xs_y = xs[2]; xs_z = xs[3]
    xe_x = xe[1]; xe_y = xe[2]; xe_z = xe[3]
    nel_x = nel[1]; nel_y = nel[2]; nel_z = nel[3]
    nel = nel_x * nel_y * nel_z
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1; n_nodes_z = nel_z + 1
    n_nodes = n_nodes_x * n_nodes_y * n_nodes_z

    coords_x = linspace(xs_x, xe_x, n_nodes_x)
    coords_y = linspace(xs_y, xe_y, n_nodes_y)
    coords_z = linspace(xs_z, xe_z, n_nodes_z)

    coords = T[]
    for k in 1:n_nodes_z, j in 1:n_nodes_y, i in 1:n_nodes_x
        ccoords = [coords_x[i], coords_y[j], coords_z[k]]
        append!(coords,ccoords)
    end
    coords = reshape(coords,(3,n_nodes))

    tensor_coords = reinterpret(Vec{3, Float64}, coords, (size(coords,2),))

    # Set up topology
    nodes = reshape(collect(1:n_nodes),(n_nodes_x,n_nodes_y,n_nodes_z))

    topology = Int[]
    for k in 1:nel_z, j in 1:nel_y, i in 1:nel_x
        ctopology = [nodes[i:i+1,j,k]; nodes[i+1:-1:i,j+1,k];
                     nodes[i:i+1,j,k+1]; nodes[i+1:-1:i,j+1,k+1]]
        append!(topology,ctopology)
    end
    topology = reshape(topology,(8,nel))

    edof = Int[]
    for ele in 1:size(topology,2), node in 1:size(topology,1), dof in reverse(collect(0:2))
        push!(edof, topology[node,ele]*3 - dof)
    end
    edof = reshape(edof,(3*8,nel))

    # Sides
    side_nodes = Vector{Int}[
                    nodes[:,:,1][:],
                    nodes[:,1,:][:],
                    nodes[end,:,:][:],
                    nodes[:,end,:][:],
                    nodes[1,:,:][:],
                    nodes[:,:,end][:]]

    side_dofs = Vector{Vector{Int}}[] # Side, dof
    for i in 1:length(side_nodes)
        temp = Vector{Int}[]
        push!(temp, side_nodes[i]*3 - 2)
        push!(temp, side_nodes[i]*3 - 1)
        push!(temp, side_nodes[i]*3 - 0)
        push!(side_dofs, temp)
    end

    return FEmesh(tensor_coords,topology,edof,side_nodes,side_dofs,nel,nel_x,nel_y,nel_z,n_nodes)
end



#############
# Utilities #
#############
@inline function get_el_coords(m::FEmesh,i::Int)
    return m.x[m.topology[:,i]]
end

@inline function get_el_dofs(m::FEmesh,i::Int)
    return m.edof[:,i]
end

@inline function number_of_dofs(m::FEmesh)
    return maximum(m.edof)
end
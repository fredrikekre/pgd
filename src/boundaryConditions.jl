function zipper{T}(a::Vector{T},b::Vector{T})
    la = length(a); lb = length(b)
    la == lb || throw(ArgumentError("Length of a is $la and length of b is $lb, they must me the same."))

    c = zeros(T,la + lb)
    n = 1
    for i in 1:la
        c[[n,n+1]] = [a[i], b[i]]
        n += 2
    end

    return c
end


immutable PGDBC
    fixed::Vector{Vector{Int}}
    prescr::Vector{Vector{Int}}
    free::Vector{Vector{Int}}
end

# Combined bc's
fixed_dofs(bc::PGDBC) = bc.fixed[end]
prescr_dofs(bc::PGDBC) = bc.prescr[end]
free_dofs(bc::PGDBC) = bc.free[end]

# Dimension specific bc's
fixed_dofs(bc::PGDBC,dim::Int) = bc.fixed[dim]
prescr_dofs(bc::PGDBC,dim::Int) = bc.prescr[dim]
free_dofs(bc::PGDBC,dim::Int) = bc.free[dim]

function displacementBC(U) # completely fixed

    x_mode_dofs = collect(1:U.components[1].mesh.nDofs)
    y_mode_dofs = collect((U.components[1].mesh.nDofs+1):(U.components[1].mesh.nDofs+U.components[2].mesh.nDofs))

    # bc_Ux: Lock y_mode_dofs
    fixed_Ux = Int[x_mode_dofs[[1,2]];
                   x_mode_dofs[[end-1,end]];
                   y_mode_dofs]
    prescr_Ux = Int[]
    free_Ux = setdiff(x_mode_dofs,[fixed_Ux; prescr_Ux])

    # bc_Uy: Lock x_mode_dofs
    fixed_Uy = Int[x_mode_dofs;
                   y_mode_dofs[[1,2]];
                   y_mode_dofs[[end-1,end]]]
    prescr_Uy = Int[]
    free_Uy = setdiff(y_mode_dofs,[fixed_Uy; prescr_Uy])

    # bc_U: Total bc's
    fixed_U = Int[x_mode_dofs[[1,2]];
               x_mode_dofs[[end-1,end]];
               y_mode_dofs[[1,2]];
               y_mode_dofs[[end-1,end]]]
    prescr_U = Int[]
    free_U = setdiff([x_mode_dofs; y_mode_dofs], [fixed_U; prescr_U])

    # Combine and return
    fixed = Vector{Int}[fixed_Ux,fixed_Uy,fixed_U]
    prescr = Vector{Int}[prescr_Ux,prescr_Uy,prescr_U]
    free = Vector{Int}[free_Ux,free_Uy,free_U]

    # Create a dirichletmode which fulfills the non-homogeneous Dirichlet bc's
    Ux_dof = 1:2:(U.components[1].mesh.nDofs-1)
    Vx_dof = 2:2:(U.components[1].mesh.nDofs)
    Uy_dof = (U.components[1].mesh.nDofs+1):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs-1)
    Vy_dof = (U.components[1].mesh.nDofs+2):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs)

    Ux_dirichlet = float(zeros(Ux_dof))
    Uy_dirichlet = float(zeros(Uy_dof))

    Vx_dirichlet = float(ones(Vx_dof))
    Vy_dirichlet = U.components[2].mesh.x / maximum(U.components[2].mesh.x)

    dirichletmode = [zipper(Ux_dirichlet,Vx_dirichlet); zipper(Uy_dirichlet,Vy_dirichlet)]

    return PGDBC(fixed,prescr,free), dirichletmode
end

# function displacementBC(U) # fixed at y = 0

#     x_mode_dofs = collect(1:U.components[1].mesh.nDofs)
#     y_mode_dofs = collect((U.components[1].mesh.nDofs+1):(U.components[1].mesh.nDofs+U.components[2].mesh.nDofs))

#     # bc_Ux: Lock y_mode_dofs
#     fixed_Ux = Int[y_mode_dofs;]
#     prescr_Ux = Int[]
#     free_Ux = setdiff(x_mode_dofs,[fixed_Ux; prescr_Ux])

#     # bc_Uy: Lock x_mode_dofs
#     fixed_Uy = Int[x_mode_dofs;
#                    y_mode_dofs[[1,2]]]
#     prescr_Uy = Int[]
#     free_Uy = setdiff(y_mode_dofs,[fixed_Uy; prescr_Uy])

#     # bc_U: Total bc's
#     fixed_U = Int[y_mode_dofs[[1,2]];]
#     prescr_U = Int[y_mode_dofs[end];]
#     free_U = setdiff([x_mode_dofs; y_mode_dofs], [fixed_U; prescr_U])

#     # Combine and return
#     fixed = Vector{Int}[fixed_Ux,fixed_Uy,fixed_U]
#     prescr = Vector{Int}[prescr_Ux,prescr_Uy,prescr_U]
#     free = Vector{Int}[free_Ux,free_Uy,free_U]

#     return PGDBC(fixed,prescr,free)
# end
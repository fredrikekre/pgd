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


###################
# Displacement BC #
###################

# function U_BC(U::PGDFunction) # completely fixed, moving top up

#     x_mode_dofs = collect(1:U.components[1].mesh.nDofs)
#     y_mode_dofs = collect((U.components[1].mesh.nDofs+1):(U.components[1].mesh.nDofs+U.components[2].mesh.nDofs))

#     # bc_Ux: Lock y_mode_dofs
#     fixed_Ux = Int[x_mode_dofs[[1,2,end-1,end]];
#                    y_mode_dofs]
#     prescr_Ux = Int[]
#     free_Ux = setdiff(x_mode_dofs,[fixed_Ux; prescr_Ux])

#     # bc_Uy: Lock x_mode_dofs
#     fixed_Uy = Int[x_mode_dofs;
#                    y_mode_dofs[[1,2,end-1,end]]]
#     prescr_Uy = Int[]
#     free_Uy = setdiff(y_mode_dofs,[fixed_Uy; prescr_Uy])

#     # bc_U: Total bc's
#     fixed_U = Int[x_mode_dofs[[1,2,end-1,end]];
#                   y_mode_dofs[[1,2,end-1,end]]]
#     prescr_U = Int[]
#     free_U = setdiff([x_mode_dofs; y_mode_dofs], [fixed_U; prescr_U])

#     # Combine and return
#     fixed = Vector{Int}[fixed_Ux,fixed_Uy,fixed_U]
#     prescr = Vector{Int}[prescr_Ux,prescr_Uy,prescr_U]
#     free = Vector{Int}[free_Ux,free_Uy,free_U]

#     # Create a dirichletmode which fulfills the non-homogeneous Dirichlet bc's
#     Ux_dof = 1:2:(U.components[1].mesh.nDofs-1)
#     Vx_dof = 2:2:(U.components[1].mesh.nDofs)
#     Uy_dof = (U.components[1].mesh.nDofs+1):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs-1)
#     Vy_dof = (U.components[1].mesh.nDofs+2):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs)

#     Ux_dirichlet = float(zeros(Ux_dof))
#     Uy_dirichlet = float(zeros(Uy_dof))

#     Vx_dirichlet = float(ones(Vx_dof))
#     Vy_dirichlet = U.components[2].mesh.x / maximum(U.components[2].mesh.x)

#     dirichletmode = [zipper(Ux_dirichlet,Vx_dirichlet); zipper(Uy_dirichlet,Vy_dirichlet)]

#     return PGDBC(fixed,prescr,free), dirichletmode
# end

function U_BC(U::PGDFunction) # fixed at bottom, moving top up, also fixed in x on top

    x_mode_dofs = collect(1:U.components[1].mesh.nDofs)
    y_mode_dofs = collect((U.components[1].mesh.nDofs+1):(U.components[1].mesh.nDofs+U.components[2].mesh.nDofs))

    # bc_Ux: Lock y_mode_dofs
    fixed_Ux = Int[x_mode_dofs[[]];
                   y_mode_dofs]
    prescr_Ux = Int[]
    free_Ux = setdiff(x_mode_dofs,[fixed_Ux; prescr_Ux])

    # bc_Uy: Lock x_mode_dofs
    fixed_Uy = Int[x_mode_dofs;
                   y_mode_dofs[[1,2,end-1,end]]]
    prescr_Uy = Int[]
    free_Uy = setdiff(y_mode_dofs,[fixed_Uy; prescr_Uy])

    # bc_U: Total bc's
    fixed_U = Int[x_mode_dofs[[]];
                  y_mode_dofs[[1,2,end-1,end]]]
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

# function U_BC(U::PGDFunction) # fixed at bottom, moving top up, not fixed in x at top

#     x_mode_dofs = collect(1:U.components[1].mesh.nDofs)
#     y_mode_dofs = collect((U.components[1].mesh.nDofs+1):(U.components[1].mesh.nDofs+U.components[2].mesh.nDofs))

#     # bc_Ux: Lock y_mode_dofs
#     fixed_Ux = Int[x_mode_dofs[[]];
#                    y_mode_dofs]
#     prescr_Ux = Int[]
#     free_Ux = setdiff(x_mode_dofs,[fixed_Ux; prescr_Ux])

#     # bc_Uy: Lock x_mode_dofs
#     fixed_Uy = Int[x_mode_dofs;
#                    y_mode_dofs[[1,2,end]]]
#     prescr_Uy = Int[]
#     free_Uy = setdiff(y_mode_dofs,[fixed_Uy; prescr_Uy])

#     # bc_U: Total bc's
#     fixed_U = Int[x_mode_dofs[[]];
#                   y_mode_dofs[[1,2,end]]]
#     prescr_U = Int[]
#     free_U = setdiff([x_mode_dofs; y_mode_dofs], [fixed_U; prescr_U])

#     # Combine and return
#     fixed = Vector{Int}[fixed_Ux,fixed_Uy,fixed_U]
#     prescr = Vector{Int}[prescr_Ux,prescr_Uy,prescr_U]
#     free = Vector{Int}[free_Ux,free_Uy,free_U]

#     # Create a dirichletmode which fulfills the non-homogeneous Dirichlet bc's
#     Ux_dof = 1:2:(U.components[1].mesh.nDofs-1)
#     Vx_dof = 2:2:(U.components[1].mesh.nDofs)
#     Uy_dof = (U.components[1].mesh.nDofs+1):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs-1)
#     Vy_dof = (U.components[1].mesh.nDofs+2):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs)

#     Ux_dirichlet = float(zeros(Ux_dof))
#     Uy_dirichlet = float(zeros(Uy_dof))

#     Vx_dirichlet = float(ones(Vx_dof))
#     Vy_dirichlet = U.components[2].mesh.x / maximum(U.components[2].mesh.x)

#     dirichletmode = [zipper(Ux_dirichlet,Vx_dirichlet); zipper(Uy_dirichlet,Vy_dirichlet)]

#     return PGDBC(fixed,prescr,free), dirichletmode
# end

# function U_BC(U::PGDFunction) # fixed at bottom (only in y), moving top up

#     x_mode_dofs = collect(1:U.components[1].mesh.nDofs)
#     y_mode_dofs = collect((U.components[1].mesh.nDofs+1):(U.components[1].mesh.nDofs+U.components[2].mesh.nDofs))

#     # bc_Ux: Lock y_mode_dofs
#     fixed_Ux = Int[x_mode_dofs[[]];
#                    y_mode_dofs]
#     prescr_Ux = Int[]
#     free_Ux = setdiff(x_mode_dofs,[fixed_Ux; prescr_Ux])

#     # bc_Uy: Lock x_mode_dofs
#     fixed_Uy = Int[x_mode_dofs;
#                    y_mode_dofs[[1,2,end]]]
#     prescr_Uy = Int[]
#     free_Uy = setdiff(y_mode_dofs,[fixed_Uy; prescr_Uy])

#     # bc_U: Total bc's
#     fixed_U = Int[x_mode_dofs[[]];
#                   y_mode_dofs[[1,2,end]]]
#     prescr_U = Int[]
#     free_U = setdiff([x_mode_dofs; y_mode_dofs], [fixed_U; prescr_U])

#     # Combine and return
#     fixed = Vector{Int}[fixed_Ux,fixed_Uy,fixed_U]
#     prescr = Vector{Int}[prescr_Ux,prescr_Uy,prescr_U]
#     free = Vector{Int}[free_Ux,free_Uy,free_U]

#     # Create a dirichletmode which fulfills the non-homogeneous Dirichlet bc's
#     Ux_dof = 1:2:(U.components[1].mesh.nDofs-1)
#     Vx_dof = 2:2:(U.components[1].mesh.nDofs)
#     Uy_dof = (U.components[1].mesh.nDofs+1):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs-1)
#     Vy_dof = (U.components[1].mesh.nDofs+2):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs)

#     Ux_dirichlet = float(zeros(Ux_dof))
#     Uy_dirichlet = float(zeros(Uy_dof))

#     Vx_dirichlet = float(ones(Vx_dof))
#     Vy_dirichlet = U.components[2].mesh.x / maximum(U.components[2].mesh.x)

#     dirichletmode = [zipper(Ux_dirichlet,Vx_dirichlet); zipper(Uy_dirichlet,Vy_dirichlet)]

#     return PGDBC(fixed,prescr,free), dirichletmode
# end

# function U_BC(U::PGDFunction) # fixed at bottom, moving top top the right

#     x_mode_dofs = collect(1:U.components[1].mesh.nDofs)
#     y_mode_dofs = collect((U.components[1].mesh.nDofs+1):(U.components[1].mesh.nDofs+U.components[2].mesh.nDofs))

#     # bc_Ux: Lock y_mode_dofs
#     fixed_Ux = Int[x_mode_dofs[[]];
#                    y_mode_dofs]
#     prescr_Ux = Int[]
#     free_Ux = setdiff(x_mode_dofs,[fixed_Ux; prescr_Ux])

#     # bc_Uy: Lock x_mode_dofs
#     fixed_Uy = Int[x_mode_dofs;
#                    y_mode_dofs[[1,2,end-1,end]]]
#     prescr_Uy = Int[]
#     free_Uy = setdiff(y_mode_dofs,[fixed_Uy; prescr_Uy])

#     # bc_U: Total bc's
#     fixed_U = Int[x_mode_dofs[[]];
#                   y_mode_dofs[[1,2,end-1,end]]]
#     prescr_U = Int[]
#     free_U = setdiff([x_mode_dofs; y_mode_dofs], [fixed_U; prescr_U])

#     # Combine and return
#     fixed = Vector{Int}[fixed_Ux,fixed_Uy,fixed_U]
#     prescr = Vector{Int}[prescr_Ux,prescr_Uy,prescr_U]
#     free = Vector{Int}[free_Ux,free_Uy,free_U]

#     # Create a dirichletmode which fulfills the non-homogeneous Dirichlet bc's
#     Ux_dof = 1:2:(U.components[1].mesh.nDofs-1)
#     Vx_dof = 2:2:(U.components[1].mesh.nDofs)
#     Uy_dof = (U.components[1].mesh.nDofs+1):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs-1)
#     Vy_dof = (U.components[1].mesh.nDofs+2):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs)

#     Ux_dirichlet = float(ones(Ux_dof))
#     Uy_dirichlet = U.components[2].mesh.x / maximum(U.components[2].mesh.x)

#     Vx_dirichlet = float(zeros(Vx_dof))
#     Vy_dirichlet = float(zeros(Vy_dof)) # U.components[2].mesh.x / maximum(U.components[2].mesh.x)

#     dirichletmode = [zipper(Ux_dirichlet,Vx_dirichlet); zipper(Uy_dirichlet,Vy_dirichlet)]

#     return PGDBC(fixed,prescr,free), dirichletmode
# end

# function U_BC(U::PGDFunction) # fixed at bot and left, no prescribed values

#     x_mode_dofs = collect(1:U.components[1].mesh.nDofs)
#     y_mode_dofs = collect((U.components[1].mesh.nDofs+1):(U.components[1].mesh.nDofs+U.components[2].mesh.nDofs))

#     # bc_Ux: Lock y_mode_dofs
#     fixed_Ux = Int[x_mode_dofs[[1,2]];
#                    y_mode_dofs]
#     prescr_Ux = Int[]
#     free_Ux = setdiff(x_mode_dofs,[fixed_Ux; prescr_Ux])

#     # bc_Uy: Lock x_mode_dofs
#     fixed_Uy = Int[x_mode_dofs;
#                    y_mode_dofs[[1,2]]]
#     prescr_Uy = Int[]
#     free_Uy = setdiff(y_mode_dofs,[fixed_Uy; prescr_Uy])

#     # bc_U: Total bc's
#     fixed_U = Int[x_mode_dofs[[1,2]];
#                   y_mode_dofs[[1,2]]]
#     prescr_U = Int[]
#     free_U = setdiff([x_mode_dofs; y_mode_dofs], [fixed_U; prescr_U])

#     # Combine and return
#     fixed = Vector{Int}[fixed_Ux,fixed_Uy,fixed_U]
#     prescr = Vector{Int}[prescr_Ux,prescr_Uy,prescr_U]
#     free = Vector{Int}[free_Ux,free_Uy,free_U]

#     # Create a dirichletmode which fulfills the non-homogeneous Dirichlet bc's
#     Ux_dof = 1:2:(U.components[1].mesh.nDofs-1)
#     Vx_dof = 2:2:(U.components[1].mesh.nDofs)
#     Uy_dof = (U.components[1].mesh.nDofs+1):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs-1)
#     Vy_dof = (U.components[1].mesh.nDofs+2):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs)

#     Ux_dirichlet = float(zeros(Ux_dof))
#     Uy_dirichlet = float(zeros(Uy_dof))

#     Vx_dirichlet = 0.0*float(ones(Vx_dof))
#     Vy_dirichlet = 0.0*U.components[2].mesh.x / maximum(U.components[2].mesh.x)

#     dirichletmode = [zipper(Ux_dirichlet,Vx_dirichlet); zipper(Uy_dirichlet,Vy_dirichlet)]

#     return PGDBC(fixed,prescr,free), dirichletmode
# end


#############
# Damage BC #
#############

function D_BC(D::PGDFunction) # No damage from start

    x_mode_dofs = collect(1:D.components[1].mesh.nDofs)
    y_mode_dofs = collect((D.components[1].mesh.nDofs+1):(D.components[1].mesh.nDofs+D.components[2].mesh.nDofs))

    # bc_Dx: Lock y_mode_dofs
    fixed_Dx = Int[x_mode_dofs[[]];
                   y_mode_dofs]
    prescr_Dx = Int[]
    free_Dx = setdiff(x_mode_dofs,[fixed_Dx; prescr_Dx])

    # bc_Dy: Lock x_mode_dofs
    fixed_Dy = Int[x_mode_dofs;
                   y_mode_dofs[[]]]
    prescr_Dy = Int[]
    free_Dy = setdiff(y_mode_dofs,[fixed_Dy; prescr_Dy])

    # bc_D: Total bc's
    fixed_D = Int[x_mode_dofs[[]];
                  y_mode_dofs[[]]]
    prescr_D = Int[]
    free_D = setdiff([x_mode_dofs; y_mode_dofs], [fixed_D; prescr_D])

    # Combine and return
    fixed = Vector{Int}[fixed_Dx,fixed_Dy,fixed_D]
    prescr = Vector{Int}[prescr_Dx,prescr_Dy,prescr_D]
    free = Vector{Int}[free_Dx,free_Dy,free_D]

    # Create a dirichletmode which fulfills the non-homogeneous Dirichlet bc's
    Dx_dof = 1:(D.components[1].mesh.nDofs)
    Dy_dof = (D.components[1].mesh.nDofs+1):(D.components[1].mesh.nDofs+D.components[1].mesh.nDofs)

    Dx_dirichlet = float(zeros(Dx_dof))
    Dy_dirichlet = float(zeros(Dy_dof))

    dirichletmode = [Dx_dirichlet; Dy_dirichlet]

    return PGDBC(fixed,prescr,free), dirichletmode
end

# function D_BC(D::PGDFunction) # Initial crack

#     x_mode_dofs = collect(1:D.components[1].mesh.nDofs)
#     y_mode_dofs = collect((D.components[1].mesh.nDofs+1):(D.components[1].mesh.nDofs+D.components[2].mesh.nDofs))

#     # Dirichlet dofs
#     dirichlet_x_dofs = 1:ceil(Int,length(x_mode_dofs)/2)
#     dirichlet_y_dofs = ceil(Int,length(y_mode_dofs)/2)
#     # dirichlet_y_dofs = [dirichlet_y_dofs-1, dirichlet_y_dofs, dirichlet_y_dofs+1]

#     # bc_Dx: Lock y_mode_dofs
#     fixed_Dx = Int[x_mode_dofs[[]];
#                    y_mode_dofs]
#     # prescr_Dx = Int[dirichlet_x_dofs;]
#     prescr_Dx = Int[]
#     free_Dx = setdiff(x_mode_dofs,[fixed_Dx; prescr_Dx])

#     # bc_Dy: Lock x_mode_dofs
#     fixed_Dy = Int[x_mode_dofs;
#                    y_mode_dofs[[]]]
#     # prescr_Dy = Int[dirichlet_y_dofs;]
#     prescr_Dy = Int[]
#     free_Dy = setdiff(y_mode_dofs,[fixed_Dy; prescr_Dy])

#     # bc_D: Total bc's
#     fixed_D = Int[x_mode_dofs[[]];
#                   y_mode_dofs[[]]]
#     # prescr_D = Int[dirichlet_y_dofs;
#     #                dirichlet_y_dofs;]
#     prescr_D = Int[]
#     free_D = setdiff([x_mode_dofs; y_mode_dofs], [fixed_D; prescr_D])

#     # Combine and return
#     fixed = Vector{Int}[fixed_Dx,fixed_Dy,fixed_D]
#     prescr = Vector{Int}[prescr_Dx,prescr_Dy,prescr_D]
#     free = Vector{Int}[free_Dx,free_Dy,free_D]

#     # Create a dirichletmode which fulfills the non-homogeneous Dirichlet bc's
#     Dx_dof = 1:(D.components[1].mesh.nDofs)
#     Dy_dof = (D.components[1].mesh.nDofs+1):(D.components[1].mesh.nDofs+D.components[1].mesh.nDofs)

#     Dx_dirichlet = float(zeros(Dx_dof))
#     Dy_dirichlet = float(zeros(Dy_dof))

#     Dx_dirichlet[dirichlet_x_dofs] = 1.0
#     Dy_dirichlet[dirichlet_y_dofs] = 1.0
#     # Dy_dirichlet[dirichlet_y_dofs] = [0.5,1.0,0.5]

#     dirichletmode = [Dx_dirichlet; Dy_dirichlet]

#     return PGDBC(fixed,prescr,free), dirichletmode
# end

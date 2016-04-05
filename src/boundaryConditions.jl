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

function displacementBC(U) # For 2D

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

    return PGDBC(fixed,prescr,free)
end
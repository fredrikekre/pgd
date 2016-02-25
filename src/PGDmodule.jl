# Module PGD

#abstract PGDfunction
#abstract PGDcomponent

# TODO: Maybe split the mesh from the components

type PGDComponent{T, dim_comp, functionspace_comp} #<: PGDcomponent # One component function of the PGD
    dim::Int
    mesh::mesh1D
    fev::JuAFEM.FEValues{T, dim_comp, functionspace_comp}
end
# Shold have gauss rules etc

type PGDFunction{T, dim_comp, dim, functionspace_comp, functionspace_main}
    dim::Int # Number of dimensions
    nComp::Int # Number of components
    mesh::mesh2D # Governing mesh
    fev::JuAFEM.FEValues{T, dim, functionspace_main}
    components::Array{PGDComponent{T, dim_comp, functionspace_comp}} # Array with the different components
    link::Vector{Matrix{Int}}
    modes::Int
end

function PGDFunction{T_PGD <: PGDComponent}(dim::Int, nComp::Int, mesh::mesh, fev::JuAFEM.FEValues,components::Array{T_PGD})
    PGDtemp = PGDFunction(dim,nComp,mesh,fev,components,[zeros(Int, 1,1) for i in 1:2],0)
    return PGDtemp
    link = create_link(PGDtemp)::Vector{Matrix{Int64}}
    return PGDFunction(dim,nComp,mesh,fev,components,link,0) # no computed modes at setup
end

nEl(pgd::PGDFunction) = pgd.mesh.nEl
nModes(pgd::PGDFunction) = pgd.modes # Number of already computed modes
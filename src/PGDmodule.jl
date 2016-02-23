# Module PGD

#abstract PGDfunction
#abstract PGDcomponent

# TODO: Maybe split the mesh from the components

type PGDComponent #<: PGDcomponent # One component function of the PGD
    dim::Int
    mesh::mesh
    fev::JuAFEM.FEValues
end
# Shold have gauss rules etc

type PGDFunction
    dim::Int64 # Number of dimensions
    nComp::Int64 # Number of components
    mesh::mesh # Governing mesh
    fev::JuAFEM.FEValues
    components::Array{PGDComponent} # Array with the different components
    link::Array{Array{Int64,2},1}
    modes::Int64
end

function PGDFunction(dim::Int64,nComp::Int64,mesh::mesh,fev::JuAFEM.FEValues,components::Array{PGDComponent})
    PGDtemp = PGDFunction(dim,nComp,mesh,fev,components,Array{Int}[zeros(1,1) for i in 1:2],0)
    link = create_link(PGDtemp)
    return PGDFunction(dim,nComp,mesh,fev,components,link,0) # no computed modes at setup
end

nEl(pgd::PGDFunction) = pgd.mesh.nEl
nModes(pgd::PGDFunction) = pgd.modes # Number of already computed modes
# Module PGD

#abstract PGDfunction
#abstract PGDcomponent

# TODO: Maybe split the mesh from the components

type PGDComponent{dim, T, functionspace} #<: PGDcomponent # One component function of the PGD
    mesh::mesh1D
    fev::JuAFEM.FEValues{dim, T, functionspace}
    compdim::Int
    totaldim::Int
end

function vtk_grid_no_compress(topology::Matrix{Int}, Coord::Matrix, filename::AbstractString)

    nele = size(topology, 2)
    nen = size(topology,1)
    nnodes = size(Coord, 2)
    ndim = size(Coord, 1)

    cell = JuAFEM.get_cell_type(nen, ndim)

    points = Coord
    points = JuAFEM.pad_zeros(points, ndim, nnodes)

    cells = MeshCell[MeshCell(cell, topology[:,i]) for i = 1:nele]

    vtk = vtk_grid(filename, points, cells, compress=false, append=false)
    return vtk
end

function vtkwriter(pvd,i,u_mesh,u,d,Ψ,σe)

    nnodes = div(length(u),2)

    displacement = reshape(u,(2,nnodes))
    displacement = [displacement; zeros(nnodes)']

    vtkfile = vtk_grid_no_compress(u_mesh.mesh,u_mesh.coord,"./vtkfiles_damage/step_$i")
    vtk_point_data(vtkfile,displacement,"displacement")
    vtk_point_data(vtkfile,d,"damage")
    vtk_cell_data(vtkfile, Ψ, "energy")
    vtk_cell_data(vtkfile, σe, "effective stress")
    collection_add_timestep(pvd,vtkfile,float(i))

end

function vtkwriter(pvd,i,u_mesh,u,d)

    nnodes = div(length(u),2)

    displacement = reshape(u,(2,nnodes))
    displacement = [displacement; zeros(nnodes)']

    vtkfile = vtk_grid_no_compress(u_mesh.mesh,u_mesh.coord,"./vtkfiles_damage/step_$i")
    vtk_point_data(vtkfile,displacement,"displacement")
    vtk_point_data(vtkfile,d,"damage")
    collection_add_timestep(pvd,vtkfile,float(i))

end

function vtkwriter(pvd,i,u_mesh,u)

    nnodes = div(length(u),2)

    displacement = reshape(u,(2,nnodes))
    displacement = [displacement; zeros(nnodes)']

    vtkfile = vtk_grid_no_compress(u_mesh.mesh,u_mesh.coord,"./vtkfiles_elastic/step_$i")
    vtk_point_data(vtkfile,displacement,"displacement")
    collection_add_timestep(pvd,vtkfile,float(i))



end

function save_pgd_format(u)
    # Save to compare with PGD
    nnodes = div(length(u),2)

    nxnodes = Int(nnodes^(1/2))
    nynodes = Int(nnodes^(1/2))

    uu = u[1:2:end-1]
    uu = reshape(uu,(nxnodes,nynodes))

    vv = u[2:2:end]
    vv = reshape(vv,(nxnodes,nynodes))

    meshsize = (nxnodes-1, nynodes-1)

    writedlm("../../../../FinalReport/DataPlots/raw_data/elastic_case2/u_FEM_$(meshsize[1])_$(meshsize[2]).txt", uu)
    writedlm("../../../../FinalReport/DataPlots/raw_data/elastic_case2/v_FEM_$(meshsize[1])_$(meshsize[2]).txt", vv)
end

function save_pgd_format(u, d)
    # Save to compare with PGD
    nnodes = div(length(u),2)

    nxnodes = Int(nnodes^(1/2))
    nynodes = Int(nnodes^(1/2))

    uu = u[1:2:end-1]
    uu = reshape(uu,(nxnodes,nynodes))

    vv = u[2:2:end]
    vv = reshape(vv,(nxnodes,nynodes))

    meshsize = (nxnodes-1, nynodes-1)

    writedlm("../../../../FinalReport/DataPlots/raw_data/damage_case1/loadstep_30/u_FEM_$(meshsize[1])_$(meshsize[2]).txt", uu)
    writedlm("../../../../FinalReport/DataPlots/raw_data/damage_case1/loadstep_30/v_FEM_$(meshsize[1])_$(meshsize[2]).txt", vv)

    # Damage
    nnodes = length(d)

    nxnodes = Int(nnodes^(1/2))
    nynodes = Int(nnodes^(1/2))

    dd = reshape(d,(nxnodes,nynodes))

    meshsize = (nxnodes-1, nynodes-1)

    writedlm("../../../../FinalReport/DataPlots/raw_data/damage_case1/loadstep_30/d_FEM_$(meshsize[1])_$(meshsize[2]).txt", dd)
end

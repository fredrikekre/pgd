function vtkwriter(pvd,i,u_mesh,u,d,Ψ,σe)

    nnodes = div(length(u),2)

    displacement = reshape(u,(2,nnodes))
    displacement = [displacement; zeros(nnodes)']

    dof = reshape(1:2*nnodes,(2,nnodes))

    vtkfile = vtk_grid(u_mesh.mesh,u_mesh.coord,"./vtkfiles/step_$i")
    vtk_point_data(vtkfile,displacement,"displacement")
    vtk_point_data(vtkfile,d,"damage")
    vtk_cell_data(vtkfile, Ψ, "energy")
    vtk_cell_data(vtkfile, σe, "effective stress")
    collection_add_timestep(pvd,vtkfile,float(i))

    vtkfile2 = vtk_grid(u_mesh.mesh,u_mesh.coord,"./vtkfiles/Stepppp_$i")
    vtk_point_data(vtkfile2,displacement,"displacement")
    vtk_point_data(vtkfile2,d,"damage")
    vtk_cell_data(vtkfile, Ψ, "energy")
    vtk_cell_data(vtkfile, σe, "effective stress")
    vtk_save(vtkfile2)


end
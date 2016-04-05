function vtkwriter(pvd,a,U,step)
    nModes = size(a,2)

    Ux_dof = 1:2:(U.components[1].mesh.nDofs-1)
    Vx_dof = 2:2:(U.components[1].mesh.nDofs)
    Uy_dof = (U.components[1].mesh.nDofs+1):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs-1)
    Vy_dof = (U.components[1].mesh.nDofs+2):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs)

    Ux = a[Ux_dof,:]
    Vx = a[Vx_dof,:]

    Uy = a[Uy_dof,:]
    Vy = a[Vy_dof,:]

    u = Ux*Uy'
    v = Vx*Vy'

    x = U.components[1].mesh.x; x = repmat(x,1,length(U.components[2].mesh.x))'
    y = U.components[2].mesh.x; y = repmat(y,1,length(U.components[1].mesh.x))

    X = zeros(U.components[1].mesh.nEl+1,U.components[2].mesh.nEl+1,1)
    Y = copy(X); Z = copy(X)
    X[:,:,1] = x
    Y[:,:,1] = y

    # Write VTK to show in ParaView
    vtkfile = vtk_grid("./resultfiles/step_$step",X,Y,Z,compress=false,append=false) # no compress and append due to Zlib problem
    vtkdisp = zeros(3,U.components[1].mesh.nEl+1,U.components[2].mesh.nEl+1,1)
    vtkdisp[1,:,:,1] = u
    vtkdisp[2,:,:,1] = v
    vtk_point_data(vtkfile, vtkdisp, "displacement")

    collection_add_timestep(pvd,vtkfile,float(step))

end
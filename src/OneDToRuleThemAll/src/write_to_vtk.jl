function vtkwriter(pvd,loadstep,Ux,aX,Uy,aY,Uz,aZ)

    # Component grids
    x = Ux.mesh.x
    x = reinterpret(Float64,x,(length(x),))

    y = Uy.mesh.x
    y = reinterpret(Float64,y,(length(y),))

    z = Uz.mesh.x
    z = reinterpret(Float64,z,(length(z),))

    # Global grid
    X = vec_mul_vec_mul_vec(x,ones(y),ones(z))
    Y = vec_mul_vec_mul_vec(ones(x),y,ones(z))
    Z = vec_mul_vec_mul_vec(ones(x),ones(y),z)

    vtkfile = vtk_grid("./vtkfiles/step$(loadstep)",X,Y,Z)

    # Displacement
    U, V, W = build_function(Ux, aX, Uy, aY, Uz, aZ)

    disp = zeros(3,size(U)...)
    disp[1,:,:,:] = U
    disp[2,:,:,:] = V
    disp[3,:,:,:] = W

    vtk_point_data(vtkfile,disp,"displacement")

    collection_add_timestep(pvd, vtkfile, float(loadstep))
end

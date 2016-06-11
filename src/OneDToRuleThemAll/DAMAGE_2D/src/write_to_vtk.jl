#############
# 2D damage #
#############
function vtkwriter(pvd,loadstep,Ux,u_ax,Uy,u_ay,Dx,d_ax,Dy,d_ay)

    # Component grids
    x = Ux.mesh.x
    x = reinterpret(Float64,x,(length(x),))

    y = Uy.mesh.x
    y = reinterpret(Float64,y,(length(y),))

    # Global grid
    X = vec_mul_vec_mul_vec(x,ones(y),[1.0])
    Y = vec_mul_vec_mul_vec(ones(x),y,[1.0])
    Z = vec_mul_vec_mul_vec(ones(x),ones(y),[0.0])

    vtkfile = vtk_grid(pvd.path[1:end-14]*"step$(loadstep)",X,Y,Z)

    # Displacement
    U, V = build_u_function(Ux, u_ax, Uy, u_ay)

    disp = zeros(3,size(U)...)
    disp[1,:,:,:] = U
    disp[2,:,:,:] = V

    vtk_point_data(vtkfile,disp,"displacement")

    # Damage
    D = build_d_function(Dx, d_ax, Dy, d_ay)

    vtk_point_data(vtkfile,D,"damage")

    collection_add_timestep(pvd, vtkfile, float(loadstep))
end

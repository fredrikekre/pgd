#############
# 3D damage #
#############
function vtkwriter(pvd,loadstep,Ux,u_ax,Uy,u_ay,Uz,u_az,Dx,d_ax,Dy,d_ay,Dz,d_az)

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

    vtkfile = vtk_grid(pvd.path[1:end-14]*"step$(loadstep)",X,Y,Z)

    # Displacement
    U, V, W = build_u_function(Ux, u_ax, Uy, u_ay, Uz, u_az)

    disp = zeros(3,size(U)...)
    disp[1,:,:,:] = U
    disp[2,:,:,:] = V
    disp[3,:,:,:] = W

    vtk_point_data(vtkfile,disp,"displacement")

    # Damage
    D = build_d_function(Dx, d_ax, Dy, d_ay, Dz, d_az)

    vtk_point_data(vtkfile,D,"damage")

    collection_add_timestep(pvd, vtkfile, float(loadstep))
end

# ######
# # 2D #
# ######
# function vtkwriter(pvd,loadstep,Ux,aX,Uy,aY)

#     # Component grids
#     x = Ux.mesh.x
#     x = reinterpret(Float64,x,(length(x),))

#     y = Uy.mesh.x
#     y = reinterpret(Float64,y,(length(y),))

#     # Global grid
#     X = vec_mul_vec_mul_vec(x,ones(y),[1.0])
#     Y = vec_mul_vec_mul_vec(ones(x),y,[1.0])
#     Z = vec_mul_vec_mul_vec(ones(x),ones(y),[0.0])

#     vtkfile = vtk_grid(pvd.path[1:end-14]*"step$(loadstep)",X,Y,Z)

#     # Displacement
#     U, V = build_function(Ux, aX, Uy, aY)

#     disp = zeros(3,size(U)...)
#     disp[1,:,:,:] = U
#     disp[2,:,:,:] = V

#     vtk_point_data(vtkfile,disp,"displacement")

#     collection_add_timestep(pvd, vtkfile, float(loadstep))
# end

function vtk_writer_3D(xs,xe,nel,u)

    # Set up grid
    xnodes = nel[1] + 1
    ynodes = nel[2] + 1
    znodes = nel[3] + 1

    x = linspace(xs[1],xe[1],xnodes)
    y = linspace(xs[2],xe[2],ynodes)
    z = linspace(xs[3],xe[3],znodes)

    X = zeros(xnodes,ynodes,znodes)
    Y = zeros(xnodes,ynodes,znodes)
    Z = zeros(xnodes,ynodes,znodes)

    for xi in 1:xnodes, yi in 1:ynodes, zi in 1:znodes
        X[xi,yi,zi] = x[xi]
        Y[xi,yi,zi] = y[yi]
        Z[xi,yi,zi] = z[zi]
    end

    vtkfile = vtk_grid("3D_output", X, Y, Z)

    # Displacement
    ux = u[1:3:(end-2)]
    uy = u[2:3:(end-1)]
    uz = u[3:3:(end-0)]

    ux = reshape(ux,(xnodes,ynodes,znodes))
    uy = reshape(uy,(xnodes,ynodes,znodes))
    uz = reshape(uz,(xnodes,ynodes,znodes))

    U = zeros(3,size(ux)...)
    U[1,:,:,:] = ux
    U[2,:,:,:] = uy
    U[3,:,:,:] = uz

    vtk_point_data(vtkfile, U, "displacement")
    vtk_save(vtkfile)
end
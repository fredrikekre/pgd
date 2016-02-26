import PyPlot


function visualize(a,U)
    # Plots the modes

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

    for i = 1:nModes
        PyPlot.figure(1)
        PyPlot.title("Ux")
        PyPlot.plot(U.components[1].mesh.x,Ux[:,i])
        PyPlot.figure(2)
        PyPlot.title("Vx")
        PyPlot.plot(U.components[1].mesh.x,Vx[:,i])
        PyPlot.figure(3)
        PyPlot.title("Uy")
        PyPlot.plot(U.components[2].mesh.x,Uy[:,i])
        PyPlot.figure(4)
        PyPlot.title("Vy")
        PyPlot.plot(U.components[2].mesh.x,Vy[:,i])
    end

    # return 1


    # Plot displacement field
    x = U.components[1].mesh.x; x = repmat(x,1,length(U.components[2].mesh.x))'
    y = U.components[2].mesh.x; y = repmat(y,1,length(U.components[1].mesh.x))

    #figure(5)
    #plot_wireframe(x,y ,0*u,color = "red")
    #plot_wireframe(x+u,y+v ,0*u)
    #figure(6)
    #plot_wireframe(X,Y,v)
    #figure(7)
    #uu = (u.^2 + v.^2).^(1/2)
    #plot_wireframe(X,Y,uu)
    #return u,v,X,Y

    # Plotta error mot FEM
    # figure(6)
    # semilogy(1:nModes,FEerror[1,:]',"-o",label = "u100")
    # semilogy(1:nModes,FEerror[2,:]',"-s",label = "u1000")
    # semilogy(1:nModes,FEerror[3,:]',"-^",label = "v100")
    # semilogy(1:nModes,FEerror[4,:]',"-D",label = "v1000")

    # legend(bbox_to_anchor=(1,0),loc=4)
    # xlabel("Number of modes")
    # ylabel("|uPGD-uFEM|/|uFEM|")

    # Write VTK to show in ParaView
    X = zeros(U.components[1].mesh.nEl+1,U.components[2].mesh.nEl+1,1)
    Y = copy(X); Z = copy(X)
    X[:,:,1] = x
    Y[:,:,1] = y
    vtkfile = JuAFEM.vtk_grid("PGD_disp_newton", X,Y,Z)
    vtkdisp = zeros(3,U.components[1].mesh.nEl+1,U.components[2].mesh.nEl+1,1)
    vtkdisp[1,:,:,1] = u
    vtkdisp[2,:,:,1] = v
    JuAFEM.vtk_point_data(vtkfile, vtkdisp, "displacement_newton")
    # #println("Norm of error u = $(norm(u-uFEM1000)/norm(uFEM1000))")
    # #println("Norm of error v = $(norm(v-vFEM1000)/norm(vFEM1000))")

    # vtkdisp = zeros(3,xmesh.nEl+1,ymesh.nEl+1,1)
    # vtkdisp[1,:,:,1] = u-uFEM100
    # vtkdisp[2,:,:,1] = v-vFEM100
    # JuAFEM.vtk_point_data(vtkfile, vtkdisp, "PGD-FEM100")
    
    # vtkdisp = zeros(3,xmesh.nEl+1,ymesh.nEl+1,1)
    # vtkdisp[1,:,:,1] = u-uFEM1000
    # vtkdisp[2,:,:,1] = v-vFEM1000
    # JuAFEM.vtk_point_data(vtkfile, vtkdisp, "PGD-FEM1000")

    JuAFEM.vtk_save(vtkfile)



end

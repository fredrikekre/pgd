function postprocesser_main_elastic_3D_1D_integration(aX, aY, aZ, Ux, Uy, Uz)

    dir = "./results_main_elastic_3D_1D_integration/"

    # Save the modes
    writedlm(dir*"xmodes.txt", aX)
    xcoords = Ux.mesh.x
    xcoords = reinterpret(Float64,xcoords,(length(xcoords),))
    writedlm(dir*"xcoords.txt", xcoords)

    writedlm(dir*"ymodes.txt", aY)
    ycoords = Uy.mesh.x
    ycoords = reinterpret(Float64,ycoords,(length(ycoords),))
    writedlm(dir*"ycoords.txt", ycoords)

    writedlm(dir*"zmodes.txt", aZ)
    zcoords = Uz.mesh.x
    zcoords = reinterpret(Float64,zcoords,(length(zcoords),))
    writedlm(dir*"zcoords.txt", zcoords)

    # Save the displacement
    number_of_modes = length(aX)
    meshsizex = Ux.mesh.nEl
    meshsizey = Uy.mesh.nEl
    meshsizez = Uz.mesh.nEl

    Ux_dof = 1:3:(Ux.mesh.nDofs-2)
    Vx_dof = 2:3:(Ux.mesh.nDofs-1)
    Wx_dof = 3:3:(Ux.mesh.nDofs-0)

    Uy_dof = 1:3:(Uy.mesh.nDofs-2)
    Vy_dof = 2:3:(Uy.mesh.nDofs-1)
    Wy_dof = 3:3:(Uy.mesh.nDofs-0)

    Uz_dof = 1:3:(Uz.mesh.nDofs-2)
    Vz_dof = 2:3:(Uz.mesh.nDofs-1)
    Wz_dof = 3:3:(Uz.mesh.nDofs-0)

    u = zero(vec_mul_vec_mul_vec(aX[1][Ux_dof], aY[1][Uy_dof], aZ[1][Uz_dof]))
    v = zero(vec_mul_vec_mul_vec(aX[1][Vx_dof], aY[1][Vy_dof], aZ[1][Vz_dof]))
    w = zero(vec_mul_vec_mul_vec(aX[1][Wx_dof], aY[1][Wy_dof], aZ[1][Wz_dof]))

    reshape_to_this = [size(u)...]

    for i in 1:number_of_modes
        u += vec_mul_vec_mul_vec(aX[i][Ux_dof], aY[i][Uy_dof], aZ[i][Uz_dof])
        v += vec_mul_vec_mul_vec(aX[i][Vx_dof], aY[i][Vy_dof], aZ[i][Vz_dof])
        w += vec_mul_vec_mul_vec(aX[i][Wx_dof], aY[i][Wy_dof], aZ[i][Wz_dof])

        u_write = [reshape_to_this; u[:]]
        v_write = [reshape_to_this; v[:]]
        w_write = [reshape_to_this; w[:]]

        writedlm(dir*"u_PGD_$(i)_modes_$(meshsizex)x$(meshsizey)x$(meshsizez).txt", u_write)
        writedlm(dir*"v_PGD_$(i)_modes_$(meshsizex)x$(meshsizey)x$(meshsizez).txt", v_write)
        writedlm(dir*"w_PGD_$(i)_modes_$(meshsizex)x$(meshsizey)x$(meshsizez).txt", w_write)

    end

end

function postprocesser_main_elastic_3D_3D_integration(aX, aY, aZ, Ux, Uy, Uz)

    dir = "./results_main_elastic_3D_3D_integration/"

    # Save the modes
    writedlm(dir*"xmodes.txt", aX)
    xcoords = Ux.mesh.x
    xcoords = reinterpret(Float64,xcoords,(length(xcoords),))
    writedlm(dir*"xcoords.txt", xcoords)

    writedlm(dir*"ymodes.txt", aY)
    ycoords = Uy.mesh.x
    ycoords = reinterpret(Float64,ycoords,(length(ycoords),))
    writedlm(dir*"ycoords.txt", ycoords)

    writedlm(dir*"zmodes.txt", aZ)
    zcoords = Uz.mesh.x
    zcoords = reinterpret(Float64,zcoords,(length(zcoords),))
    writedlm(dir*"zcoords.txt", zcoords)

    # Save the displacement
    number_of_modes = length(aX)
    meshsizex = Ux.mesh.nEl
    meshsizey = Uy.mesh.nEl
    meshsizez = Uz.mesh.nEl

    Ux_dof = 1:3:(Ux.mesh.nDofs-2)
    Vx_dof = 2:3:(Ux.mesh.nDofs-1)
    Wx_dof = 3:3:(Ux.mesh.nDofs-0)

    Uy_dof = 1:3:(Uy.mesh.nDofs-2)
    Vy_dof = 2:3:(Uy.mesh.nDofs-1)
    Wy_dof = 3:3:(Uy.mesh.nDofs-0)

    Uz_dof = 1:3:(Uz.mesh.nDofs-2)
    Vz_dof = 2:3:(Uz.mesh.nDofs-1)
    Wz_dof = 3:3:(Uz.mesh.nDofs-0)

    u = zero(vec_mul_vec_mul_vec(aX[1][Ux_dof], aY[1][Uy_dof], aZ[1][Uz_dof]))
    v = zero(vec_mul_vec_mul_vec(aX[1][Vx_dof], aY[1][Vy_dof], aZ[1][Vz_dof]))
    w = zero(vec_mul_vec_mul_vec(aX[1][Wx_dof], aY[1][Wy_dof], aZ[1][Wz_dof]))

    reshape_to_this = [size(u)...]

    for i in 1:number_of_modes
        u += vec_mul_vec_mul_vec(aX[i][Ux_dof], aY[i][Uy_dof], aZ[i][Uz_dof])
        v += vec_mul_vec_mul_vec(aX[i][Vx_dof], aY[i][Vy_dof], aZ[i][Vz_dof])
        w += vec_mul_vec_mul_vec(aX[i][Wx_dof], aY[i][Wy_dof], aZ[i][Wz_dof])

        u_write = [reshape_to_this; u[:]]
        v_write = [reshape_to_this; v[:]]
        w_write = [reshape_to_this; w[:]]

        writedlm(dir*"u_PGD_$(i)_modes_$(meshsizex)x$(meshsizey)x$(meshsizez).txt", u_write)
        writedlm(dir*"v_PGD_$(i)_modes_$(meshsizex)x$(meshsizey)x$(meshsizez).txt", v_write)
        writedlm(dir*"w_PGD_$(i)_modes_$(meshsizex)x$(meshsizey)x$(meshsizez).txt", w_write)

    end

end


function postprocesser_main_elastic_2D_1D_integration(aX, aY, Ux, Uy)

    dir = "./results_main_elastic_2D_1D_integration/"

    # Save the modes
    writedlm(dir*"xmodes.txt", aX)
    xcoords = Ux.mesh.x
    xcoords = reinterpret(Float64,xcoords,(length(xcoords),))
    writedlm(dir*"xcoords.txt", xcoords)

    writedlm(dir*"ymodes.txt", aY)
    ycoords = Uy.mesh.x
    ycoords = reinterpret(Float64,ycoords,(length(ycoords),))
    writedlm(dir*"ycoords.txt", ycoords)

    # Save the displacement
    number_of_modes = length(aX)
    meshsizex = Ux.mesh.nEl
    meshsizey = Uy.mesh.nEl

    Ux_dof = 1:2:(Ux.mesh.nDofs-1)
    Vx_dof = 2:2:(Ux.mesh.nDofs-0)
    Uy_dof = 1:2:(Uy.mesh.nDofs-1)
    Vy_dof = 2:2:(Uy.mesh.nDofs-0)

    u = zero(aX[1][Ux_dof]*aY[1][Uy_dof]')
    v = zero(aX[1][Vx_dof]*aY[1][Vy_dof]')

    for i in 1:number_of_modes
        u += aX[i][Ux_dof]*aY[i][Uy_dof]'
        v += aX[i][Vx_dof]*aY[i][Vy_dof]'

        writedlm(dir*"u_PGD_$(i)_modes_$(meshsizex)x$(meshsizey).txt", u)
        writedlm(dir*"v_PGD_$(i)_modes_$(meshsizex)x$(meshsizey).txt", v)
    end

end

function postprocesser_main_elastic_2D_2D_integration(aX, aY, Ux, Uy)

    dir = "./results_main_elastic_2D_2D_integration/"

    # Save the modes
    writedlm(dir*"xmodes.txt", aX)
    xcoords = Ux.mesh.x
    xcoords = reinterpret(Float64,xcoords,(length(xcoords),))
    writedlm(dir*"xcoords.txt", xcoords)

    writedlm(dir*"ymodes.txt", aY)
    ycoords = Uy.mesh.x
    ycoords = reinterpret(Float64,ycoords,(length(ycoords),))
    writedlm(dir*"ycoords.txt", ycoords)

    # Save the displacement
    number_of_modes = length(aX)
    meshsizex = Ux.mesh.nEl
    meshsizey = Uy.mesh.nEl

    Ux_dof = 1:2:(Ux.mesh.nDofs-1)
    Vx_dof = 2:2:(Ux.mesh.nDofs-0)
    Uy_dof = 1:2:(Uy.mesh.nDofs-1)
    Vy_dof = 2:2:(Uy.mesh.nDofs-0)

    u = zero(aX[1][Ux_dof]*aY[1][Uy_dof]')
    v = zero(aX[1][Vx_dof]*aY[1][Vy_dof]')

    for i in 1:number_of_modes
        u += aX[i][Ux_dof]*aY[i][Uy_dof]'
        v += aX[i][Vx_dof]*aY[i][Vy_dof]'

        writedlm(dir*"u_PGD_$(i)_modes_$(meshsizex)x$(meshsizey).txt", u)
        writedlm(dir*"v_PGD_$(i)_modes_$(meshsizex)x$(meshsizey).txt", v)
    end

end

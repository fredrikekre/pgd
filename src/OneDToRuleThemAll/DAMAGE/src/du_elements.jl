#################################
# Shape functions for damage 3D #
#################################
function get_d_N_3D{dimcomp,T,functionspace}(Dx::PGDComponent{dimcomp,T,functionspace},Dy::PGDComponent{dimcomp,T,functionspace},Dz::PGDComponent{dimcomp,T,functionspace}, global_fe_values)

    ex = Dx.mesh.ex[1]
    reinit!(Dx.fev,ex)
    ey = Dy.mesh.ex[1]
    reinit!(Dy.fev,ey)
    ez = Dz.mesh.ex[1]
    reinit!(Dz.fev,ez)

    node1 = Tensor{1,3}((ex[1][1], ey[1][1], ez[1][1]))
    node2 = Tensor{1,3}((ex[2][1], ey[1][1], ez[1][1]))
    node3 = Tensor{1,3}((ex[2][1], ey[2][1], ez[1][1]))
    node4 = Tensor{1,3}((ex[1][1], ey[2][1], ez[1][1]))
    node5 = Tensor{1,3}((ex[1][1], ey[1][1], ez[2][1]))
    node6 = Tensor{1,3}((ex[2][1], ey[1][1], ez[2][1]))
    node7 = Tensor{1,3}((ex[2][1], ey[2][1], ez[2][1]))
    node8 = Tensor{1,3}((ex[1][1], ey[2][1], ez[2][1]))
    EX = Tensor{1,3,Float64,3}[node1, node2, node3, node4, node5, node6, node7, node8]
    reinit!(global_fe_values, EX)

    nqpx = length(points(Dx.fev.quad_rule))
    nqpy = length(points(Dy.fev.quad_rule))
    nqpz = length(points(Dz.fev.quad_rule))

    d_Nx = Vector{T}[]
    d_Ny = Vector{T}[]
    d_Nz = Vector{T}[]
    d_dNx = Vector{Vec{3,T}}[]
    d_dNy = Vector{Vec{3,T}}[]
    d_dNz = Vector{Vec{3,T}}[]

    for qpx in 1:nqpx, qpy in 1:nqpy, qpz in 1:nqpz
        d_Nx_qp = T[]
        d_Ny_qp = T[]
        d_Nz_qp = T[]
        d_dNx_qp = Vec{3,T}[]
        d_dNy_qp = Vec{3,T}[]
        d_dNz_qp = Vec{3,T}[]

        ##########################
        # Shape functions for Dx #
        ##########################
        for dof in 1:Dx.mesh.nElDofs

            # Shape functions
            d_Nx_ = Dx.fev.N[qpx][dof]
            push!(d_Nx_qp, d_Nx_)

            # Derivatives
            this_d_dNx = Dx.fev.dNdx[qpx][dof][1]
            this_d_dNx_t = [this_d_dNx, d_Nx_, d_Nx_]
            d_dNx_ = Vec{3}((this_d_dNx_t...))
            push!(d_dNx_qp,d_dNx_)
        end
        push!(d_Nx,d_Nx_qp)
        push!(d_dNx,d_dNx_qp)

        ##########################
        # Shape functions for Dy #
        ##########################
        for dof in 1:Dy.mesh.nElDofs

            # Shape functions
            d_Ny_ = Dy.fev.N[qpy][dof]
            push!(d_Ny_qp, d_Ny_)

            # Derivatives
            this_d_dNy = Dy.fev.dNdx[qpy][dof][1]
            this_d_dNy_t = [d_Ny_, this_d_dNy, d_Ny_]
            d_dNy_ = Vec{3}((this_d_dNy_t...))
            push!(d_dNy_qp,d_dNy_)
        end
        push!(d_Ny,d_Ny_qp)
        push!(d_dNy,d_dNy_qp)

        ##########################
        # Shape functions for Dz #
        ##########################
        for dof in 1:Dz.mesh.nElDofs

            # Shape functions
            d_Nz_ = Dz.fev.N[qpz][dof]
            push!(d_Nz_qp, d_Nz_)

            # Derivatives
            this_d_dNz = Dz.fev.dNdx[qpz][dof][1]
            this_d_dNz_t = [d_Nz_, d_Nz_, this_d_dNz]
            d_dNz_ = Vec{3}((this_d_dNz_t...))
            push!(d_dNz_qp,d_dNz_)
        end
        push!(d_Nz,d_Nz_qp)
        push!(d_dNz,d_dNz_qp)
    end

    return d_shapefunctions(d_Nx,d_Ny,d_Nz,d_dNx,d_dNy,d_dNz), global_fe_values
end

###################################################################
# Element functions for 3D damage with energy with 3D integration #
###################################################################
#x
function du_x_ele_3D_3D{T}(fe::Vector{T},Ke::Matrix{T},
                           Dx::PGDComponent,d_ax::Vector{Vector{T}},
                           Dy::PGDComponent,d_ay::Vector{Vector{T}},
                           Dz::PGDComponent,d_az::Vector{Vector{T}},
                           DN::d_shapefunctions{3,Float64},global_d_fe_values::FEValues,
                           dmp::damage_params,Ψ::Vector{Float64})

    # Set up
    nElDofsx = Dx.mesh.nElDofs
    nElDofsy = Dy.mesh.nElDofs
    nElDofsz = Dz.mesh.nElDofs
    fill!(fe,zero(T))
    fill!(Ke,zero(T))

    # Integrate this element
    nqp = length(points(global_d_fe_values.quad_rule))
    number_of_modes = length(d_ax)-1

    for qp in 1:nqp

        # Integration weight
        dV = detJdV(global_d_fe_values,qp)

        # y-sum of test function
        v_x_y = zero(Float64)
        ∇v_x_y = zero(Tensor{1,3,Float64})
        for dofy in 1:nElDofsy
            v_x_y += DN.Ny[qp][dofy] * d_ay[end][dofy]
            ∇v_x_y += DN.dNy[qp][dofy] * d_ay[end][dofy]
        end

        # z-sum of test function
        v_x_z = zero(Float64)
        ∇v_x_z = zero(Tensor{1,3,Float64})
        for dofz in 1:nElDofsz
            v_x_z += DN.Nz[qp][dofz] * d_az[end][dofz]
            ∇v_x_z += DN.dNz[qp][dofz] * d_az[end][dofz]
        end

        for dofx1 in 1:nElDofsx
            # Test function for this dof
            v_x = DN.Nx[qp][dofx1]
            ∇v_x = DN.dNx[qp][dofx1]

            # Add on the y-sym
            v_x = v_x * v_x_y
            ∇v_x = hadamard(∇v_x,∇v_x_y)

            # Add on the z-sym
            v_x = v_x * v_x_z
            ∇v_x = hadamard(∇v_x,∇v_x_z)

            # Energy force
            fe[dofx1] += (v_x * 2*dmp.l/dmp.Gc*Ψ[qp]) * dV

            # Previous modes
            for mode in 1:number_of_modes
                Dxh = zero(Float64)
                dDxh = zero(Tensor{1,3,Float64})
                Dyh = zero(Float64)
                dDyh = zero(Tensor{1,3,Float64})
                Dzh = zero(Float64)
                dDzh = zero(Tensor{1,3,Float64})
                for dofxx in 1:nElDofsx
                    Dxh += DN.Nx[qp][dofxx] * d_ax[mode][dofxx]
                    dDxh += DN.dNx[qp][dofxx] * d_ax[mode][dofxx]
                end
                for dofyy in 1:nElDofsy
                    Dyh += DN.Ny[qp][dofyy] * d_ay[mode][dofyy]
                    dDyh += DN.dNy[qp][dofyy] * d_ay[mode][dofyy]
                end
                for dofzz in 1:nElDofsz
                    Dzh += DN.Nz[qp][dofzz] * d_az[mode][dofzz]
                    dDzh += DN.dNz[qp][dofzz] * d_az[mode][dofzz]
                end
                d_n = Dxh * Dyh * Dzh
                ∇d_n = hadamard(hadamard(dDxh,dDyh),dDzh)

                fe[dofx1] -= (dmp.l^2 * ∇v_x ⋅ ∇d_n + (1+2*dmp.l/dmp.Gc*Ψ[qp]) * v_x * d_n) * dV
            end

            # Stiffness matrix
            for dofx2 in 1:nElDofsx
                # Testfunction for dof 2
                v_x2 = DN.Nx[qp][dofx2]
                ∇v_x2 = DN.dNx[qp][dofx2]
                # Add on y and z sums
                v_x2 = v_x2*v_x_y
                v_x2 = v_x2*v_x_z
                ∇v_x2 = hadamard(∇v_x2,∇v_x_y)
                ∇v_x2 = hadamard(∇v_x2,∇v_x_z)

                Ke[dofx1,dofx2] += (dmp.l^2 * ∇v_x ⋅ ∇v_x2 + (1+2*dmp.l/dmp.Gc*Ψ[qp]) * v_x * v_x2) * dV
            end
        end
    end

    return fe, Ke
end

#y
function du_y_ele_3D_3D{T}(fe::Vector{T},Ke::Matrix{T},
                           Dx::PGDComponent,d_ax::Vector{Vector{T}},
                           Dy::PGDComponent,d_ay::Vector{Vector{T}},
                           Dz::PGDComponent,d_az::Vector{Vector{T}},
                           DN::d_shapefunctions{3,Float64},global_d_fe_values::FEValues,
                           dmp::damage_params,Ψ::Vector{Float64})

    # Set up
    nElDofsx = Dx.mesh.nElDofs
    nElDofsy = Dy.mesh.nElDofs
    nElDofsz = Dz.mesh.nElDofs
    fill!(fe,zero(T))
    fill!(Ke,zero(T))

    # Integrate this element
    nqp = length(points(global_d_fe_values.quad_rule))
    number_of_modes = length(d_ax)-1

    for qp in 1:nqp

        # Integration weight
        dV = detJdV(global_d_fe_values,qp)

        # x-sum of test function
        v_y_x = zero(Float64)
        ∇v_y_x = zero(Tensor{1,3,Float64})
        for dofx in 1:nElDofsx
            v_y_x += DN.Nx[qp][dofx] * d_ax[end][dofx]
            ∇v_y_x += DN.dNx[qp][dofx] * d_ax[end][dofx]
        end

        # z-sum of test function
        v_y_z = zero(Float64)
        ∇v_y_z = zero(Tensor{1,3,Float64})
        for dofz in 1:nElDofsz
            v_y_z += DN.Nz[qp][dofz] * d_az[end][dofz]
            ∇v_y_z += DN.dNz[qp][dofz] * d_az[end][dofz]
        end

        for dofy1 in 1:nElDofsy
            # Test function for this dof
            v_y = DN.Ny[qp][dofy1]
            ∇v_y = DN.dNy[qp][dofy1]

            # Add on the x-sym
            v_y = v_y * v_y_x
            ∇v_y = hadamard(∇v_y,∇v_y_x)

            # Add on the z-sym
            v_y = v_y * v_y_z
            ∇v_y = hadamard(∇v_y,∇v_y_z)

            # Energy force
            fe[dofy1] += (v_y * 2*dmp.l/dmp.Gc*Ψ[qp]) * dV

            # Previous modes
            for mode in 1:number_of_modes
                Dxh = zero(Float64)
                dDxh = zero(Tensor{1,3,Float64})
                Dyh = zero(Float64)
                dDyh = zero(Tensor{1,3,Float64})
                Dzh = zero(Float64)
                dDzh = zero(Tensor{1,3,Float64})
                for dofxx in 1:nElDofsx
                    Dxh += DN.Nx[qp][dofxx] * d_ax[mode][dofxx]
                    dDxh += DN.dNx[qp][dofxx] * d_ax[mode][dofxx]
                end
                for dofyy in 1:nElDofsy
                    Dyh += DN.Ny[qp][dofyy] * d_ay[mode][dofyy]
                    dDyh += DN.dNy[qp][dofyy] * d_ay[mode][dofyy]
                end
                for dofzz in 1:nElDofsz
                    Dzh += DN.Nz[qp][dofzz] * d_az[mode][dofzz]
                    dDzh += DN.dNz[qp][dofzz] * d_az[mode][dofzz]
                end
                d_n = Dxh * Dyh * Dzh
                ∇d_n = hadamard(hadamard(dDxh,dDyh),dDzh)

                fe[dofy1] -= (dmp.l^2 * ∇v_y ⋅ ∇d_n + (1+2*dmp.l/dmp.Gc*Ψ[qp]) * v_y * d_n) * dV
            end

            # Stiffness matrix
            for dofy2 in 1:nElDofsy
                # Testfunction for dof 2
                v_y2 = DN.Ny[qp][dofy2]
                ∇v_y2 = DN.dNy[qp][dofy2]
                # Add on x and z sums
                v_y2 = v_y2*v_y_x
                v_y2 = v_y2*v_y_z
                ∇v_y2 = hadamard(∇v_y2,∇v_y_x)
                ∇v_y2 = hadamard(∇v_y2,∇v_y_z)

                Ke[dofy1,dofy2] += (dmp.l^2 * ∇v_y ⋅ ∇v_y2 + (1+2*dmp.l/dmp.Gc*Ψ[qp]) * v_y * v_y2) * dV
            end
        end
    end

    return fe, Ke
end

#z
function du_z_ele_3D_3D{T}(fe::Vector{T},Ke::Matrix{T},
                           Dx::PGDComponent,d_ax::Vector{Vector{T}},
                           Dy::PGDComponent,d_ay::Vector{Vector{T}},
                           Dz::PGDComponent,d_az::Vector{Vector{T}},
                           DN::d_shapefunctions{3,Float64},global_d_fe_values::FEValues,
                           dmp::damage_params,Ψ::Vector{Float64})

    # Set up
    nElDofsx = Dx.mesh.nElDofs
    nElDofsy = Dy.mesh.nElDofs
    nElDofsz = Dz.mesh.nElDofs
    fill!(fe,zero(T))
    fill!(Ke,zero(T))

    # Integrate this element
    nqp = length(points(global_d_fe_values.quad_rule))
    number_of_modes = length(d_ax)-1

    for qp in 1:nqp

        # Integration weight
        dV = detJdV(global_d_fe_values,qp)

        # x-sum of test function
        v_z_x = zero(Float64)
        ∇v_z_x = zero(Tensor{1,3,Float64})
        for dofx in 1:nElDofsx
            v_z_x += DN.Nx[qp][dofx] * d_ax[end][dofx]
            ∇v_z_x += DN.dNx[qp][dofx] * d_ax[end][dofx]
        end

        # y-sum of test function
        v_z_y = zero(Float64)
        ∇v_z_y = zero(Tensor{1,3,Float64})
        for dofy in 1:nElDofsy
            v_z_y += DN.Ny[qp][dofy] * d_ay[end][dofy]
            ∇v_z_y += DN.dNy[qp][dofy] * d_ay[end][dofy]
        end

        for dofz1 in 1:nElDofsz
            # Test function for this dof
            v_z = DN.Nz[qp][dofz1]
            ∇v_z = DN.dNz[qp][dofz1]

            # Add on the x-sym
            v_z = v_z * v_z_x
            ∇v_z = hadamard(∇v_z,∇v_z_x)

            # Add on the y-sym
            v_z = v_z * v_z_y
            ∇v_z = hadamard(∇v_z,∇v_z_y)

            # Energy force
            fe[dofz1] += (v_z * 2*dmp.l/dmp.Gc*Ψ[qp]) * dV

            # Previous modes
            for mode in 1:number_of_modes
                Dxh = zero(Float64)
                dDxh = zero(Tensor{1,3,Float64})
                Dyh = zero(Float64)
                dDyh = zero(Tensor{1,3,Float64})
                Dzh = zero(Float64)
                dDzh = zero(Tensor{1,3,Float64})
                for dofxx in 1:nElDofsx
                    Dxh += DN.Nx[qp][dofxx] * d_ax[mode][dofxx]
                    dDxh += DN.dNx[qp][dofxx] * d_ax[mode][dofxx]
                end
                for dofyy in 1:nElDofsy
                    Dyh += DN.Ny[qp][dofyy] * d_ay[mode][dofyy]
                    dDyh += DN.dNy[qp][dofyy] * d_ay[mode][dofyy]
                end
                for dofzz in 1:nElDofsz
                    Dzh += DN.Nz[qp][dofzz] * d_az[mode][dofzz]
                    dDzh += DN.dNz[qp][dofzz] * d_az[mode][dofzz]
                end
                d_n = Dxh * Dyh * Dzh
                ∇d_n = hadamard(hadamard(dDxh,dDyh),dDzh)

                fe[dofz1] -= (dmp.l^2 * ∇v_z ⋅ ∇d_n + (1+2*dmp.l/dmp.Gc*Ψ[qp]) * v_z * d_n) * dV
            end

            # Stiffness matrix
            for dofz2 in 1:nElDofsz
                # Testfunction for dof 2
                v_z2 = DN.Nz[qp][dofz2]
                ∇v_z2 = DN.dNz[qp][dofz2]
                # Add on x and y sums
                v_z2 = v_z2*v_z_x
                v_z2 = v_z2*v_z_y
                ∇v_z2 = hadamard(∇v_z2,∇v_z_x)
                ∇v_z2 = hadamard(∇v_z2,∇v_z_y)

                Ke[dofz1,dofz2] += (dmp.l^2 * ∇v_z ⋅ ∇v_z2 + (1+2*dmp.l/dmp.Gc*Ψ[qp]) * v_z * v_z2) * dV
            end
        end
    end

    return fe, Ke
end

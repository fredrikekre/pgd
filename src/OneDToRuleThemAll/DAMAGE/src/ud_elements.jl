#######################################
# Shape functions for displacement 3D #
#######################################
function get_u_N_3D{dimcomp,T,functionspace}(Ux::PGDComponent{dimcomp,T,functionspace},Uy::PGDComponent{dimcomp,T,functionspace},Uz::PGDComponent{dimcomp,T,functionspace}, global_fe_values)

    ex = Ux.mesh.ex[1]
    reinit!(Ux.fev,ex)
    ey = Uy.mesh.ex[1]
    reinit!(Uy.fev,ey)
    ez = Uz.mesh.ex[1]
    reinit!(Uz.fev,ez)

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

    nqpx = length(points(Ux.fev.quad_rule))
    nqpy = length(points(Uy.fev.quad_rule))
    nqpz = length(points(Uz.fev.quad_rule))

    u_Nx = Vector{Vec{3,T,3}}[]
    u_Ny = Vector{Vec{3,T,3}}[]
    u_Nz = Vector{Vec{3,T,3}}[]
    u_dNx = Vector{Tensor{2,3,T,9}}[]
    u_dNy = Vector{Tensor{2,3,T,9}}[]
    u_dNz = Vector{Tensor{2,3,T,9}}[]

    for qpx in 1:nqpx, qpy in 1:nqpy, qpz in 1:nqpz
        u_Nx_qp = Vec{3,T,3}[]
        u_Ny_qp = Vec{3,T,3}[]
        u_Nz_qp = Vec{3,T,3}[]
        u_dNx_qp = Tensor{2,3,T,9}[]
        u_dNy_qp = Tensor{2,3,T,9}[]
        u_dNz_qp = Tensor{2,3,T,9}[]

        ##########################
        # Shape functions for Ux #
        ##########################
        for dof in 1:Ux.mesh.nElDofs
            counterN = ceil(Int,dof/3)
            counterDim = mod(dof,3); if counterDim == 0; counterDim = 3; end

            # Shape functions
            this_u_Nx = zeros(T,3)
            this_u_Nx[counterDim] = Ux.fev.N[qpx][counterN]
            u_Nx_ = Vec{3,T,3}((this_u_Nx...))
            push!(u_Nx_qp, u_Nx_)

            # Derivatives
            this_u_dNx = zeros(T,3)
            this_u_dNx[counterDim] = Ux.fev.dNdx[qpx][counterN][1]
            this_u_dNx_t = [this_u_dNx; this_u_Nx; this_u_Nx]
            u_dNx_ = Tensor{2,3,T,9}((this_u_dNx_t...))
            push!(u_dNx_qp,u_dNx_)
        end
        push!(u_Nx,u_Nx_qp)
        push!(u_dNx,u_dNx_qp)

        ##########################
        # Shape functions for Uy #
        ##########################
        for dof in 1:Uy.mesh.nElDofs
            counterN = ceil(Int,dof/3)
            counterDim = mod(dof,3); if counterDim == 0; counterDim = 3; end

            # Shape functions
            this_u_Ny = zeros(T,3)
            this_u_Ny[counterDim] = Uy.fev.N[qpy][counterN]
            u_Ny_ = Vec{3,T,3}((this_u_Ny...))
            push!(u_Ny_qp, u_Ny_)

            # Derivatives
            this_u_dNy = zeros(T,3)
            this_u_dNy[counterDim] = Uy.fev.dNdx[qpy][counterN][1]
            this_u_dNy_t = [this_u_Ny; this_u_dNy; this_u_Ny]
            u_dNy_ = Tensor{2,3,T,9}((this_u_dNy_t...))
            push!(u_dNy_qp,u_dNy_)
        end
        push!(u_Ny,u_Ny_qp)
        push!(u_dNy,u_dNy_qp)

        ##########################
        # Shape functions for Uz #
        ##########################
        for dof in 1:Uz.mesh.nElDofs
            counterN = ceil(Int,dof/3)
            counterDim = mod(dof,3); if counterDim == 0; counterDim = 3; end

            # Shape functions
            this_u_Nz = zeros(T,3)
            this_u_Nz[counterDim] = Uz.fev.N[qpz][counterN]
            u_Nz_ = Vec{3,T,3}((this_u_Nz...))
            push!(u_Nz_qp, u_Nz_)

            # Derivatives
            this_u_dNz = zeros(T,3)
            this_u_dNz[counterDim] = Uz.fev.dNdx[qpz][counterN][1]
            this_u_dNz_t = [this_u_Nz; this_u_Nz; this_u_dNz]
            u_dNz_ = Tensor{2,3,T,9}((this_u_dNz_t...))
            push!(u_dNz_qp,u_dNz_)
        end
        push!(u_Nz,u_Nz_qp)
        push!(u_dNz,u_dNz_qp)
    end

    return u_shapefunctions(u_Nx,u_Ny,u_Nz,u_dNx,u_dNy,u_dNz), global_fe_values
end

#######################################################################
# Element functions for 3D elasticity with damage with 3D integration #
#######################################################################
#x
function ud_x_ele_3D_3D{T}(fe::Vector{T},Ke::Matrix{T},
                           Ux::PGDComponent,u_ax::Vector{Vector{T}},
                           Uy::PGDComponent,u_ay::Vector{Vector{T}},
                           Uz::PGDComponent,u_az::Vector{Vector{T}},
                           UN::u_shapefunctions{3,Float64,9},global_u_fe_values::FEValues,
                           E::Tensor{4,3,Float64,81},b::Tensor{1,3,Float64,3},
                           Dx::PGDComponent,d_ax::Vector{Vector{T}},
                           Dy::PGDComponent,d_ay::Vector{Vector{T}},
                           Dz::PGDComponent,d_az::Vector{Vector{T}},
                           DN::d_shapefunctions{3,Float64})

    # Set up
    nElDofsx = Ux.mesh.nElDofs
    nElDofsy = Uy.mesh.nElDofs
    nElDofsz = Uz.mesh.nElDofs
    fill!(fe,zero(T))
    fill!(Ke,zero(T))

    # Integrate this element
    nqp = length(points(global_u_fe_values.quad_rule))
    number_of_modes = length(u_ax)-1

    # Energy
    Ψe = zeros(nqp)

    for qp in 1:nqp

        # Evaluate damage variable
        d = 0.0
        for d_mode in 1:length(d_ax)
            dx = 0.0
            dy = 0.0
            dz = 0.0
            for d_dofx in 1:Dx.mesh.nElDofs
                dx += DN.Nx[qp][d_dofx] * d_ax[d_mode][d_dofx]
            end
            for d_dofy in 1:Dy.mesh.nElDofs
                dy += DN.Ny[qp][d_dofy] * d_ay[d_mode][d_dofy]
            end
            for d_dofz in 1:Dz.mesh.nElDofs
                dz += DN.Nz[qp][d_dofz] * d_az[d_mode][d_dofz]
            end
            d += dx*dy*dz
        end
        rf = 0.01 # 1%
        degrade = max((1-d)^2, rf)

        # Integration weight
        dV = detJdV(global_u_fe_values,qp)

        # y-sum of test function
        v_x_y = zero(Vec{3,Float64})
        ∇v_x_y = zero(Tensor{2,3,Float64})
        for dofy in 1:nElDofsy
            v_x_y += UN.Ny[qp][dofy] * u_ay[end][dofy]
            ∇v_x_y += UN.dNy[qp][dofy] * u_ay[end][dofy]
        end

        # z-sum of test function
        v_x_z = zero(Vec{3,Float64})
        ∇v_x_z = zero(Tensor{2,3,Float64})
        for dofz in 1:nElDofsz
            v_x_z += UN.Nz[qp][dofz] * u_az[end][dofz]
            ∇v_x_z += UN.dNz[qp][dofz] * u_az[end][dofz]
        end

        for dofx1 in 1:nElDofsx
            # Test function for this dof
            v_x = UN.Nx[qp][dofx1]
            ∇v_x = UN.dNx[qp][dofx1]

            # Add on the y-sym
            v_x = hadamard(v_x,v_x_y)
            ∇v_x = hadamard(∇v_x,∇v_x_y)

            # Add on the z-sym
            v_x = hadamard(v_x,v_x_z)
            ∇v_x = hadamard(∇v_x,∇v_x_z)

            # Body force
            fe[dofx1] += (v_x ⋅ b) * dV

            # Previous modes
            for mode in 1:number_of_modes
                dUxh = zero(Tensor{2,3})
                dUyh = zero(Tensor{2,3})
                dUzh = zero(Tensor{2,3})
                for dofxx in 1:nElDofsx
                    dUxh += UN.dNx[qp][dofxx] * u_ax[mode][dofxx]
                end
                for dofyy in 1:nElDofsy
                    dUyh += UN.dNy[qp][dofyy] * u_ay[mode][dofyy]
                end
                for dofzz in 1:nElDofsz
                    dUzh += UN.dNz[qp][dofzz] * u_az[mode][dofzz]
                end
                ∇u_n = hadamard(hadamard(dUxh,dUyh),dUzh)
                ε_n = symmetric(∇u_n)
                σ_n = dcontract(E,ε_n)

                σ_eff = degrade * σ_n

                fe[dofx1] -= (dcontract(∇v_x, σ_eff)) * dV

                # Calculate energy from this mode
                Ψe[qp] += 0.5 * dcontract(ε_n,dcontract(E,ε_n))
            end

            # Stiffness matrix
            ε_np1 = zero(Tensor{2,3})
            for dofx2 in 1:nElDofsx
                # Testfunction for dof 2
                ∇v_x2 = UN.dNx[qp][dofx2]
                # Add on y and z sums
                ∇v_x2 = hadamard(∇v_x2,∇v_x_y)
                ∇v_x2 = hadamard(∇v_x2,∇v_x_z)
                E_eff = degrade * E

                Ke[dofx1,dofx2] += (dcontract(∇v_x, dcontract(E_eff, ∇v_x2))) * dV

                ε_np1 += ∇v_x2 * u_ax[end][dofx2]
            end
            # Add energy for last mode
            Ψe[qp] += 0.5 * dcontract(ε_np1,dcontract(E,ε_np1))

        end
    end

    return fe, Ke, Ψe
end

# y
function ud_y_ele_3D_3D{T}(fe::Vector{T},Ke::Matrix{T},
                           Ux::PGDComponent,u_ax::Vector{Vector{T}},
                           Uy::PGDComponent,u_ay::Vector{Vector{T}},
                           Uz::PGDComponent,u_az::Vector{Vector{T}},
                           UN::u_shapefunctions{3,Float64,9},global_u_fe_values::FEValues,
                           E::Tensor{4,3,Float64,81},b::Tensor{1,3,Float64,3},
                           Dx::PGDComponent,d_ax::Vector{Vector{T}},
                           Dy::PGDComponent,d_ay::Vector{Vector{T}},
                           Dz::PGDComponent,d_az::Vector{Vector{T}},
                           DN::d_shapefunctions{3,Float64})

    # Set up
    nElDofsx = Ux.mesh.nElDofs
    nElDofsy = Uy.mesh.nElDofs
    nElDofsz = Uz.mesh.nElDofs
    fill!(fe,zero(T))
    fill!(Ke,zero(T))

    # Integrate this element
    nqp = length(points(global_u_fe_values.quad_rule))
    number_of_modes = length(u_ay)-1

    # Energy
    Ψe = zeros(nqp)

    for qp in 1:nqp

        # Evaluate damage variable
        d = 0.0
        for d_mode in 1:length(d_ax)
            dx = 0.0
            dy = 0.0
            dz = 0.0
            for d_dofx in 1:Dx.mesh.nElDofs
                dx += DN.Nx[qp][d_dofx] * d_ax[d_mode][d_dofx]
            end
            for d_dofy in 1:Dy.mesh.nElDofs
                dy += DN.Ny[qp][d_dofy] * d_ay[d_mode][d_dofy]
            end
            for d_dofz in 1:Dz.mesh.nElDofs
                dz += DN.Nz[qp][d_dofz] * d_az[d_mode][d_dofz]
            end
            d += dx*dy*dz
        end
        rf = 0.01 # 1%
        degrade = max((1-d)^2, rf)

        # Integration weight
        dV = detJdV(global_u_fe_values,qp)

        # y-sum of test function
        v_y_x = zero(Vec{3,Float64})
        ∇v_y_x = zero(Tensor{2,3,Float64})
        for dofx in 1:nElDofsx
            v_y_x += UN.Nx[qp][dofx] * u_ax[end][dofx]
            ∇v_y_x += UN.dNx[qp][dofx] * u_ax[end][dofx]
        end

        # z-sum of test function
        v_y_z = zero(Vec{3,Float64})
        ∇v_y_z = zero(Tensor{2,3,Float64})
        for dofz in 1:nElDofsz
            v_y_z += UN.Nz[qp][dofz] * u_az[end][dofz]
            ∇v_y_z += UN.dNz[qp][dofz] * u_az[end][dofz]
        end

        for dofy1 in 1:nElDofsy
            # Test function for this dof
            v_y = UN.Ny[qp][dofy1]
            ∇v_y = UN.dNy[qp][dofy1]

            # Add on the x-sym
            v_y = hadamard(v_y,v_y_x)
            ∇v_y = hadamard(∇v_y,∇v_y_x)

            # Add on the z-sym
            v_y = hadamard(v_y,v_y_z)
            ∇v_y = hadamard(∇v_y,∇v_y_z)

            # Body force
            fe[dofy1] += (v_y ⋅ b) * dV

            # Previous modes
            for mode in 1:number_of_modes
                dUxh = zero(Tensor{2,3})
                dUyh = zero(Tensor{2,3})
                dUzh = zero(Tensor{2,3})
                for dofxx in 1:nElDofsx
                    dUxh += UN.dNx[qp][dofxx] * u_ax[mode][dofxx]
                end
                for dofyy in 1:nElDofsy
                    dUyh += UN.dNy[qp][dofyy] * u_ay[mode][dofyy]
                end
                for dofzz in 1:nElDofsz
                    dUzh += UN.dNz[qp][dofzz] * u_az[mode][dofzz]
                end
                ∇u_n = hadamard(hadamard(dUxh,dUyh),dUzh)
                ε_n = symmetric(∇u_n)
                σ_n = dcontract(E,ε_n)

                σ_eff = degrade * σ_n

                fe[dofy1] -= (dcontract(∇v_y, σ_eff)) * dV

                # Calculate energy from this mode
                Ψe[qp] += 0.5 * dcontract(ε_n,dcontract(E,ε_n))
            end

            # Stiffness matrix
            ε_np1 = zero(Tensor{2,3})
            for dofy2 in 1:nElDofsy
                # Testfunction for dof 2
                ∇v_y2 = UN.dNy[qp][dofy2]
                # Add on x and z sums
                ∇v_y2 = hadamard(∇v_y2,∇v_y_x)
                ∇v_y2 = hadamard(∇v_y2,∇v_y_z)
                E_eff = degrade * E

                Ke[dofy1,dofy2] += (dcontract(∇v_y, dcontract(E_eff, ∇v_y2))) * dV

                ε_np1 += ∇v_y2 * u_ay[end][dofy2]
            end
            # Add energy for last mode
            Ψe[qp] += 0.5 * dcontract(ε_np1,dcontract(E,ε_np1))

        end
    end

    return fe, Ke, Ψe
end

# z
function ud_z_ele_3D_3D{T}(fe::Vector{T},Ke::Matrix{T},
                           Ux::PGDComponent,u_ax::Vector{Vector{T}},
                           Uy::PGDComponent,u_ay::Vector{Vector{T}},
                           Uz::PGDComponent,u_az::Vector{Vector{T}},
                           UN::u_shapefunctions{3,Float64,9},global_u_fe_values::FEValues,
                           E::Tensor{4,3,Float64,81},b::Tensor{1,3,Float64,3},
                           Dx::PGDComponent,d_ax::Vector{Vector{T}},
                           Dy::PGDComponent,d_ay::Vector{Vector{T}},
                           Dz::PGDComponent,d_az::Vector{Vector{T}},
                           DN::d_shapefunctions{3,Float64})

    # Set up
    nElDofsx = Ux.mesh.nElDofs
    nElDofsy = Uy.mesh.nElDofs
    nElDofsz = Uz.mesh.nElDofs
    fill!(fe,zero(T))
    fill!(Ke,zero(T))

    # Integrate this element
    nqp = length(points(global_u_fe_values.quad_rule))
    number_of_modes = length(u_az)-1

    # Energy
    Ψe = zeros(nqp)

    for qp in 1:nqp

        # Evaluate damage variable
        d = 0.0
        for d_mode in 1:length(d_ax)
            dx = 0.0
            dy = 0.0
            dz = 0.0
            for d_dofx in 1:Dx.mesh.nElDofs
                dx += DN.Nx[qp][d_dofx] * d_ax[d_mode][d_dofx]
            end
            for d_dofy in 1:Dy.mesh.nElDofs
                dy += DN.Ny[qp][d_dofy] * d_ay[d_mode][d_dofy]
            end
            for d_dofz in 1:Dz.mesh.nElDofs
                dz += DN.Nz[qp][d_dofz] * d_az[d_mode][d_dofz]
            end
            d += dx*dy*dz
        end
        rf = 0.01 # 1%
        degrade = max((1-d)^2, rf)

        # Integration weight
        dV = detJdV(global_u_fe_values,qp)

        # x-sum of test function
        v_z_x = zero(Vec{3,Float64})
        ∇v_z_x = zero(Tensor{2,3,Float64})
        for dofx in 1:nElDofsx
            v_z_x += UN.Nx[qp][dofx] * u_ax[end][dofx]
            ∇v_z_x += UN.dNx[qp][dofx] * u_ax[end][dofx]
        end

        # y-sum of test function
        v_z_y = zero(Vec{3,Float64})
        ∇v_z_y = zero(Tensor{2,3,Float64})
        for dofy in 1:nElDofsy
            v_z_y += UN.Ny[qp][dofy] * u_ay[end][dofy]
            ∇v_z_y += UN.dNy[qp][dofy] * u_ay[end][dofy]
        end

        for dofz1 in 1:nElDofsz
            # Test function for this dof
            v_z = UN.Nz[qp][dofz1]
            ∇v_z = UN.dNz[qp][dofz1]

            # Add on the x-sym
            v_z = hadamard(v_z,v_z_x)
            ∇v_z = hadamard(∇v_z,∇v_z_x)

            # Add on the y-sym
            v_z = hadamard(v_z,v_z_y)
            ∇v_z = hadamard(∇v_z,∇v_z_y)

            # Body force
            fe[dofz1] += (v_z ⋅ b) * dV

            # Previous modes
            for mode in 1:number_of_modes
                dUxh = zero(Tensor{2,3})
                dUyh = zero(Tensor{2,3})
                dUzh = zero(Tensor{2,3})
                for dofxx in 1:nElDofsx
                    dUxh += UN.dNx[qp][dofxx] * u_ax[mode][dofxx]
                end
                for dofyy in 1:nElDofsy
                    dUyh += UN.dNy[qp][dofyy] * u_ay[mode][dofyy]
                end
                for dofzz in 1:nElDofsz
                    dUzh += UN.dNz[qp][dofzz] * u_az[mode][dofzz]
                end
                ∇u_n = hadamard(hadamard(dUxh,dUyh),dUzh)
                ε_n = symmetric(∇u_n)
                σ_n = dcontract(E,ε_n)

                σ_eff = degrade * σ_n

                fe[dofz1] -= (dcontract(∇v_z, σ_eff)) * dV

                # Calculate energy from this mode
                Ψe[qp] += 0.5 * dcontract(ε_n,dcontract(E,ε_n))
            end

            # Stiffness matrix
            ε_np1 = zero(Tensor{2,3})
            for dofz2 in 1:nElDofsz
                # Testfunction for dof 2
                ∇v_z2 = UN.dNz[qp][dofz2]
                # Add on x and y sums
                ∇v_z2 = hadamard(∇v_z2,∇v_z_x)
                ∇v_z2 = hadamard(∇v_z2,∇v_z_y)
                E_eff = degrade * E

                Ke[dofz1,dofz2] += (dcontract(∇v_z, dcontract(E_eff, ∇v_z2))) * dV

                ε_np1 += ∇v_z2 * u_az[end][dofz2]
            end
            # Add energy for last mode
            Ψe[qp] += 0.5 * dcontract(ε_np1,dcontract(E,ε_np1))

        end
    end

    return fe, Ke, Ψe
end

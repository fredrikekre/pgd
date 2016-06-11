#######################################
# Shape functions for displacement 2D #
#######################################
function get_u_N_2D{dimcomp,T,functionspace}(Ux::PGDComponent{dimcomp,T,functionspace},Uy::PGDComponent{dimcomp,T,functionspace}, global_fe_values)

    ex = Ux.mesh.ex[1]
    reinit!(Ux.fev,ex)
    ey = Uy.mesh.ex[1]
    reinit!(Uy.fev,ey)

    node1 = Tensor{1,2}((ex[1][1], ey[1][1]))
    node2 = Tensor{1,2}((ex[2][1], ey[1][1]))
    node3 = Tensor{1,2}((ex[2][1], ey[2][1]))
    node4 = Tensor{1,2}((ex[1][1], ey[2][1]))
    EX = Tensor{1,2,Float64,2}[node1, node2, node3, node4]
    reinit!(global_fe_values, EX)

    nqpx = length(points(Ux.fev.quad_rule))
    nqpy = length(points(Uy.fev.quad_rule))

    u_Nx = Vector{Vec{2,T,2}}[]
    u_Ny = Vector{Vec{2,T,2}}[]
    u_dNx = Vector{Tensor{2,2,T,4}}[]
    u_dNy = Vector{Tensor{2,2,T,4}}[]

    for qpx in 1:nqpx, qpy in 1:nqpy
        u_Nx_qp = Vec{2,T,2}[]
        u_Ny_qp = Vec{2,T,2}[]
        u_dNx_qp = Tensor{2,2,T,4}[]
        u_dNy_qp = Tensor{2,2,T,4}[]

        ##########################
        # Shape functions for Ux #
        ##########################
        for dof in 1:Ux.mesh.nElDofs
            counterN = ceil(Int,dof/2)
            counterDim = mod(dof,2); if counterDim == 0; counterDim = 2; end

            # Shape functions
            this_u_Nx = zeros(T,2)
            this_u_Nx[counterDim] = Ux.fev.N[qpx][counterN]
            u_Nx_ = Vec{2,T,2}((this_u_Nx...))
            push!(u_Nx_qp, u_Nx_)

            # Derivatives
            this_u_dNx = zeros(T,2)
            this_u_dNx[counterDim] = Ux.fev.dNdx[qpx][counterN][1]
            this_u_dNx_t = [this_u_dNx; this_u_Nx]
            u_dNx_ = Tensor{2,2,T,4}((this_u_dNx_t...))
            push!(u_dNx_qp,u_dNx_)
        end
        push!(u_Nx,u_Nx_qp)
        push!(u_dNx,u_dNx_qp)

        ##########################
        # Shape functions for Uy #
        ##########################
        for dof in 1:Uy.mesh.nElDofs
            counterN = ceil(Int,dof/2)
            counterDim = mod(dof,2); if counterDim == 0; counterDim = 2; end

            # Shape functions
            this_u_Ny = zeros(T,2)
            this_u_Ny[counterDim] = Uy.fev.N[qpy][counterN]
            u_Ny_ = Vec{2,T,2}((this_u_Ny...))
            push!(u_Ny_qp, u_Ny_)

            # Derivatives
            this_u_dNy = zeros(T,2)
            this_u_dNy[counterDim] = Uy.fev.dNdx[qpy][counterN][1]
            this_u_dNy_t = [this_u_Ny; this_u_dNy]
            u_dNy_ = Tensor{2,2,T,4}((this_u_dNy_t...))
            push!(u_dNy_qp,u_dNy_)
        end
        push!(u_Ny,u_Ny_qp)
        push!(u_dNy,u_dNy_qp)

    end

    return u_shapefunctions(u_Nx,u_Ny,u_dNx,u_dNy), global_fe_values
end

#######################################################################
# Element functions for 2D elasticity with damage with 2D integration #
#######################################################################
#x
function ud_x_ele_2D_2D{T}(fe::Vector{T},Ke::Matrix{T},
                           Ux::PGDComponent,u_ax::Vector{Vector{T}},
                           Uy::PGDComponent,u_ay::Vector{Vector{T}},
                           UN::u_shapefunctions{2,Float64,4},global_u_fe_values::FEValues,
                           E::Tensor{4,2,Float64,16},b::Tensor{1,2,Float64,2},
                           Dx::PGDComponent,d_ax::Vector{Vector{T}},
                           Dy::PGDComponent,d_ay::Vector{Vector{T}},
                           DN::d_shapefunctions{2,Float64})

    # Set up
    nElDofsx = Ux.mesh.nElDofs
    nElDofsy = Uy.mesh.nElDofs
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
            for d_dofx in 1:Dx.mesh.nElDofs
                dx += DN.Nx[qp][d_dofx] * d_ax[d_mode][d_dofx]
            end
            for d_dofy in 1:Dy.mesh.nElDofs
                dy += DN.Ny[qp][d_dofy] * d_ay[d_mode][d_dofy]
            end
            d += dx*dy
        end
        rf = 0.01 # 1%
        degrade = max((1-d)^2, rf)

        # Integration weight
        dV = detJdV(global_u_fe_values,qp)

        # y-sum of test function
        v_x_y = zero(Vec{2,Float64})
        ∇v_x_y = zero(Tensor{2,2,Float64})
        for dofy in 1:nElDofsy
            v_x_y += UN.Ny[qp][dofy] * u_ay[end][dofy]
            ∇v_x_y += UN.dNy[qp][dofy] * u_ay[end][dofy]
        end

        for dofx1 in 1:nElDofsx
            # Test function for this dof
            v_x = UN.Nx[qp][dofx1]
            ∇v_x = UN.dNx[qp][dofx1]

            # Add on the y-sym
            v_x = hadamard(v_x,v_x_y)
            ∇v_x = hadamard(∇v_x,∇v_x_y)


            # Body force
            fe[dofx1] += (v_x ⋅ b) * dV

            # Previous modes
            for mode in 1:number_of_modes
                dUxh = zero(Tensor{2,2})
                dUyh = zero(Tensor{2,2})
                for dofxx in 1:nElDofsx
                    dUxh += UN.dNx[qp][dofxx] * u_ax[mode][dofxx]
                end
                for dofyy in 1:nElDofsy
                    dUyh += UN.dNy[qp][dofyy] * u_ay[mode][dofyy]
                end
                ∇u_n = hadamard(dUxh,dUyh)
                ε_n = symmetric(∇u_n)
                σ_n = dcontract(E,ε_n)

                σ_eff = degrade * σ_n

                fe[dofx1] -= (dcontract(∇v_x, σ_eff)) * dV

                # Calculate energy from this mode
                Ψe[qp] += 0.5 * dcontract(ε_n,dcontract(E,ε_n))
            end

            # Stiffness matrix
            ε_np1 = zero(Tensor{2,2})
            for dofx2 in 1:nElDofsx
                # Testfunction for dof 2
                ∇v_x2 = UN.dNx[qp][dofx2]
                # Add on y sum
                ∇v_x2 = hadamard(∇v_x2,∇v_x_y)
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
function ud_y_ele_2D_2D{T}(fe::Vector{T},Ke::Matrix{T},
                           Ux::PGDComponent,u_ax::Vector{Vector{T}},
                           Uy::PGDComponent,u_ay::Vector{Vector{T}},
                           UN::u_shapefunctions{2,Float64,4},global_u_fe_values::FEValues,
                           E::Tensor{4,2,Float64,16},b::Tensor{1,2,Float64,2},
                           Dx::PGDComponent,d_ax::Vector{Vector{T}},
                           Dy::PGDComponent,d_ay::Vector{Vector{T}},
                           DN::d_shapefunctions{2,Float64})

    # Set up
    nElDofsx = Ux.mesh.nElDofs
    nElDofsy = Uy.mesh.nElDofs
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
            for d_dofx in 1:Dx.mesh.nElDofs
                dx += DN.Nx[qp][d_dofx] * d_ax[d_mode][d_dofx]
            end
            for d_dofy in 1:Dy.mesh.nElDofs
                dy += DN.Ny[qp][d_dofy] * d_ay[d_mode][d_dofy]
            end
            d += dx*dy
        end
        rf = 0.01 # 1%
        degrade = max((1-d)^2, rf)

        # Integration weight
        dV = detJdV(global_u_fe_values,qp)

        # y-sum of test function
        v_y_x = zero(Vec{2,Float64})
        ∇v_y_x = zero(Tensor{2,2,Float64})
        for dofx in 1:nElDofsx
            v_y_x += UN.Nx[qp][dofx] * u_ax[end][dofx]
            ∇v_y_x += UN.dNx[qp][dofx] * u_ax[end][dofx]
        end


        for dofy1 in 1:nElDofsy
            # Test function for this dof
            v_y = UN.Ny[qp][dofy1]
            ∇v_y = UN.dNy[qp][dofy1]

            # Add on the x-sym
            v_y = hadamard(v_y,v_y_x)
            ∇v_y = hadamard(∇v_y,∇v_y_x)


            # Body force
            fe[dofy1] += (v_y ⋅ b) * dV

            # Previous modes
            for mode in 1:number_of_modes
                dUxh = zero(Tensor{2,2})
                dUyh = zero(Tensor{2,2})
                for dofxx in 1:nElDofsx
                    dUxh += UN.dNx[qp][dofxx] * u_ax[mode][dofxx]
                end
                for dofyy in 1:nElDofsy
                    dUyh += UN.dNy[qp][dofyy] * u_ay[mode][dofyy]
                end
                ∇u_n = hadamard(dUxh,dUyh)
                ε_n = symmetric(∇u_n)
                σ_n = dcontract(E,ε_n)

                σ_eff = degrade * σ_n

                fe[dofy1] -= (dcontract(∇v_y, σ_eff)) * dV

                # Calculate energy from this mode
                Ψe[qp] += 0.5 * dcontract(ε_n,dcontract(E,ε_n))
            end

            # Stiffness matrix
            ε_np1 = zero(Tensor{2,2})
            for dofy2 in 1:nElDofsy
                # Testfunction for dof 2
                ∇v_y2 = UN.dNy[qp][dofy2]
                # Add on x sum
                ∇v_y2 = hadamard(∇v_y2,∇v_y_x)
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

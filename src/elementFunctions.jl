include("buffers.jl")

###############################
# Displacement internal force #
###############################

function U_intf{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,
                                   x,U_mp::LinearElastic_withTangent,b::Vector)
    # U_an is the unknowns
    # U_a are the already computed modes
    # 4 node quadrilateral element
    U_anx = U_an[1:4] # Not general
    U_any = U_an[5:8]
    U_ax = U_a[1:4,:]
    U_ay = U_a[5:8,:]

    buff_coll = get_buffer(U_buff_colls, T)

    g = buff_coll.g # Not general

    Nx = buff_coll.Nx # Shape functions
    Ny = buff_coll.Ny

    dNdx = buff_coll.dNdx # Derivatives
    dNdy = buff_coll.dNdy

    NNx = buff_coll.NNx# N matrix for force vector
    NNy = buff_coll.NNy

    BBx = buff_coll.BBx  # B matrix for stiffness
    BBy = buff_coll.BBy

    for (q_point, (ξ,w)) in enumerate(zip(U.fev.quad_rule.points,U.fev.quad_rule.weights))

        ξ_x = Tensor{1,1}((ξ[1],))
        ξ_y = Tensor{1,1}((ξ[2],))

        # Update values
        ex_x = [U.components[1].mesh.x[1] U.components[1].mesh.x[2]] # only for equidistant mesh
        ex_x = reinterpret(Vec{1,Float64},ex_x,(size(ex_x,2),))
        ex_y = [U.components[2].mesh.x[1] U.components[2].mesh.x[2]]
        ex_y = reinterpret(Vec{1,Float64},ex_y,(size(ex_y,2),))

        dNdx = reinterpret(Vec{1,Float64},dNdx,(2,))
        dNdy = reinterpret(Vec{1,Float64},dNdy,(2,))

        evaluate_at_gauss_point!(U.components[1].fev,ξ_x,ex_x,Nx,dNdx)
        evaluate_at_gauss_point!(U.components[2].fev,ξ_y,ex_y,Ny,dNdy)

        dNdx = reinterpret(Float64,dNdx,(size(dNdx[1],1),length(dNdx)))
        dNdy = reinterpret(Float64,dNdy,(size(dNdy[1],1),length(dNdy)))

        # TODO: Simplify these expressions, potentially using ContMechTensors.jl somehow and include the Hadamard product.
        NNx[1,1] = Nx[1] * Ny[1] * U_any[1] + Nx[1] * Ny[2] * U_any[3]
        NNx[2,2] = Nx[1] * Ny[1] * U_any[2] + Nx[1] * Ny[2] * U_any[4]
        NNx[1,3] = Nx[2] * Ny[1] * U_any[1] + Nx[2] * Ny[2] * U_any[3]
        NNx[2,4] = Nx[2] * Ny[1] * U_any[2] + Nx[2] * Ny[2] * U_any[4]

        NNy[1,1] = Nx[1] * Ny[1] * U_anx[1] + Nx[2] * Ny[1] * U_anx[3]
        NNy[2,2] = Nx[1] * Ny[1] * U_anx[2] + Nx[2] * Ny[1] * U_anx[4]
        NNy[1,3] = Nx[1] * Ny[2] * U_anx[1] + Nx[2] * Ny[2] * U_anx[3]
        NNy[2,4] = Nx[1] * Ny[2] * U_anx[2] + Nx[2] * Ny[2] * U_anx[4]

        BBx[1,1] = dNdx[1] * Ny[1] * U_any[1] + dNdx[1] * Ny[2] * U_any[3]
        BBx[3,1] = Nx[1] * dNdy[1] * U_any[1] + Nx[1] * dNdy[2] * U_any[3]
        BBx[2,2] = Nx[1] * dNdy[1] * U_any[2] + Nx[1] * dNdy[2] * U_any[4]
        BBx[3,2] = dNdx[1] * Ny[1] * U_any[2] + dNdx[1] * Ny[2] * U_any[4]
        BBx[1,3] = dNdx[2] * Ny[1] * U_any[1] + dNdx[2] * Ny[2] * U_any[3]
        BBx[3,3] = Nx[2] * dNdy[1] * U_any[1] + Nx[2] * dNdy[2] * U_any[3]
        BBx[2,4] = Nx[2] * dNdy[1] * U_any[2] + Nx[2] * dNdy[2] * U_any[4]
        BBx[3,4] = dNdx[2] * Ny[1] * U_any[2] + dNdx[2] * Ny[2] * U_any[4]

        BBy[1,1] = dNdx[1] * Ny[1] * U_anx[1] + dNdx[2] * Ny[1] * U_anx[3]
        BBy[3,1] = Nx[1] * dNdy[1] * U_anx[1] + Nx[2] * dNdy[1] * U_anx[3]
        BBy[2,2] = Nx[1] * dNdy[1] * U_anx[2] + Nx[2] * dNdy[1] * U_anx[4]
        BBy[3,2] = dNdx[1] * Ny[1] * U_anx[2] + dNdx[2] * Ny[1] * U_anx[4]
        BBy[1,3] = dNdx[1] * Ny[2] * U_anx[1] + dNdx[2] * Ny[2] * U_anx[3]
        BBy[3,3] = Nx[1] * dNdy[2] * U_anx[1] + Nx[2] * dNdy[2] * U_anx[3]
        BBy[2,4] = Nx[1] * dNdy[2] * U_anx[2] + Nx[2] * dNdy[2] * U_anx[4]
        BBy[3,4] = dNdx[1] * Ny[2] * U_anx[2] + dNdx[2] * Ny[2] * U_anx[4]

        ε = buff_coll.ε
        fill!(ε, 0.0)
        ε_m = buff_coll.ε_m

        for m = 1:nModes(U)
            ε_m[1] = dNdx[1] * Ny[1] * U_ax[1,m] * U_ay[1,m] + dNdx[2] * Ny[1] * U_ax[3,m] * U_ay[1,m] +
                     dNdx[1] * Ny[2] * U_ax[1,m] * U_ay[3,m] + dNdx[2] * Ny[2] * U_ax[3,m] * U_ay[3,m]

            ε_m[2] = Nx[1] * dNdy[1] * U_ax[2,m] * U_ay[2,m] + Nx[1] * dNdy[2] * U_ax[2,m] * U_ay[4,m] +
                     Nx[2] * dNdy[1] * U_ax[4,m] * U_ay[2,m] + Nx[2] * dNdy[2] * U_ax[4,m] * U_ay[4,m]

            ε_m[3] = Nx[1] * dNdy[1] * U_ax[1,m] * U_ay[1,m] + Nx[1] * dNdy[2] * U_ax[1,m] * U_ay[3,m] +
                     Nx[2] * dNdy[1] * U_ax[3,m] * U_ay[1,m] + Nx[2] * dNdy[2] * U_ax[3,m] * U_ay[3,m] +
                     dNdx[1] * Ny[1] * U_ax[2,m] * U_ay[2,m] + dNdx[2] * Ny[1] * U_ax[4,m] * U_ay[2,m] +
                     dNdx[1] * Ny[2] * U_ax[2,m] * U_ay[4,m] + dNdx[2] * Ny[2] * U_ax[4,m] * U_ay[4,m]

            ε += ε_m
        end

        ε += BBx*U_anx # eller BBy*U_ay, blir samma
        σ = U_mp.tangent*ε


        dΩ = U.fev.detJdV[q_point]

        gx = (BBx' * σ - NNx' * b) * dΩ
        gy = (BBy' * σ - NNy' * b) * dΩ

        g += [gx;
              gy] # Här får man typ assemblera då istället om man har en unstructured mesh
    end

    return g
end


#####################################################
# Displacement internal force as function of damage #
#####################################################

function UD_intf{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,
                                    D_a::Matrix,D::PGDFunction,
                                    x,U_mp::LinearElastic_withTangent,b::Vector)

    Ψe = zeros(4)

    # Displacement
    U_anx = U_an[1:4] # Not general
    U_any = U_an[5:8]
    U_ax = U_a[1:4,:]
    U_ay = U_a[5:8,:]

    # Damage
    D_ax = D_a[1:2,:]
    D_ay = D_a[3:4,:]

    buff_coll = get_buffer(UD_buff_colls, T)

    g = buff_coll.g # Not general

    Nx = buff_coll.Nx # Shape functions
    Ny = buff_coll.Ny

    dNdx = buff_coll.dNdx # Derivatives
    dNdy = buff_coll.dNdy

    NNx = buff_coll.NNx# N matrix for force vector
    NNy = buff_coll.NNy

    BBx = buff_coll.BBx  # B matrix for stiffness
    BBy = buff_coll.BBy

    for (q_point, (ξ,w)) in enumerate(zip(U.fev.quad_rule.points,U.fev.quad_rule.weights))

        ξ_x = Tensor{1,1}((ξ[1],))
        ξ_y = Tensor{1,1}((ξ[2],))

        # Update values
        ex_x = [U.components[1].mesh.x[1] U.components[1].mesh.x[2]] # only for equidistant mesh
        ex_x = reinterpret(Vec{1,Float64},ex_x,(size(ex_x,2),))
        ex_y = [U.components[2].mesh.x[1] U.components[2].mesh.x[2]]
        ex_y = reinterpret(Vec{1,Float64},ex_y,(size(ex_y,2),))

        dNdx = reinterpret(Vec{1,Float64},dNdx,(2,))
        dNdy = reinterpret(Vec{1,Float64},dNdy,(2,))

        evaluate_at_gauss_point!(U.components[1].fev,ξ_x,ex_x,Nx,dNdx)
        evaluate_at_gauss_point!(U.components[2].fev,ξ_y,ex_y,Ny,dNdy)

        dNdx = reinterpret(Float64,dNdx,(size(dNdx[1],1),length(dNdx)))
        dNdy = reinterpret(Float64,dNdy,(size(dNdy[1],1),length(dNdy)))

        # TODO: Simplify these expressions, potentially using ContMechTensors.jl somehow and include the Hadamard product.
        NNx[1,1] = Nx[1] * Ny[1] * U_any[1] + Nx[1] * Ny[2] * U_any[3]
        NNx[2,2] = Nx[1] * Ny[1] * U_any[2] + Nx[1] * Ny[2] * U_any[4]
        NNx[1,3] = Nx[2] * Ny[1] * U_any[1] + Nx[2] * Ny[2] * U_any[3]
        NNx[2,4] = Nx[2] * Ny[1] * U_any[2] + Nx[2] * Ny[2] * U_any[4]

        NNy[1,1] = Nx[1] * Ny[1] * U_anx[1] + Nx[2] * Ny[1] * U_anx[3]
        NNy[2,2] = Nx[1] * Ny[1] * U_anx[2] + Nx[2] * Ny[1] * U_anx[4]
        NNy[1,3] = Nx[1] * Ny[2] * U_anx[1] + Nx[2] * Ny[2] * U_anx[3]
        NNy[2,4] = Nx[1] * Ny[2] * U_anx[2] + Nx[2] * Ny[2] * U_anx[4]

        BBx[1,1] = dNdx[1] * Ny[1] * U_any[1] + dNdx[1] * Ny[2] * U_any[3]
        BBx[3,1] = Nx[1] * dNdy[1] * U_any[1] + Nx[1] * dNdy[2] * U_any[3]
        BBx[2,2] = Nx[1] * dNdy[1] * U_any[2] + Nx[1] * dNdy[2] * U_any[4]
        BBx[3,2] = dNdx[1] * Ny[1] * U_any[2] + dNdx[1] * Ny[2] * U_any[4]
        BBx[1,3] = dNdx[2] * Ny[1] * U_any[1] + dNdx[2] * Ny[2] * U_any[3]
        BBx[3,3] = Nx[2] * dNdy[1] * U_any[1] + Nx[2] * dNdy[2] * U_any[3]
        BBx[2,4] = Nx[2] * dNdy[1] * U_any[2] + Nx[2] * dNdy[2] * U_any[4]
        BBx[3,4] = dNdx[2] * Ny[1] * U_any[2] + dNdx[2] * Ny[2] * U_any[4]

        BBy[1,1] = dNdx[1] * Ny[1] * U_anx[1] + dNdx[2] * Ny[1] * U_anx[3]
        BBy[3,1] = Nx[1] * dNdy[1] * U_anx[1] + Nx[2] * dNdy[1] * U_anx[3]
        BBy[2,2] = Nx[1] * dNdy[1] * U_anx[2] + Nx[2] * dNdy[1] * U_anx[4]
        BBy[3,2] = dNdx[1] * Ny[1] * U_anx[2] + dNdx[2] * Ny[1] * U_anx[4]
        BBy[1,3] = dNdx[1] * Ny[2] * U_anx[1] + dNdx[2] * Ny[2] * U_anx[3]
        BBy[3,3] = Nx[1] * dNdy[2] * U_anx[1] + Nx[2] * dNdy[2] * U_anx[3]
        BBy[2,4] = Nx[1] * dNdy[2] * U_anx[2] + Nx[2] * dNdy[2] * U_anx[4]
        BBy[3,4] = dNdx[1] * Ny[2] * U_anx[2] + dNdx[2] * Ny[2] * U_anx[4]

        ε = buff_coll.ε
        fill!(ε, 0.0)
        ε_m = buff_coll.ε_m

        for m = 1:nModes(U)
            ε_m[1] = dNdx[1] * Ny[1] * U_ax[1,m] * U_ay[1,m] + dNdx[2] * Ny[1] * U_ax[3,m] * U_ay[1,m] +
                     dNdx[1] * Ny[2] * U_ax[1,m] * U_ay[3,m] + dNdx[2] * Ny[2] * U_ax[3,m] * U_ay[3,m]

            ε_m[2] = Nx[1] * dNdy[1] * U_ax[2,m] * U_ay[2,m] + Nx[1] * dNdy[2] * U_ax[2,m] * U_ay[4,m] +
                     Nx[2] * dNdy[1] * U_ax[4,m] * U_ay[2,m] + Nx[2] * dNdy[2] * U_ax[4,m] * U_ay[4,m]

            ε_m[3] = Nx[1] * dNdy[1] * U_ax[1,m] * U_ay[1,m] + Nx[1] * dNdy[2] * U_ax[1,m] * U_ay[3,m] +
                     Nx[2] * dNdy[1] * U_ax[3,m] * U_ay[1,m] + Nx[2] * dNdy[2] * U_ax[3,m] * U_ay[3,m] +
                     dNdx[1] * Ny[1] * U_ax[2,m] * U_ay[2,m] + dNdx[2] * Ny[1] * U_ax[4,m] * U_ay[2,m] +
                     dNdx[1] * Ny[2] * U_ax[2,m] * U_ay[4,m] + dNdx[2] * Ny[2] * U_ax[4,m] * U_ay[4,m]

            ε += ε_m
        end
        ε += BBx*U_anx # eller BBy*ay, blir samma

        ##########
        # Damage #
        ##########
        d = 0.0
        for m = 1:nModes(D)
            d += dot(Nx,D_ax[:,m]) * dot(Ny,D_ay[:,m])
        end

        rf = 1e-6
        σ_degradation = (1.0-d)^2 + rf

        # Stress
        σ = U_mp.tangent*ε * σ_degradation


        dΩ = U.fev.detJdV[q_point]

        gx = (BBx' * σ - NNx' * b) * dΩ
        gy = (BBy' * σ - NNy' * b) * dΩ

        g += [gx;
              gy] # Här får man typ assemblera då istället om man har en unstructured mesh

        #############################
        # Calculate the energy here #
        #############################
        if T == Float64 # Otherwise its GradientNumber...
            ε = εv_to_εt(ε)
            σ =  2 * U_mp.mp.G * dev(ε) + U_mp.mp.K * trace(ε)* one(ε)
            Ψ = 1/2 * σ ⊡ ε
            Ψe[q_point] = Ψ
        end
    end

    return g, Ψe
end


#####################################################
# Damage internal force as function of displacement #
#####################################################

function DU_intf{T}(D_an::Vector{T},D_a::Matrix,D::PGDFunction,
                                    x,D_mp::PhaseFieldDamage,Ψ::Vector{Float64})

    # Damage
    D_anx = D_an[1:2]
    D_any = D_an[3:4]
    D_ax = D_a[1:2,:]
    D_ay = D_a[3:4,:]


    buff_coll = get_buffer(DU_buff_colls, T)

    g = buff_coll.g # Not general

    Nx = buff_coll.Nx # Shape functions
    Ny = buff_coll.Ny

    dNdx = buff_coll.dNdx # Derivatives
    dNdy = buff_coll.dNdy

    NNx = buff_coll.NNx# N matrix for force vector
    NNy = buff_coll.NNy

    BBx = buff_coll.BBx  # B matrix for stiffness
    BBy = buff_coll.BBy

    for (q_point, (ξ,w)) in enumerate(zip(D.fev.quad_rule.points,D.fev.quad_rule.weights))

        ξ_x = Tensor{1,1}((ξ[1],))
        ξ_y = Tensor{1,1}((ξ[2],))

        # Update values
        ex_x = [D.components[1].mesh.x[1] D.components[1].mesh.x[2]] # only for equidistant mesh
        ex_x = reinterpret(Vec{1,Float64},ex_x,(size(ex_x,2),))
        ex_y = [D.components[2].mesh.x[1] D.components[2].mesh.x[2]]
        ex_y = reinterpret(Vec{1,Float64},ex_y,(size(ex_y,2),))

        dNdx = reinterpret(Vec{1,Float64},dNdx,(2,))
        dNdy = reinterpret(Vec{1,Float64},dNdy,(2,))

        evaluate_at_gauss_point!(D.components[1].fev,ξ_x,ex_x,Nx,dNdx)
        evaluate_at_gauss_point!(D.components[2].fev,ξ_y,ex_y,Ny,dNdy)

        dNdx = reinterpret(Float64,dNdx,(size(dNdx[1],1),length(dNdx)))
        dNdy = reinterpret(Float64,dNdy,(size(dNdy[1],1),length(dNdy)))

        ##########
        # Damage #
        ##########
        NNx[1,1] = Nx[1] * Ny[1] * D_any[1] + Nx[1] * Ny[2] * D_any[2]
        NNx[1,2] = Nx[2] * Ny[1] * D_any[1] + Nx[2] * Ny[2] * D_any[2]

        NNy[1,1] = Nx[1] * Ny[1] * D_anx[1] + Nx[2] * Ny[1] * D_anx[2]
        NNy[1,2] = Nx[1] * Ny[2] * D_anx[1] + Nx[2] * Ny[2] * D_anx[2]

        BBx[1,1] = dNdx[1] * Ny[1] * D_any[1] + dNdx[1] * Ny[2] * D_any[2]
        BBx[2,1] = Nx[1] * dNdy[1] * D_any[1] + Nx[1] * dNdy[2] * D_any[2]
        BBx[1,2] = dNdx[2] * Ny[1] * D_any[1] + dNdx[2] * Ny[2] * D_any[2]
        BBx[2,2] = Nx[2] * dNdy[1] * D_any[1] + Nx[2] * dNdy[2] * D_any[2]

        BBy[1,1] = dNdx[1] * Ny[1] * D_anx[1] + dNdx[2] * Ny[1] * D_anx[2]
        BBy[2,1] = Nx[1] * dNdy[1] * D_anx[1] + Nx[2] * dNdy[1] * D_anx[2]
        BBy[1,2] = dNdx[1] * Ny[2] * D_anx[1] + dNdx[2] * Ny[2] * D_anx[2]
        BBy[2,2] = Nx[1] * dNdy[2] * D_anx[1] + Nx[2] * dNdy[2] * D_anx[2]


        # Scalar value of damage, d and gradient of damage, ∂d
        d = T(0.0)
        ∂d = zeros(T,2)
        for m = 1:nModes(D)
            d += dot(Nx,D_ax[:,m]) * dot(Ny,D_ay[:,m])
            ∂d[1] += dNdx[1] * Ny[1] * D_ax[1,m] * D_ay[1,m] + dNdx[2] * Ny[1] * D_ax[2,m] * D_ay[1,m] +
                     dNdx[1] * Ny[2] * D_ax[1,m] * D_ay[2,m] + dNdx[2] * Ny[2] * D_ax[2,m] * D_ay[2,m]

            ∂d[2] += Nx[1] * dNdy[1] * D_ax[1,m] * D_ay[1,m] + Nx[1] * dNdy[2] * D_ax[1,m] * D_ay[2,m] +
                     Nx[2] * dNdy[1] * D_ax[2,m] * D_ay[1,m] + Nx[2] * dNdy[2] * D_ax[2,m] * D_ay[2,m]
        end
        d += dot(Nx,D_anx) * dot(Ny,D_any) # Add current mode
        ∂d[1] += dNdx[1] * Ny[1] * D_anx[1] * D_any[1] + dNdx[2] * Ny[1] * D_anx[2] * D_any[1] +
                 dNdx[1] * Ny[2] * D_anx[1] * D_any[2] + dNdx[2] * Ny[2] * D_anx[2] * D_any[2]
        ∂d[2] += Nx[1] * dNdy[1] * D_anx[1] * D_any[1] + Nx[1] * dNdy[2] * D_anx[1] * D_any[2] +
                 Nx[2] * dNdy[1] * D_anx[2] * D_any[1] + Nx[2] * dNdy[2] * D_anx[2] * D_any[2]


        dΩ = D.fev.detJdV[q_point]
        gx = (NNx'[:] * (d - 2 * D_mp.l / D_mp.gc * (1-d) * Ψ[q_point]) + D_mp.l^2 * BBx' * ∂d) * dΩ
        gy = (NNy'[:] * (d - 2 * D_mp.l / D_mp.gc * (1-d) * Ψ[q_point]) + D_mp.l^2 * BBy' * ∂d) * dΩ

        g += [gx;
              gy]

    end

    return g
end

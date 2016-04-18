########################
# Elastic displacement #
########################

# Residual
function calc_globres_U{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,U_edof::Matrix{Int},U_free::Vector{Int},
                                           U_mp_tangent::Matrix,b::Vector)
    # Calculate global residual, g_glob
    g_glob = zeros(T,number_of_dofs(U_edof))

    for i = 1:U.mesh.nEl
        # println("Residual element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(U.fev,x)
        U_m = U_edof[:,i]
        ge = U_intf(U_an[U_m],U_a[U_m,:],U,
                              x,U_mp_tangent,b)

        g_glob[U_m] += ge
    end

    return g_glob[U_free]
end

# Tangent
function calc_globK_U{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,U_edof::Matrix{Int},U_free::Vector{Int},
                                         U_mp_tangent::Matrix,b::Vector)
    # Calculate global tangent stiffness matrix, K
    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:U.mesh.nEl
        # println("Stiffness element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(U.fev,x)
        U_m = U_edof[:,i]

        U_intf_closure(U_an) = U_intf(U_an,U_a[U_m,:],U,
                                           U_mp_tangent,b)

        kefunc = ForwardDiff.jacobian(U_intf_closure, cache=cache)
        Ke = kefunc(U_an[U_m])

        JuAFEM.assemble(U_m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[U_free, U_free]
end


################################
# Damaged elastic displacement #
################################

# Residual
function calc_globres_UD{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,U_edof::Matrix{Int},U_free::Vector{Int},
                                            D_a::Matrix,D::PGDFunction,D_edof::Matrix{Int},
                                            U_mp_tangent::Matrix,b::Vector)
    # Calculate global residual, g_glob
    g_glob = zeros(T,number_of_dofs(U_edof))
    Ψ = [zeros(Float64,length(JuAFEM.points(D.fev.quad_rule))) for i in 1:D.mesh.nEl]

    for i = 1:U.mesh.nEl
        # println("Residual element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(U.fev,x)
        JuAFEM.reinit!(D.fev,x)
        U_m = U_edof[:,i]
        D_m = D_edof[:,i]

        ge, Ψ[i] = UD_intf(U_an[U_m],U_a[U_m,:],U,
                               D_a[D_m,:],D,
                               x,U_mp_tangent,b)

        g_glob[U_m] += ge
    end

    return g_glob[U_free], Ψ
end

# Tangent
function calc_globK_UD{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,U_edof::Matrix{Int},U_free::Vector{Int},
                                          D_a::Matrix,D::PGDFunction,D_edof::Matrix{Int},
                                          U_mp_tangent::Matrix,b::Vector)
    # Calculate global tangent stiffness matrix, K
    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:U.mesh.nEl
        # println("Stiffness element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(U.fev,x)
        JuAFEM.reinit!(D.fev,x)
        U_m = U_edof[:,i]
        D_m = D_edof[:,i]

        function UD_intf_closure(U_an)
            g, _ = UD_intf(U_an,U_a[U_m,:],U,
                                D_a[D_m,:],D,
                                x,U_mp_tangent,b)
            return g
        end

        kefunc = ForwardDiff.jacobian(UD_intf_closure, cache=cache)
        Ke = kefunc(U_an[U_m])

        JuAFEM.assemble(U_m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[U_free, U_free]
end


##########
# Damage #
##########

# Residual
function calc_globres_DU{T}(D_an::Vector{T},D_a::Matrix,D::PGDFunction,D_edof::Matrix{Int},D_free::Vector{Int},
                                            U_a::Matrix,U::PGDFunction,U_edof::Matrix{Int},
                                            D_mp::PhaseFieldDamage,Ψ::Vector{Vector{Float64}})
    # Calculate global residual, g_glob
    g_glob = zeros(T,number_of_dofs(D_edof))

    for i = 1:D.mesh.nEl
        # println("Residual element #$i")
        x = [D.mesh.ex[:,i] D.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(D.fev,x)
        JuAFEM.reinit!(U.fev,x)
        D_m = D_edof[:,i]
        U_m = U_edof[:,i]

        ge = DU_intf(D_an[D_m],D_a[D_m,:],D,
                               U_a[U_m,:],U,
                               x,D_mp,Ψ[i])

        g_glob[D_m] += ge
    end

    return g_glob[D_free]
end

# Tangent
function calc_globK_DU{T}(D_an::Vector{T},D_a::Matrix,D::PGDFunction,D_edof::Matrix{Int},D_free::Vector{Int},
                                          U_a::Matrix,U::PGDFunction,U_edof::Matrix{Int},
                                          D_mp::PhaseFieldDamage,Ψ::Vector{Vector{Float64}})

    # Calculate global tangent stiffness matrix, K
    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:U.mesh.nEl
        # println("Stiffness element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(D.fev,x)
        JuAFEM.reinit!(U.fev,x)
        D_m = D_edof[:,i]
        U_m = U_edof[:,i]

        DU_intf_closure(D_an) = DU_intf(D_an,D_a[D_m,:],D,
                                             U_a[U_m,:],U,
                                             x,D_mp,Ψ[i])

        kefunc = ForwardDiff.jacobian(DU_intf_closure, cache=cache)
        Ke = kefunc(D_an[D_m])

        JuAFEM.assemble(D_m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[D_free, D_free]
end

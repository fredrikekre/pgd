########################
# Elastic displacement #
########################

# Residual
function calc_globres_U{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,D_mat::Matrix,U_edof::Matrix,b::Vector,free)
    # Calculate global residual, g_glob
    g_glob = zeros(T,number_of_dofs(U_edof))

    for i = 1:U.mesh.nEl
        # println("Residual element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(U.fev,x)
        m = U_edof[:,i]
        ge = U_intf(U_an[m],U_a[m,:],x,U,D_mat,b)

        g_glob[m] += ge
    end

    return g_glob[free]
end

# Tangent
function calc_globK_U{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,D_mat::Matrix,U_edof::Matrix,b::Vector,free)
    # Calculate global tangent stiffness matrix, K
    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:U.mesh.nEl
        # println("Stiffness element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(U.fev,x)
        m = U_edof[:,i]

        U_intf_closure(U_an) = U_intf(U_an,U_a[m,:],x,U,D_mat,b)

        kefunc = ForwardDiff.jacobian(U_intf_closure, cache=cache)
        Ke = kefunc(U_an[m])

        JuAFEM.assemble(m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[free, free]
end


######################
# Damaged elasticity #
######################

# Residual
function calc_globres_UD{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,U_edof::Matrix{Int},U_free::Vector{Int},
                                            D_a::Matrix,D::PGDFunction,D_edof::Matrix{Int},
                                            D_mat::Matrix,b::Vector)
    # Calculate global residual, g_glob
    g_glob = zeros(T,number_of_dofs(U_edof))

    for i = 1:U.mesh.nEl
        # println("Residual element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(U.fev,x)
        JuAFEM.reinit!(D.fev,x)
        U_m = U_edof[:,i]
        D_m = D_edof[:,i]

        ge = UD_intf(U_an[U_m],U_a[U_m,:],U,
                               D_a[D_m,:],D,
                     x,D_mat,b)

        g_glob[U_m] += ge
    end

    return g_glob[U_free]
end

# Tangent
function calc_globK_UD{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,U_edof::Matrix{Int},U_free::Vector{Int},
                                          D_a::Matrix,D::PGDFunction,D_edof::Matrix{Int},
                                          D_mat::Matrix,b::Vector)
    # Calculate global tangent stiffness matrix, K
    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:U.mesh.nEl
        # println("Stiffness element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(U.fev,x)
        m = U_edof[:,i]

        U_intf_closure(U_an) = U_intf(U_an,U_a[m,:],x,U,D_mat,b)

        kefunc = ForwardDiff.jacobian(U_intf_closure, cache=cache)
        Ke = kefunc(U_an[m])

        JuAFEM.assemble(m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[U_free, U_free]
end


##########
# Damage #
##########

# Residual
function calc_globres_DU{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,D_mat::Matrix,U_edof::Matrix,b::Vector,free)
    # Calculate global residual, g_glob
    g_glob = zeros(T,number_of_dofs(U_edof))

    for i = 1:U.mesh.nEl
        # println("Residual element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(U.fev,x)
        m = edof[:,i]
        ge = U_intf(U_an[m],U_a[m,:],x,U,D_mat,b)

        g_glob[m] += ge
    end

    return g_glob[free]
end

# Tangent
function calc_globK_DU{T}(U_an::Vector{T},U_a::Matrix,U::PGDFunction,D_mat::Matrix,U_edof::Matrix,b::Vector,free)
    # Calculate global tangent stiffness matrix, K
    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:U.mesh.nEl
        # println("Stiffness element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(U.fev,x)
        m = U_edof[:,i]

        U_intf_closure(U_an) = U_intf(U_an,U_a[m,:],x,U,D_mat,b)

        kefunc = ForwardDiff.jacobian(U_intf_closure, cache=cache)
        Ke = kefunc(U_an[m])

        JuAFEM.assemble(m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[free, free]
end

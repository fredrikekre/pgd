################
# Displacement #
################

function U_residual{T}(u::Vector{T},u_mesh,u_free,u_fe_values,mp,b::Vector=zeros(2))
    # Calculate global residual, g
    ndofs = maximum(u_mesh.edof)
    g = zeros(ndofs)

    for i = 1:u_mesh.nEl
        x = [u_mesh.ex[:,i] u_mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(u_fe_values,x)
        u_m = u_mesh.edof[:,i]

        ge = U_intf(u[u_m],u_fe_values,mp,b)

        g[u_m] += ge
    end

    return g[u_free]
end

function U_jacobian{T}(u::Vector{T},u_mesh,u_free,u_fe_values,mp,b::Vector=zeros(2))
    # Calculate global tangent stiffness matrix, K

    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:u_mesh.nEl
        x = [u_mesh.ex[:,i] u_mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(u_fe_values,x)

        JuAFEM.reinit!(u_fe_values,x)
        u_m = u_mesh.edof[:,i]

        U_intf_closure(u) = U_intf(u,u_fe_values,mp,b)

        kefunc = ForwardDiff.jacobian(U_intf_closure, cache=cache)
        Ke = kefunc(u[u_m])

        JuAFEM.assemble(u_m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[u_free, u_free]
end


############################
# Displacement with damage #
############################

function UD_residual{T}(u::Vector{T},u_mesh,u_free,u_fe_values,d,d_mesh,d_fe_values,mp,b::Vector)
    # Calculate global residual, g
    ndofs = maximum(u_mesh.edof)
    g = zeros(ndofs)
    Ψ = [zeros(length(JuAFEM.points(u_fe_values.quad_rule))) for i in 1:u_mesh.nEl]

    for i = 1:u_mesh.nEl
        x = [u_mesh.ex[:,i] u_mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(u_fe_values,x)
        JuAFEM.reinit!(d_fe_values,x)
        u_m = u_mesh.edof[:,i]
        d_m = d_mesh.edof[:,i]

        ge, Ψ[i] = UD_intf(u[u_m],d[d_m],u_fe_values,d_fe_values,mp,b)

        g[u_m] += ge
    end

    return g[u_free], Ψ
end

function UD_jacobian{T}(u::Vector{T},u_mesh,u_free,u_fe_values,d,d_mesh,d_fe_values,mp,b::Vector)
    # Calculate global tangent stiffness matrix, K

    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:u_mesh.nEl
        x = [u_mesh.ex[:,i] u_mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(u_fe_values,x)

        JuAFEM.reinit!(u_fe_values,x)
        JuAFEM.reinit!(d_fe_values,x)
        u_m = u_mesh.edof[:,i]
        d_m = d_mesh.edof[:,i]

        function UD_intf_closure(u)
            g,_ = UD_intf(u,d[d_m],u_fe_values,d_fe_values,mp,b)
            return g
        end

        kefunc = ForwardDiff.jacobian(UD_intf_closure, cache=cache)
        Ke = kefunc(u[u_m])

        JuAFEM.assemble(u_m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[u_free, u_free]
end


##########
# Damage #
##########

function DU_residual(d,d_mesh,d_free,d_fe_values,d_mp,Ψ)
    # Calculate global residual, g
    ndofs = maximum(d_mesh.edof)
    g = zeros(ndofs)

    for i = 1:d_mesh.nEl
        x = [d_mesh.ex[:,i] d_mesh.ey[:,i]]'
        x = reinterpret(Vec{2,Float64},x,(size(x,2),))
        JuAFEM.reinit!(d_fe_values,x)
        d_m = d_mesh.edof[:,i]

        ge = DU_intf_min_egen(d[d_m],d_fe_values,d_mp,Ψ[i])

        g[d_m] += ge
    end

    return g[d_free]
end

function DU_jacobian(d,d_mesh,d_free,d_fe_values,d_mp,Ψ)
    # Calculate global tangent stiffness matrix, K

    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:d_mesh.nEl
        x = [d_mesh.ex[:,i] d_mesh.ey[:,i]]'
        x = reinterpret(Vec{2,Float64},x,(size(x,2),))
        JuAFEM.reinit!(d_fe_values,x)
        d_m = d_mesh.edof[:,i]

        # # AD 1
        # d_intf_closure(d) = d_intf(d,d_old[d_m],u[u_m],d_fe_values,u_fe_values,mp,Ψ[i],Δt)

        # kefunc = ForwardDiff.jacobian(d_intf_closure, cache=cache)
        # Ke = kefunc(d[d_m])

        # AD 2
        DU_intf_min_egen_closure(d) = DU_intf_min_egen(d,d_fe_values,d_mp,Ψ[i])

        kefunc = ForwardDiff.jacobian(DU_intf_min_egen_closure, cache=cache)
        Ke = kefunc(d[d_m])

        # Ke = damageStiffness(d[d_m],d_old[d_m],u[u_m],d_fe_values,u_fe_values,mp,Ψ[i],Δt)

        JuAFEM.assemble(d_m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[d_free, d_free]
end

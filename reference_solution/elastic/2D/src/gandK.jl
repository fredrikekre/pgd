################
# Displacement #
################

function u_intf{T}(u::Vector{T},u_fe_values,mp,b::Vector=zeros(2))

    n_basefuncs = n_basefunctions(get_functionspace(u_fe_values))
    u_tensor = reinterpret(Vec{2,T},u,(4,))

    g = [zero(Tensor{1, 2, T}) for i in 1:n_basefuncs]

    for q_point in 1:length(u_fe_values.quad_rule.points)

        ε = function_vector_symmetric_gradient(u_fe_values, q_point, u_tensor)

        σ = 2 * mp.G * dev(ε) + mp.K * trace(ε)* one(ε)

        for i = 1:n_basefuncs
            g[i] += σ ⋅ shape_gradient(u_fe_values,q_point,i) *  detJdV(u_fe_values,q_point)
        end
    end

    return reinterpret(T, g, (2 * n_basefuncs,))
end

function u_residual{T}(u::Vector{T},u_mesh,u_free,u_fe_values,mp,b::Vector=zeros(2))
    # Calculate global residual, g
    ndofs = maximum(u_mesh.edof)
    g = zeros(ndofs)

    for i = 1:u_mesh.nEl
        x = [u_mesh.ex[:,i] u_mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(u_fe_values,x)
        u_m = u_mesh.edof[:,i]

        ge = u_intf(u[u_m],u_fe_values,mp,b)

        g[u_m] += ge
    end

    return g[u_free]
end

function u_jacobian{T}(u::Vector{T},u_mesh,u_free,u_fe_values,mp,b::Vector=zeros(2))
    # Calculate global tangent stiffness matrix, K

    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:u_mesh.nEl
        x = [u_mesh.ex[:,i] u_mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(u_fe_values,x)

        JuAFEM.reinit!(u_fe_values,x)
        u_m = u_mesh.edof[:,i]

        u_intf_closure(u) = u_intf(u,u_fe_values,mp,b)

        kefunc = ForwardDiff.jacobian(u_intf_closure, cache=cache)
        Ke = kefunc(u[u_m])

        JuAFEM.assemble(u_m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[u_free, u_free]
end

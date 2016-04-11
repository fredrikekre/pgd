################
# Displacement #
################

function u_intf{T,Q}(u::Vector{T},d::Vector{Q},u_fe_values,d_fe_values,mp,b::Vector=zeros(2))

    n_basefuncs = n_basefunctions(get_functionspace(u_fe_values))
    u_tensor = reinterpret(Vec{2,T},u,(4,))
    # d_tensor = reinterpret(Vec{1,Q},d,(4,))

    g = [zero(Tensor{1, 2, T}) for i in 1:n_basefuncs]

    for q_point in 1:length(u_fe_values.quad_rule.points)

        ε = function_vector_symmetric_gradient(u_fe_values, q_point, u_tensor)

        d_value = function_scalar_value(d_fe_values,q_point,d)
        rf = 1e-6
        σ_dev_degradation = (1.0-d_value)^2 + rf

        σ = σ_dev_degradation * (2 * mp.G * dev(ε) + mp.K * trace(ε)* one(ε))

        for i = 1:n_basefuncs
            g[i] += σ ⋅ shape_gradient(u_fe_values,q_point,i) *  detJdV(u_fe_values,q_point)
        end
    end

    return reinterpret(T, g, (2 * n_basefuncs,))
end

function u_residual{T}(u::Vector{T},u_mesh,u_free,u_fe_values,d,d_mesh,d_fe_values,mp,b::Vector=zeros(2))
    # Calculate global residual, g
    ndofs = maximum(u_mesh.edof)
    g = zeros(ndofs)

    for i = 1:u_mesh.nEl
        x = [u_mesh.ex[:,i] u_mesh.ey[:,i]]'
        x = reinterpret(Vec{2,T},x,(size(x,2),))
        JuAFEM.reinit!(u_fe_values,x)
        JuAFEM.reinit!(d_fe_values,x)
        u_m = u_mesh.edof[:,i]
        d_m = d_mesh.edof[:,i]

        ge = u_intf(u[u_m],d[d_m],u_fe_values,d_fe_values,mp,b)

        g[u_m] += ge
    end

    return g[u_free]
end

function u_jacobian{T}(u::Vector{T},u_mesh,u_free,u_fe_values,d,d_mesh,d_fe_values,mp,b::Vector=zeros(2))
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

        u_intf_closure(u) = u_intf(u,d[d_m],u_fe_values,d_fe_values,mp,b)

        kefunc = ForwardDiff.jacobian(u_intf_closure, cache=cache)
        Ke = kefunc(u[u_m])

        JuAFEM.assemble(u_m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[u_free, u_free]
end


##########
# Damage #
##########

# function d_intf{T}(d::Vector{T},d_old,ue::Vector,d_fe_values,u_fe_values,mp,Ψe,Δt)

#     n_basefuncs = n_basefunctions(get_functionspace(d_fe_values))

#     g = zeros(T,n_basefuncs)
    
#     for q_point in 1:length(d_fe_values.quad_rule.points)

#         d_value = function_scalar_value(d_fe_values,q_point,d)
#         d_old_value = function_scalar_value(d_fe_values,q_point,d_old)

#         McB = 1/2 * (abs(d_value-d_old_value) - (d_value-d_old_value))
#         ∂d = function_scalar_gradient(d_fe_values,q_point,d)
#         gprim = - 2 * (1-d_value)

#         for i in 1:n_basefuncs
#             N = shape_value(d_fe_values,q_point,i)
#             B = shape_gradient(d_fe_values,q_point,i)

#             ge = - N * mp.rp / Δt * McB + mp.gc / mp.l *(N*d_value + mp.l^2 * B ⋅ ∂d) + N * gprim * Ψe[q_point]
#             #ge *=Δt
#             g[i] += ge * detJdV(d_fe_values,q_point)
#         end

#     end

#     return g
# end

# function damageStiffness{T}(d::Vector{T},d_old,ue::Vector,d_fe_values,u_fe_values,mp,Ψe,Δt)

#     n_basefuncs = n_basefunctions(get_functionspace(d_fe_values))

#     ke = zeros(T,n_basefuncs,n_basefuncs)

#     for q_point in 1:length(d_fe_values.quad_rule.points)

#         d_value = function_scalar_value(d_fe_values,q_point,d)
#         d_old_value = function_scalar_value(d_fe_values,q_point,d_old)

#         McB = 1/2 * (abs(d_value-d_old_value) - (d_value-d_old_value))
#         ∂d = function_scalar_gradient(d_fe_values,q_point,d)
#         gprim = - 2 * (1-d_value)
#         gprimprim = 2.0

#         N = shape_value(d_fe_values,q_point)
#         B = shape_gradient(d_fe_values,q_point)
#         B = reinterpret(Float64,B,(2,4))

#         kee = (N * N') * (-mp.rp/2*(sign(d_value-d_old_value)-1) + mp.gc/mp.l*Δt + Ψe[q_point] * gprimprim*Δt) + Δt*mp.gc*mp.l*(B'*B)
#         # kee = kee ./ Δt

#         ke += kee * detJdV(d_fe_values,q_point)

#     end

#     return ke
# end

function d_intf_min_egen{T}(d::Vector{T},d_old,ue::Vector,d_fe_values,u_fe_values,mp,Ψe)

    n_basefuncs = n_basefunctions(get_functionspace(d_fe_values))

    g = zeros(T,n_basefuncs)
    
    for q_point in 1:length(d_fe_values.quad_rule.points)

        d_value = function_scalar_value(d_fe_values,q_point,d)
        #d_old_value = function_scalar_value(d_fe_values,q_point,d_old)

        #McB = 1/2 * (abs(d_value-d_old_value) - (d_value-d_old_value))
        ∂d = function_scalar_gradient(d_fe_values,q_point,d)
        #gprim = - 2 * (1-d_value)

        for i in 1:n_basefuncs
            N = shape_value(d_fe_values,q_point,i)
            B = shape_gradient(d_fe_values,q_point,i)
            
            ge = N * (mp.gc / mp.l * d_value -2*(1-d_value)*Ψe[q_point]) + mp.gc*mp.l * (B ⋅ ∂d)
            g[i] += ge * detJdV(d_fe_values,q_point)
        end
    end

    return g
end

function d_residual(d,d_old,d_mesh,d_free,d_fe_values,u,u_mesh,u_fe_values,mp,Ψ)
    # Calculate global residual, g
    ndofs = maximum(d_mesh.edof)
    g = zeros(ndofs)

    for i = 1:d_mesh.nEl
        x = [d_mesh.ex[:,i] d_mesh.ey[:,i]]'
        x = reinterpret(Vec{2,Float64},x,(size(x,2),))
        JuAFEM.reinit!(d_fe_values,x)
        JuAFEM.reinit!(u_fe_values,x)
        d_m = d_mesh.edof[:,i]
        u_m = u_mesh.edof[:,i]

        Δt = 0.015
        # ge = d_intf(d[d_m],d_old[d_m],u[u_m],d_fe_values,u_fe_values,mp,Ψ[i],Δt)
        ge = d_intf_min_egen(d[d_m],d_old[d_m],u[u_m],d_fe_values,u_fe_values,mp,Ψ[i])

        g[d_m] += ge
    end

    return g[d_free]
end

function d_jacobian(d,d_old,d_mesh,d_free,d_fe_values,u,u_mesh,u_fe_values,mp,Ψ)
    # Calculate global tangent stiffness matrix, K

    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:u_mesh.nEl
        x = [d_mesh.ex[:,i] d_mesh.ey[:,i]]'
        x = reinterpret(Vec{2,Float64},x,(size(x,2),))
        JuAFEM.reinit!(d_fe_values,x)
        JuAFEM.reinit!(u_fe_values,x)
        d_m = d_mesh.edof[:,i]
        u_m = u_mesh.edof[:,i]

        # # AD 1
        # d_intf_closure(d) = d_intf(d,d_old[d_m],u[u_m],d_fe_values,u_fe_values,mp,Ψ[i],Δt)

        # kefunc = ForwardDiff.jacobian(d_intf_closure, cache=cache)
        # Ke = kefunc(d[d_m])

        # AD 2
        d_intf_min_egen_closure(d) = d_intf_min_egen(d,d_old[d_m],u[u_m],d_fe_values,u_fe_values,mp,Ψ[i])

        kefunc = ForwardDiff.jacobian(d_intf_min_egen_closure, cache=cache)
        Ke = kefunc(d[d_m])

        # Ke = damageStiffness(d[d_m],d_old[d_m],u[u_m],d_fe_values,u_fe_values,mp,Ψ[i],Δt)

        JuAFEM.assemble(d_m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[d_free, d_free]
end


###############
# Free energy #
###############

function Ψ_intf{T}(u::Vector{T},de::Vector,u_fe_values,d_fe_values,mp,Ψe::Vector)

    n_basefuncs = n_basefunctions(get_functionspace(u_fe_values))
    u = reinterpret(Vec{2,T},u,(n_basefuncs,))
    σe_plot = zeros(4)

    for q_point in 1:length(u_fe_values.quad_rule.points)

        ε = function_vector_symmetric_gradient(u_fe_values,q_point,u)
        d_value = function_scalar_value(d_fe_values,q_point,de)
        σ_dev_degradation = (1.0-d_value)^2
        σ_dev_degradation = 1.0

        ε_3 = convert(SymmetricTensor{2,3},ε)

        σ = σ_dev_degradation*( 2 * mp.G * dev(ε_3) + mp.K * trace(ε_3)* one(ε_3))
        σ_dev = 2 * mp.G * dev(ε_3) * σ_dev_degradation #+ mp.K * trace(ε)* one(dev(ε))

        # Calculate free energy
        σ_e = sqrt(3/2) * sqrt(σ_dev ⊡ σ_dev)
        σe_plot[q_point] = σ_e

        
        Ψe[q_point] = max(1/2 * ε_3 ⊡ σ,Ψe[q_point])

    end

    return Ψe, σe_plot
end

negpart(x) = 1/2*(abs(x) - x)
pospart(x) = 1/2*(abs(x) + x)

function Ψ_intf_plus{T}(u::Vector{T},de::Vector,u_fe_values,d_fe_values,mp,Ψe::Vector)

    n_basefuncs = n_basefunctions(get_functionspace(u_fe_values))
    u = reinterpret(Vec{2,T},u,(n_basefuncs,))
    σe_plot = zeros(4)

    # Material params
    λ = mp.E*mp.ν/(1+mp.ν)/(1-2*mp.ν)
    μ = mp.G

    for q_point in 1:length(u_fe_values.quad_rule.points)

        ε = function_vector_symmetric_gradient(u_fe_values,q_point,u)
        d_value = function_scalar_value(d_fe_values,q_point,de)
        σ_dev_degradation = (1.0-d_value)^2

        # Effective degraded stress
        σ_dev3D = 2 * mp.G * dev(convert(SymmetricTensor{2,3},ε)) * σ_dev_degradation #+ mp.K * trace(ε)* one(dev(ε))
        σ_e = sqrt(3/2) * sqrt(σ_dev3D ⊡ σ_dev3D)
        σe_plot[q_point] = σ_e

        # Calculate free energy
        ε = reshape(ε[:],(2,2))
        Λ,Φ = eig(ε)
        Λ_plus = diagm(pospart(Λ))
        ε_plus = Φ*Λ_plus*Φ'
        # ε_plus_t = Tensor{2,3}()

        Ψ_plus = λ*pospart(trace(ε))^2 + μ*trace(ε_plus*ε_plus)
        # Ψ_plus2 = 2*mp.G * dev(convert(Tensor{1,3},ε_plus)) + mp.K * trace()

        # Ψe[q_point] = max(Ψ_plus,Ψe[q_point])
        if Ψ_plus > Ψ[q_point]
            Ψ[q_point] = Ψ_plus
        else
            println("I AM LOWER!")
        end

    end

    return Ψe, σe_plot
end
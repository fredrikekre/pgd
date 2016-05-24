################
# Displacement #
################

function U_intf{T}(u::Vector{T},u_fe_values,mp,b::Vector)

    n_basefuncs = n_basefunctions(get_functionspace(u_fe_values))
    u_tensor = reinterpret(Vec{2,T},u,(4,))

    g = [zero(Tensor{1, 2, T}) for i in 1:n_basefuncs]

    for q_point in 1:length(u_fe_values.quad_rule.points)

        ε = function_vector_symmetric_gradient(u_fe_values, q_point, u_tensor)

        σ = 2 * mp.G * dev(ε) + mp.K * trace(ε)* one(ε)

        for i = 1:n_basefuncs
            g[i] += (σ ⋅ shape_gradient(u_fe_values,q_point,i)- shape_value(u_fe_values,q_point,i) * convert(Tensor{1,2},b) ) *  detJdV(u_fe_values,q_point)
        end
    end

    return reinterpret(T, g, (2 * n_basefuncs,))
end


############################
# Displacement with damage #
############################

function UD_intf{T,Q}(u::Vector{T},d::Vector{Q},u_fe_values,d_fe_values,mp,b::Vector)

    n_basefuncs = n_basefunctions(get_functionspace(u_fe_values))
    u_tensor = reinterpret(Vec{2,T},u,(4,))
    # d_tensor = reinterpret(Vec{1,Q},d,(4,))
    Ψe = zeros(length(u_fe_values.quad_rule.points))

    g = [zero(Tensor{1, 2, T}) for i in 1:n_basefuncs]

    for q_point in 1:length(u_fe_values.quad_rule.points)

        ε = function_vector_symmetric_gradient(u_fe_values, q_point, u_tensor)

        d_value = function_scalar_value(d_fe_values,q_point,d)
        rf = 1e-6*1000
        σ_dev_degradation = (1.0-d_value)^2 + rf

        σ = σ_dev_degradation * (2 * mp.G * dev(ε) + mp.K * trace(ε)* one(ε))

        for i = 1:n_basefuncs
            g[i] += (σ ⋅ shape_gradient(u_fe_values,q_point,i) - shape_value(u_fe_values,q_point,i) * convert(Tensor{1,2},b) ) *  detJdV(u_fe_values,q_point)
        end

        #############################
        # Calculate the energy here #
        #############################
        if T == Float64 # Otherwise its GradientNumber...
            ε3 = convert(SymmetricTensor{2,3},ε)
            σ = 2 * mp.G * dev(ε3) + mp.K * trace(ε3)* one(ε3)
            Ψ = 1/2 * σ ⊡ ε3
            Ψe[q_point] = Ψ
        end
    end

    return reinterpret(T, g, (2 * n_basefuncs,)), Ψe
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

function DU_intf_min_egen{T}(d::Vector{T},d_fe_values,d_mp,Ψe)

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

            ge = N * (d_value - 2 * d_mp.l / d_mp.gc * (1-d_value) * Ψe[q_point]) + d_mp.l^2 * (B ⋅ ∂d)

            g[i] += ge * detJdV(d_fe_values,q_point)
        end
    end

    return g
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

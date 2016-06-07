function get_shape_functions{dim,T}(u_fe_values::FEValues{dim,T})

    N = Vector{Vec{dim,T}}[]
    dN = Vector{Tensor{2,dim,T}}[]

    nqp = length(points(u_fe_values.quad_rule))

    for qp in 1:nqp
        Nqp = Vec{dim,T}[]
        dNqp = Tensor{2,dim,T}[]

        for dof in 1:3*2^dim
            counterN = ceil(Int,dof/dim)
            counterDim = mod(dof,dim); if counterDim == 0; counterDim = dim; end

            # Shape functions
            thisN = zeros(T,dim)
            thisN[counterDim] = u_fe_values.N[qp][counterN]
            N_ = Vec{dim,T}((thisN...))
            push!(Nqp,N_)

            # Derivatives
            thisdN = zeros(T,dim,dim)
            thisdN[counterDim,:] = u_fe_values.dNdx[qp][counterN][:]
            dN_ = Tensor{2,dim,T}((thisdN...))
            push!(dNqp, dN_)
        end
        push!(N,Nqp)
        push!(dN,dNqp)
    end
    return N, dN
end

function elastic_3D_element{dim,T}(ue, N, dN, u_fe_values::FEValues{dim,T}, E, b)

    ge = zeros(T,24)
    Ke = zeros(T,24,24)

    for qp in 1:length(points(u_fe_values.quad_rule))

        dΩ = u_fe_values.quad_rule.weights[qp]

        ∇u = zero(Tensor{2,dim,T})
        for dof in 1:24
            ∇u += ue[dof] * dN[qp][dof]
        end
        ε = symmetric(∇u)

        for dof1 in 1:24
            gedof1 = (dN[qp][dof1] ⊡ (E ⊡ ε) - N[qp][dof1] ⋅ b) * dΩ
            ge[dof1] += gedof1
            for dof2 in 1:24
                kedof1dof2 = (dN[qp][dof1] ⊡ (E ⊡ symmetric(dN[qp][dof2]))) * dΩ
                Ke[dof1,dof2] += kedof1dof2
            end
        end
    end
    return ge, Ke
end

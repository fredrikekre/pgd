# Cache some stuff so we don't need to recreate them in every function call
type Buffers{T}
    NNx::Matrix{T}
    NNy::Matrix{T}
    BBx::Matrix{T}
    BBy::Matrix{T}
    g::Vector{T}
    ε::Vector{T}
    ε_m::Vector{T} # ε for a mode
    dNdx::Matrix{Float64}
    dNdy::Matrix{Float64}
    Nx::Vector{Float64}
    Ny::Vector{Float64}

end

function Buffers{T}(Tv::Type{T})
    g = zeros(T,8) # Not general

    Nx = zeros(2) # Shape functions
    Ny = zeros(2)

    dNdx = zeros(1,2) # Derivatives
    dNdy = zeros(1,2)

    NNx = zeros(T,2,4) # N matrix for force vector
    NNy = zeros(T,2,4)

    BBx = zeros(T,3,4) # B matrix for stiffness
    BBy = zeros(T,3,4)

    ε = zeros(T, 3)
    ε_m = zeros(T, 3)

    return Buffers(NNx, NNy, BBx, BBy, g, ε, ε_m, dNdx, dNdy, Nx, Ny)
end

type BufferCollection{Q, T}
    buff_grad::Buffers{Q}
    buff_float::Buffers{T}
end

function BufferCollection{Q, T}(Tq::Type{Q}, Tt::Type{T})
    BufferCollection(Buffers(Tq), Buffers(Tt))
end

import ForwardDiff.GradientNumber
# The 8 here is the "chunk size" used
Tgrad = ForwardDiff.GradientNumber{8, Float64, NTuple{8, Float64}}
get_buffer{T <: GradientNumber}(buff_coll::BufferCollection, ::Type{T}) = buff_coll.buff_grad
get_buffer{T}(buff_coll::BufferCollection, ::Type{T}) = buff_coll.buff_float


const buff_colls = BufferCollection(Tgrad, Float64)

function intf{T}(an::Vector{T},a::Matrix,x::Matrix,U::PGDFunction,D::Matrix,b::Vector=zeros(2))
    # an is the unknowns
    # a are the already computed modes
    # 4 node quadrilateral element
    anx = an[1:4] # Not general
    any = an[5:8]
    ax = a[1:4,:]
    ay = a[5:8,:]

    buff_coll = get_buffer(buff_colls, T)

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
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        # Update values
        ex_x = [U.components[1].mesh.x[1] U.components[1].mesh.x[2]] # only for equidistant mesh
        ex_y = [U.components[2].mesh.x[1] U.components[2].mesh.x[2]]
        evaluate_at_gauss_point!(U.components[1].fev,[ξ_x],ex_x,Nx,dNdx)
        evaluate_at_gauss_point!(U.components[2].fev,[ξ_y],ex_y,Ny,dNdy)

        NNx[1,1] = Nx[1] * Ny[1] * any[1] + Nx[1] * Ny[2] * any[3]
        NNx[2,2] = Nx[1] * Ny[1] * any[2] + Nx[1] * Ny[2] * any[4]
        NNx[1,3] = Nx[2] * Ny[1] * any[1] + Nx[2] * Ny[2] * any[3]
        NNx[2,4] = Nx[2] * Ny[1] * any[2] + Nx[2] * Ny[2] * any[4]

        NNy[1,1] = Nx[1] * Ny[1] * anx[1] + Nx[2] * Ny[1] * anx[3]
        NNy[2,2] = Nx[1] * Ny[1] * anx[2] + Nx[2] * Ny[1] * anx[4]
        NNy[1,3] = Nx[1] * Ny[2] * anx[1] + Nx[2] * Ny[2] * anx[3]
        NNy[2,4] = Nx[1] * Ny[2] * anx[2] + Nx[2] * Ny[2] * anx[4]

        BBx[1,1] = dNdx[1] * Ny[1] * any[1] + dNdx[1] * Ny[2] * any[3]
        BBx[3,1] = Nx[1] * dNdy[1] * any[1] + Nx[1] * dNdy[2] * any[3]
        BBx[2,2] = Nx[1] * dNdy[1] * any[2] + Nx[1] * dNdy[2] * any[4]
        BBx[3,2] = dNdx[1] * Ny[1] * any[2] + dNdx[1] * Ny[2] * any[4]
        BBx[1,3] = dNdx[2] * Ny[1] * any[1] + dNdx[2] * Ny[2] * any[3]
        BBx[3,3] = Nx[2] * dNdy[1] * any[1] + Nx[2] * dNdy[2] * any[3]
        BBx[2,4] = Nx[2] * dNdy[1] * any[2] + Nx[2] * dNdy[2] * any[4]
        BBx[3,4] = dNdx[2] * Ny[1] * any[2] + dNdx[2] * Ny[2] * any[4]

        BBy[1,1] = dNdx[1] * Ny[1] * anx[1] + dNdx[2] * Ny[1] * anx[3]
        BBy[3,1] = Nx[1] * dNdy[1] * anx[1] + Nx[2] * dNdy[1] * anx[3]
        BBy[2,2] = Nx[1] * dNdy[1] * anx[2] + Nx[2] * dNdy[1] * anx[4]
        BBy[3,2] = dNdx[1] * Ny[1] * anx[2] + dNdx[2] * Ny[1] * anx[4]
        BBy[1,3] = dNdx[1] * Ny[2] * anx[1] + dNdx[2] * Ny[2] * anx[3]
        BBy[3,3] = Nx[1] * dNdy[2] * anx[1] + Nx[2] * dNdy[2] * anx[3]
        BBy[2,4] = Nx[1] * dNdy[2] * anx[2] + Nx[2] * dNdy[2] * anx[4]
        BBy[3,4] = dNdx[1] * Ny[2] * anx[2] + dNdx[2] * Ny[2] * anx[4]

        ε = buff_coll.ε
        fill!(ε, 0.0)
        ε_m = buff_coll.ε_m

        for m = 1:nModes(U)
            ε_m[1] = dNdx[1] * Ny[1] * ax[1,m] * ay[1,m] + dNdx[2] * Ny[1] * ax[3,m] * ay[1,m] +
                     dNdx[1] * Ny[2] * ax[1,m] * ay[3,m] + dNdx[2] * Ny[2] * ax[3,m] * ay[3,m]

            ε_m[2] = Nx[1] * dNdy[1] * ax[2,m] * ay[2,m] + Nx[1] * dNdy[2] * ax[2,m] * ay[4,m] +
                     Nx[2] * dNdy[1] * ax[4,m] * ay[2,m] + Nx[2] * dNdy[2] * ax[4,m] * ay[4,m]

            ε_m[3] = Nx[1] * dNdy[1] * ax[1,m] * ay[1,m] + Nx[1] * dNdy[2] * ax[1,m] * ay[3,m] +
                     Nx[2] * dNdy[1] * ax[3,m] * ay[1,m] + Nx[2] * dNdy[2] * ax[3,m] * ay[3,m] +
                     dNdx[1] * Ny[1] * ax[2,m] * ay[2,m] + dNdx[2] * Ny[1] * ax[4,m] * ay[2,m] +
                     dNdx[1] * Ny[2] * ax[2,m] * ay[4,m] + dNdx[2] * Ny[2] * ax[4,m] * ay[4,m]

            ε += ε_m
        end

        ε += BBx*anx # eller BBy*ay, blir samma
        σ = D*ε


        dΩ = U.fev.detJdV[q_point]

        gx = (BBx' * σ - NNx' * b) * dΩ
        gy = (BBy' * σ - NNy' * b) * dΩ

        g += [gx; gy] # Här får man typ assemblera då istället om man har en unstructured mesh
    end

    return g

end


# function intf(an::Vector,a::Matrix,x::Matrix,U::PGDFunction,D::Matrix)
#     # an is the unknowns
#     # a are the already computed modes

#     # 4 node quadrilaterla element
#     anx = an[1:4] # Obs inte generellt
#     any = an[5:8]
#     ax = a[1:4,:]
#     ay = a[5:8,:]

#     g = zeros(8) # Residual inte heller generell än

#     _Nx = zeros(2) # Bara basfunktionerna
#     _Ny = zeros(2)

#     _dNdx = zeros(1,2) # Ensamma derivator
#     _dNdy = zeros(1,2)

#     Nx = zeros(2,4) # Utplacerade för 2D
#     Ny = zeros(2,4)

#     dNdx = zeros(2,4) # Utplacerade derivator (detta kan nog göras bättre)
#     dNdy = zeros(2,4)

#     NNx = zeros(2,4) # Slutgiltig N matris (för kraftvektorn typ)
#     NNy = zeros(2,4)

#     BBx = zeros(3,4) # Final B-matrix for internal forces
#     BBy = zeros(3,4)
#     m_BBx = zeros(3,4)

#     for (q_point, (ξ,w)) in enumerate(zip(U.fev.quad_rule.points,U.fev.quad_rule.weights))
#         ξ_x = ξ[1]
#         ξ_y = ξ[2]
#         # Update values
#         evaluate_at_gauss_point!(U.components[1].fev,[ξ_x],[0 1],_Nx,_dNdx)
#         evaluate_at_gauss_point!(U.components[2].fev,[ξ_y],[0 1],_Ny,_dNdy)

#         Nx[1,[1,3]] = _Nx
#         Nx[2,[2,4]] = _Nx
#         Ny[1,[1,3]] = _Ny
#         Ny[2,[2,4]] = _Ny

#         dNdx[1,[1,3]] = _dNdx
#         dNdx[2,[2,4]] = _dNdx
#         dNdy[1,[1,3]] = _dNdy
#         dNdy[2,[2,4]] = _dNdy


#         NxNy = Nx'*(Ny*any)

#         NNx[1,1] = NxNy[1]
#         NNx[2,2] = NxNy[2]
#         NNx[1,3] = NxNy[3]
#         NNx[2,4] = NxNy[4]

#         dNdxNy = dNdx'*(Ny*any)
#         NxdNdy = Nx'*(dNdy*any)

#         BBx[1,1] = dNdxNy[1]
#         BBx[3,1] = NxdNdy[1]
#         BBx[2,2] = NxdNdy[2]
#         BBx[3,2] = dNdxNy[2]
#         BBx[1,3] = dNdxNy[3]
#         BBx[3,3] = NxdNdy[3]
#         BBx[2,4] = NxdNdy[4]
#         BBx[3,4] = dNdxNy[4]

#         NyNx = Ny'*(Nx*anx)

#         NNy[1,1] = NyNx[1]
#         NNy[2,2] = NyNx[2]
#         NNy[1,3] = NyNx[3]
#         NNy[2,4] = NyNx[4]

#         dNdyNx = dNdy'*(Nx*anx)
#         NydNdx = Ny'*(dNdx*anx)

#         BBy[1,1] = NydNdx[1]
#         BBy[3,1] = dNdyNx[1]
#         BBy[2,2] = dNdyNx[2]
#         BBy[3,2] = NydNdx[2]
#         BBy[1,3] = NydNdx[3]
#         BBy[3,3] = dNdyNx[3]
#         BBy[2,4] = dNdyNx[4]
#         BBy[3,4] = NydNdx[4]

#         ε = zeros(3)
#         for m = 1:nModes(U) # Meh, måste man göra såhär :(
#             m_dNdxNy = dNdx'*(Ny*ay[:,m])
#             m_NxdNdy = Nx'*(dNdy*ay[:,m])

#             m_BBx[1,1] = m_dNdxNy[1]
#             m_BBx[3,1] = m_NxdNdy[1]
#             m_BBx[2,2] = m_NxdNdy[2]
#             m_BBx[3,2] = m_dNdxNy[2]
#             m_BBx[1,3] = m_dNdxNy[3]
#             m_BBx[3,3] = m_NxdNdy[3]
#             m_BBx[2,4] = m_NxdNdy[4]
#             m_BBx[3,4] = m_dNdxNy[4]

#             ε += m_BBx*ax[:,m]
#         end

#         ε += BBx*anx # elelr BBy*ay, blir samma
#         σ = D*ε


#         dΩ = U.fev.detJdV[q_point]

#         gx = BBx' * σ * dΩ
#         gy = BBy' * σ * dΩ

#         g += [gx; gy] # Här får man typ assemblera då istället om man har en unstructured mesh
#     end

#     return g

# end


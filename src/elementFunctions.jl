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
#Tgrad = ForwardDiff.GradientNumber{8, Float64, NTuple{8, Float64}}
Tgrad = ForwardDiff.GradientNumber{12, Float64, Vector{Float64}}
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
        evaluate_at_gauss_point!(U.components[1].fev,[ξ_x],ex_x,Nx,dNdx) # This is not really needed yet, but for generalizing in the future
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

# function intfUλ{T}(an::Vector{T},fev::JuAFEM.FEValues)

#     gUλ = zeros(T,6)
#     gU = zeros(T,4)
#     gλ = zeros(T,2)

#     ax = an[1:4] # Element displacements
#     λ = an[5:6] # Lagrange multipliers

#     for (q_point, (ξ,w)) in enumerate(zip(fev.quad_rule.points,fev.quad_rule.weights))
#         Nx = JuAFEM.shape_value(fev,q_point)
#         dx = JuAFEM.detJdV(fev,q_point)

#         Ux = Nx[1] * ax[1] + Nx[2] * ax[3] # First component
#         Vx = Nx[1] * ax[2] + Nx[2] * ax[4] # Second component

#         gU[1] = λ[1] * Nx[1] * Ux * dx
#         gU[2] = λ[2] * Nx[1] * Vx * dx
#         gU[3] = λ[1] * Nx[2] * Ux * dx
#         gU[4] = λ[2] * Nx[2] * Vx * dx

#         gλ[1] = 0.5 * Ux^2 * dx #-1 # Ska inte bara här ju, din noob
#         gλ[2] = 0.5 * Vx^2 * dx #-1

#         gUλ += [gU; gλ] # * dx might be faster
#     end

#     return gUλ
# end

function intf_AmplitudeFormulation{T}(an::Vector{T},a::Matrix,x::Matrix,U::PGDFunction,D::Matrix,b::Vector=zeros(2))

    α_un = an[1]
    α_vn = an[2]
    ax_n = an[3:6] # Not general
    ay_n = an[7:10]
    λ = an[11:12] # λ_u and λ_v

    α_u = a[1,:] # This will be a Matrix in 0.4 and Vector on 0.5 (probably wont cause trouble though)
    α_v = a[2,:]
    ax = a[3:6,:]
    ay = a[7:10,:]

    buff_coll = get_buffer(buff_colls, T)

    # g = buff_coll.g # Not general
    g = zeros(T,12)

    Nx = buff_coll.Nx # Shape functions
    Ny = buff_coll.Ny

    dNdx = buff_coll.dNdx # Derivatives
    dNdy = buff_coll.dNdy

    N_x = buff_coll.NNx# N matrix for force vector
    N_y = buff_coll.NNy
    N_α = zeros(T,2,2)

    B_x = buff_coll.BBx  # B matrix for stiffness
    B_y = buff_coll.BBy
    B_α = zeros(T,3,2)

    # TODO: Include the two α values in the strain and testfunction
    for (q_point, (ξ,w)) in enumerate(zip(U.fev.quad_rule.points,U.fev.quad_rule.weights))
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        # Update values
        ex_x = [U.components[1].mesh.x[1] U.components[1].mesh.x[2]] # only for equidistant mesh
        ex_y = [U.components[2].mesh.x[1] U.components[2].mesh.x[2]]
        evaluate_at_gauss_point!(U.components[1].fev,[ξ_x],ex_x,Nx,dNdx) # This is not really needed yet, but for generalizing in the future
        evaluate_at_gauss_point!(U.components[2].fev,[ξ_y],ex_y,Ny,dNdy)

        # N_x[1,1] = Nx[1] * Ny[1] * ay_n[1] + Nx[1] * Ny[2] * ay_n[3]
        # N_x[2,2] = Nx[1] * Ny[1] * ay_n[2] + Nx[1] * Ny[2] * ay_n[4]
        # N_x[1,3] = Nx[2] * Ny[1] * ay_n[1] + Nx[2] * Ny[2] * ay_n[3]
        # N_x[2,4] = Nx[2] * Ny[1] * ay_n[2] + Nx[2] * Ny[2] * ay_n[4]

        N_x[1,1] = α_un * Nx[1] * (Ny[1] * ay[1] + Ny[2] * ay[3])
        N_x[2,2] = α_vn * Nx[1] * (Ny[1] * ay[2] + Ny[2] * ay[4])
        N_x[1,3] = α_un * Nx[2] * (Ny[1] * ay[1] + Ny[2] * ay[3])
        N_x[2,4] = α_vn * Nx[2] * (Ny[1] * ay[2] + Ny[2] * ay[4])

        # N_y[1,1] = Nx[1] * Ny[1] * ax_n[1] + Nx[2] * Ny[1] * ax_n[3]
        # N_y[2,2] = Nx[1] * Ny[1] * ax_n[2] + Nx[2] * Ny[1] * ax_n[4]
        # N_y[1,3] = Nx[1] * Ny[2] * ax_n[1] + Nx[2] * Ny[2] * ax_n[3]
        # N_y[2,4] = Nx[1] * Ny[2] * ax_n[2] + Nx[2] * Ny[2] * ax_n[4]

        N_y[1,1] = α_un * Ny[1] * (Nx[1] * ax[1] + Nx[2] * ax[3])
        N_y[2,2] = α_vn * Ny[1] * (Nx[1] * ax[2] + Nx[2] * ax[4])
        N_y[1,3] = α_un * Ny[2] * (Nx[1] * ax[1] + Nx[2] * ax[3])
        N_y[2,4] = α_vn * Ny[2] * (Nx[1] * ax[2] + Nx[2] * ax[4])

        N_α[1,1] = (Nx[1] * ax[1] + Nx[2] * ax[3]) * (Ny[1] * ay[1] + Ny[2] * ay[3])
        N_α[2,2] = (Nx[1] * ax[2] + Nx[2] * ax[4]) * (Ny[1] * ay[2] + Ny[2] * ay[4])

        # B_x[1,1] = dNdx[1] * Ny[1] * ay_n[1] + dNdx[1] * Ny[2] * ay_n[3]
        # B_x[3,1] = Nx[1] * dNdy[1] * ay_n[1] + Nx[1] * dNdy[2] * ay_n[3]
        # B_x[2,2] = Nx[1] * dNdy[1] * ay_n[2] + Nx[1] * dNdy[2] * ay_n[4]
        # B_x[3,2] = dNdx[1] * Ny[1] * ay_n[2] + dNdx[1] * Ny[2] * ay_n[4]
        # B_x[1,3] = dNdx[2] * Ny[1] * ay_n[1] + dNdx[2] * Ny[2] * ay_n[3]
        # B_x[3,3] = Nx[2] * dNdy[1] * ay_n[1] + Nx[2] * dNdy[2] * ay_n[3]
        # B_x[2,4] = Nx[2] * dNdy[1] * ay_n[2] + Nx[2] * dNdy[2] * ay_n[4]
        # B_x[3,4] = dNdx[2] * Ny[1] * ay_n[2] + dNdx[2] * Ny[2] * ay_n[4]

        B_x[1,1] = α_un * dNdx[1] * (Ny[1] * ay[1] + Ny[2] * ay[3])
        B_x[3,1] = α_un * Nx[1] * (ay[1] * dNdy[1] + ay[3] * dNdy[2])
        B_x[2,2] = α_vn * Nx[1] * (ay[2] * dNdy[1] + ay[4] * dNdy[2])
        B_x[3,2] = α_vn * dNdx[1] * (Ny[1] * ay[2] + Ny[2] * ay[4])
        B_x[1,3] = α_un * dNdx[2] * (Ny[1] * ay[1] + Ny[2] * ay[3])
        B_x[3,3] = α_un * Nx[2] * (ay[1] * dNdy[1] + ay[3] * dNdy[2])
        B_x[2,4] = α_vn * Nx[2] * (ay[2] * dNdy[1] + ay[4] * dNdy[2])
        B_x[3,4] = α_vn * dNdx[2] * (Ny[1] * ay[2] + Ny[2] * ay[4])

        # B_y[1,1] = dNdx[1] * Ny[1] * ax_n[1] + dNdx[2] * Ny[1] * ax_n[3]
        # B_y[3,1] = Nx[1] * dNdy[1] * ax_n[1] + Nx[2] * dNdy[1] * ax_n[3]
        # B_y[2,2] = Nx[1] * dNdy[1] * ax_n[2] + Nx[2] * dNdy[1] * ax_n[4]
        # B_y[3,2] = dNdx[1] * Ny[1] * ax_n[2] + dNdx[2] * Ny[1] * ax_n[4]
        # B_y[1,3] = dNdx[1] * Ny[2] * ax_n[1] + dNdx[2] * Ny[2] * ax_n[3]
        # B_y[3,3] = Nx[1] * dNdy[2] * ax_n[1] + Nx[2] * dNdy[2] * ax_n[3]
        # B_y[2,4] = Nx[1] * dNdy[2] * ax_n[2] + Nx[2] * dNdy[2] * ax_n[4]
        # B_y[3,4] = dNdx[1] * Ny[2] * ax_n[2] + dNdx[2] * Ny[2] * ax_n[4]

        B_y[1,1] = α_un * Ny[1] * (dNdx[1] * ax[1] + dNdx[2] * ax[3])
        B_y[3,1] = α_un * dNdy[1] * (Nx[1] * ax[1] + Nx[2] * ax[3])
        B_y[2,2] = α_vn * dNdy[1] * (Nx[1] * ax[2] + Nx[2] * ax[4])
        B_y[3,2] = α_vn * Ny[1] * (ax[2] * dNdx[1] + ax[4] * dNdx[2])
        B_y[1,3] = α_un * Ny[2] * (ax[1] * dNdx[1] + ax[3] * dNdx[2])
        B_y[3,3] = α_un * dNdy[2] * (Nx[1] * ax[1] + Nx[2] * ax[3])
        B_y[2,4] = α_vn * dNdy[2] * (Nx[1] * ax[2] + Nx[2] * ax[4])
        B_y[3,4] = α_vn * Ny[2] * (ax[2]*dNdx[1] + ax[4]*dNdx[2])

        # TODO: Clean up this mess, its probably just a matrix multiplication or somethign
        B_α[1,1] = (Ny[1] * ay[1] + Ny[2] * ay[3]) * (ax[1] * dNdx[1] + ax[3] * dNdx[2])
        B_α[3,1] = (Nx[1] * ax[1] + Nx[2] * ax[3]) * (ay[1] * dNdy[1] + ay[3] * dNdy[2])
        B_α[2,2] = (Nx[1] * ax[2] + Nx[2] * ax[4]) * (ay[2] * dNdy[1] + ay[4] * dNdy[2])
        B_α[3,2] = (Ny[1] * ay[2] + Ny[2] * ay[4]) * (ax[2] * dNdx[1] + ax[4] * dNdx[2])



        ε = buff_coll.ε
        fill!(ε, 0.0)
        ε_m = buff_coll.ε_m

        for m = 1:nModes(U)
            # ε_m[1] = dNdx[1] * Ny[1] * ax[1,m] * ay[1,m] + dNdx[2] * Ny[1] * ax[3,m] * ay[1,m] +
            #          dNdx[1] * Ny[2] * ax[1,m] * ay[3,m] + dNdx[2] * Ny[2] * ax[3,m] * ay[3,m]

            # ε_m[2] = Nx[1] * dNdy[1] * ax[2,m] * ay[2,m] + Nx[1] * dNdy[2] * ax[2,m] * ay[4,m] +
            #          Nx[2] * dNdy[1] * ax[4,m] * ay[2,m] + Nx[2] * dNdy[2] * ax[4,m] * ay[4,m]

            # ε_m[3] = Nx[1] * dNdy[1] * ax[1,m] * ay[1,m] + Nx[1] * dNdy[2] * ax[1,m] * ay[3,m] +
            #          Nx[2] * dNdy[1] * ax[3,m] * ay[1,m] + Nx[2] * dNdy[2] * ax[3,m] * ay[3,m] +
            #          dNdx[1] * Ny[1] * ax[2,m] * ay[2,m] + dNdx[2] * Ny[1] * ax[4,m] * ay[2,m] +
            #          dNdx[1] * Ny[2] * ax[2,m] * ay[4,m] + dNdx[2] * Ny[2] * ax[4,m] * ay[4,m]

            ε_m[1] = α_u[m] * (Ny[1] * ay[1,m] + Ny[2] * ay[3,m]) * (ax[1,m] * dNdx[1] + ax[3,m] * dNdx[2])

            ε_m[2] = α_v[m] * (Nx[1] * ax[2,m] + Nx[2] * ax[4,m]) * (ay[2,m] * dNdy[1] + ay[4,m] * dNdy[2])

            ε_m[3] = Nx[1] * α_un * ax[1,m] * ay[1,m] * dNdy[1] + Nx[1] * α_un * ax[1,m] * ay[3,m] * dNdy[2] +
                     Nx[2] * α_un * ax[3,m] * ay[1,m] * dNdy[1] + Nx[2] * α_un * ax[3,m] * ay[3,m] * dNdy[2] +
                     Ny[1] * α_vn * ax[2,m] * ay[2,m] * dNdx[1] + Ny[1] * α_vn * ax[4,m] * ay[2,m] * dNdx[2] +
                     Ny[2] * α_vn * ax[2,m] * ay[4,m] * dNdx[1] + Ny[2] * α_vn * ax[4,m] * ay[4,m] * dNdx[2]


            ε += ε_m
        end

        ε_n = B_x * ax_n # Strain for last mode (can also be calculated as B_y*ay_n or B_α * [α_un; α_vn])
        ε += ε_n

        σ = D*ε

        # Modeshape of displacement components of mode n (fix better variable name)
        UU_n = [(Nx[1] * ax_n[1] + Nx[2] * ax_n[3]) * (Ny[1] * ay_n[1] + Ny[2] * ay_n[3]),
                (Nx[1] * ax_n[2] + Nx[2] * ax_n[4]) * (Ny[1] * ay_n[2] + Ny[2] * ay_n[4])]

        dΩ = U.fev.detJdV[q_point]

        gα = (B_α' * σ - N_α' * b) * dΩ # α equation

        gx = (B_x' * σ - N_x' * b) * dΩ # x-mode equation
        gy = (B_y' * σ - N_y' * b) * dΩ # y-mode equation

        # From lagrange multiplier, ∫|u|²dΩ = 1 and ∫|v|²dΩ = 1
        # lol vad fult
        gλx = [ λ[1] * Nx[1] * (Ny[1]*ay_n[1] + Ny[2]*ay_n[3]) * UU_n[1],
                λ[2] * Nx[1] * (Ny[1]*ay_n[2] + Ny[2]*ay_n[4]) * UU_n[2],
                λ[1] * Nx[2] * (Ny[1]*ay_n[1] + Ny[2]*ay_n[3]) * UU_n[1],
                λ[2] * Nx[2] * (Ny[1]*ay_n[2] + Ny[2]*ay_n[4]) * UU_n[2]] * dΩ

        gλy = [ λ[1] * Ny[1] * (Nx[1]*ax_n[1] + Nx[2]*ax_n[3]) * UU_n[1],
                λ[2] * Ny[1] * (Nx[1]*ax_n[2] + Nx[2]*ax_n[4]) * UU_n[2],
                λ[1] * Ny[2] * (Nx[1]*ax_n[1] + Nx[2]*ax_n[3]) * UU_n[1],
                λ[2] * Ny[2] * (Nx[1]*ax_n[2] + Nx[2]*ax_n[4]) * UU_n[2]] * dΩ

        gλuv = (0.5 * UU_n.^2) * dΩ # Dont forget the -0.5 on the global residual


        g += [gα;
              gx + gλx;
              gy + gλy;
              gλuv] # Maybe scale by dΩ here instead of every component
    end

    return g
end
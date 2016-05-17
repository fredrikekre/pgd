###################
# Shape functions #
###################

function getN{dimcomp,T,functionspace}(U::PGDComponent{dimcomp,T,functionspace},::Val{3})

    ex = U.mesh.ex[1]
    reinit!(U.fev,ex)

    nqp = length(points(U.fev.quad_rule))

    N = Vector{Vec{U.totaldim,T}}[]
    dN = Vector{Tensor{2,U.totaldim,T}}[]

    for qp in 1:nqp
        Nqp = Vec{U.totaldim,T}[]
        dNqp = Tensor{2,U.totaldim,T}[]
        for dof in 1:U.mesh.nElDofs
            counterN = ceil(Int,dof/U.totaldim)
            counterDim = mod(dof,U.totaldim); if counterDim == 0; counterDim = U.totaldim; end

            # Shape functions
            thisN = zeros(T,U.totaldim)
            thisN[counterDim] = U.fev.N[qp][counterN]
            N_ = Vec{U.totaldim,T}((thisN...))
            push!(Nqp, N_)

            # Derivatives
            thisdN = zeros(T,U.totaldim)
            thisdN[counterDim] = U.fev.dNdx[qp][counterN][1]
            if U.compdim == 1
                thisdNt = [thisdN; thisN; thisN]
            elseif U.compdim == 2
                thisdNt = [thisN; thisdN; thisN]
            elseif U.compdim == 3
                thisdNt = [thisN; thisN; thisdN]
            end
            dN_ = Tensor{2,U.totaldim}((thisdNt...))
            push!(dNqp,dN_)
        end
        push!(N,Nqp)
        push!(dN,dNqp)
    end
    return N, dN
end


###################
# Pre integration #
###################

function integrate_one_dimension{T,dimcomp,functionspace}(a::Vector{Vector{T}},
                                 U::PGDComponent{dimcomp,T,functionspace},
                                 E,
                                 f)
    N, dN = getN(U, Val{3}())

    Ex = [zero(E[i]) for i in 1:length(E)]
    fx = zero(f)

    for el in 1:U.mesh.nEl
        m = U.mesh.edof[:,el]
        for qp in 1:length(points(U.fev.quad_rule))
            dV = detJdV(U.fev,qp)

            Uh = zero(Tensor{1,U.totaldim})
            dUh = zero(Tensor{2,U.totaldim})
            for dof in 1:U.mesh.nElDofs
                dUh += dN[qp][dof] * a[end][m[dof]]
                Uh += N[qp][dof] * a[end][m[dof]]
            end

            # Mode n
            Ex[end] += hadamard(dUh,E[end],dUh) * dV

            fx += hadamard(Uh,f) * dV

            # Modes 1:(n-1)
            number_modes = length(a)-1
            for mode in 1:number_modes
                dUhmode = zero(Tensor{2,U.totaldim})
                for dof in 1:U.mesh.nElDofs
                    dUhmode += dN[qp][dof] * a[mode][m[dof]]
                end
                Ex[mode] += hadamard(dUh,E[mode],dUhmode) * dV
            end

        end
    end

    return Ex, fx
end

#####################
# Element functions #
#####################
function ele_3D_1D{T}(fe::Vector{T},Ke::Matrix{T},m::Vector{Int},
                                  U1::PGDComponent,a1::Vector{Vector{T}},N,dN,
                                  E,f)


    nElDofs = U1.mesh.nElDofs
    fill!(fe,zero(T))
    fill!(Ke,zero(T))

    nqp = length(points(U1.fev.quad_rule))

    number_of_modes = length(a1)-1

    for qp in 1:nqp
        dV = detJdV(U1.fev,qp)

        for dof1 in 1:nElDofs

            # Body force
            fe[dof1] += (N[qp][dof1] ⋅ f) * dV

            # Previous modes
            for mode in 1:number_of_modes
                Un = zero(Tensor{1,U1.totaldim})
                dUn = zero(Tensor{2,U1.totaldim})
                for dof3 in 1:nElDofs
                    dUn += dN[qp][dof3] * a1[mode][m[dof3]]
                    Un += N[qp][dof3] * a1[mode][m[dof3]]
                end
                fe[dof1] -= (dcontract(dN[qp][dof1], dcontract(E[mode], dUn))) * dV
            end

            # Stiffness matrix
            for dof2 in 1:nElDofs
                Ke[dof1,dof2] += (dcontract(dN[qp][dof1], dcontract(E[end], dN[qp][dof2]))) * dV
            end

        end
    end

    return fe, Ke
end

function integrate_one_dimension_element_level{T,dimcomp,functionspace}(
                       a::Vector{Vector{T}},U::PGDComponent{dimcomp,T,functionspace},
                       N,dN,
                       E,f)

    Ex = [zero(E[i]) for i in 1:length(E)]
    fx = zero(f)

    for qp in 1:length(points(U.fev.quad_rule))
        dV = detJdV(U.fev,qp)
        Uh = zero(Tensor{1,U.totaldim})
        dUh = zero(Tensor{2,U.totaldim})
        for dof in 1:U.mesh.nElDofs
            dUh += dN[qp][dof] * a[end][dof]
            Uh += N[qp][dof] * a[end][dof]
        end

        # Mode n
        Ex[end] += hadamard(dUh,E[end],dUh) * dV

        fx += hadamard(Uh,f) * dV

        # Modes 1:(n-1)
        number_modes = length(a)-1
        for mode in 1:number_modes
            dUhmode = zero(Tensor{2,U.totaldim})
            for dof in 1:length(a)
                dUhmode += dN[qp][dof] * a[mode][dof]
            end
            Ex[mode] += hadamard(dUh,E[mode],dUhmode) * dV
        end

    end

    return Ex, fx
end

function ele_3D_3D{T}(fe::Vector{T},Ke::Matrix{T},
                      U1::PGDComponent,a1::Vector{Vector{T}},N1,dN1,
                      U2::PGDComponent,a2::Vector{Vector{T}},N2,dN2,
                      U3::PGDComponent,a3::Vector{Vector{T}},N3,dN3,
                      E,f)


    nElDofs = U1.mesh.nElDofs
    fill!(fe,zero(T))
    fill!(Ke,zero(T))

    # Pre integrate on element level
    Evec = Tensor{4,3}[E for i in 1:length(a1)]
    E3, f3 = integrate_one_dimension_element_level(a3,U3,N3,dN3,Evec,f)
    E32, f32 = integrate_one_dimension_element_level(a2,U2,N2,dN2,E3,f3)

    # Integrate this element
    nqp = length(points(U1.fev.quad_rule))

    number_of_modes = length(a1)-1

    for qp in 1:nqp
        dV = detJdV(U1.fev,qp)

        for dof1 in 1:nElDofs

            # Body force
            fe[dof1] += (N1[qp][dof1] ⋅ f32) * dV

            # Previous modes
            for mode in 1:number_of_modes
                Un = zero(Tensor{1,U1.totaldim})
                dUn = zero(Tensor{2,U1.totaldim})
                for dof3 in 1:nElDofs
                    dUn += dN1[qp][dof3] * a1[mode][dof3]
                    Un += N1[qp][dof3] * a1[mode][dof3]
                end
                fe[dof1] -= (dcontract(dN1[qp][dof1], dcontract(E32[mode], dUn))) * dV
            end

            # Stiffness matrix
            for dof2 in 1:nElDofs
                Ke[dof1,dof2] += (dcontract(dN1[qp][dof1], dcontract(E32[end], dN1[qp][dof2]))) * dV
            end

        end
    end

    return fe, Ke
end

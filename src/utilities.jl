"""
    createLink() ->
    Sets up the connection between the governing mesh's integration points
    and the components meshes

"""

# Maybe calculate corresponding ξ in this function too
function create_link(gFunc::PGDFunction)
    gngp = length(JuAFEM.get_quadrule(gFunc.fev).weights) # n gausspoints in gov. mesh
    gnEl = gFunc.mesh.nEl # Number of elements in gov. mesh

    link = [zeros(Int,gngp,gnEl) for i in 1:gFunc.nComp]

    #edof = zeros(8,gnEl)

    for i = 1:gnEl
        x = [gFunc.mesh.ex[:,i] gFunc.mesh.ey[:,i]]' # Welp, not so general

        for j = 1:gngp
            xyz = x*gFunc.fev.N[j]

            for k = 1:length(xyz)
                exComp = gFunc.components[k].mesh.ex
                index = 0
                for l = 1:size(exComp,2)
                    if xyz[k]>=exComp[1,l] && xyz[k]<=exComp[end,l] # Better way to find this?
                        index = l
                        break
                    end
                end
                if index == 0; error("Did not find element"); end
                link[k][j,i] = index
            end
        end
        # Very bad and not general at all
        #edof[i,:]
    end

    # Add error if some component elements are not in the link matrix (underintegration-ish)

    return link
end

function create_edof(gFunc::PGDFunction,nNodeDofs::Int)
    gnEl = gFunc.mesh.nEl # Number of elements in gov. mesh

    if nNodeDofs == 1
        edof = zeros(Int64,4,gnEl)
        edof[1:2,:] = hcat([gFunc.components[1].mesh.edof for i in 1:gFunc.components[2].mesh.nEl]...)
        edof[3:4,:] = reshape(vcat([gFunc.components[2].mesh.edof for i in 1:gFunc.components[1].mesh.nEl]...),(2,gnEl))
        edof[3:4,:] += gFunc.components[1].mesh.nDofs
    elseif nNodeDofs == 2
        edof = zeros(Int64,8,gnEl)
        edof[1:4,:] = hcat([gFunc.components[1].mesh.edof for i in 1:gFunc.components[2].mesh.nEl]...)
        edof[5:8,:] = reshape(vcat([gFunc.components[2].mesh.edof for i in 1:gFunc.components[1].mesh.nEl]...),(4,gnEl))
        edof[5:8,:] += gFunc.components[1].mesh.nDofs
    end
    return edof
end

number_of_dofs(x::Array{Int,2}) = maximum(x)

function evaluate_at_gauss_point!(fe_v::JuAFEM.FEValues, ξ, x, N::Vector, dNdx)
    # Evaluates N and dNdx at a speciefied Gauss-point
    n_basefuncs = JuAFEM.n_basefunctions(fe_v.function_space)

    dNdξ = [zero(Vec{1,Float64}) for i in 1:n_basefuncs]
    JuAFEM.value!(fe_v.function_space,N,ξ)
    JuAFEM.derivative!(fe_v.function_space,dNdξ,ξ)

    J = zero(Tensor{2,1})
    for i in 1:n_basefuncs
        J += dNdξ[i] ⊗ x[i]
    end

    Jinv = inv(J)

    for i in 1:n_basefuncs
        dNdx[i] = Jinv ⋅ dNdξ[i]
    end

    return N, dNdx
end

function norm_of_mode(U::PGDFunction, a::Array)
    # Calculate the norm of mode `a` for PGD function `U`
    # if `U` is 2D and 2 dimensionally mode `a`

    Ux_dof = 1:2:(U.components[1].mesh.nDofs-1)
    Vx_dof = 2:2:(U.components[1].mesh.nDofs)
    Uy_dof = (U.components[1].mesh.nDofs+1):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs-1)
    Vy_dof = (U.components[1].mesh.nDofs+2):2:(U.components[1].mesh.nDofs+U.components[1].mesh.nDofs)

    Ux = a[Ux_dof,:]
    Vx = a[Vx_dof,:]

    Uy = a[Uy_dof,:]
    Vy = a[Vy_dof,:]

    u = Uy*Ux'; normu = norm(u)
    v = Vy*Vx'; normv = norm(v)

    return normu, normv
end

# Convert Voigt strain to strain tensor
function εv_to_εt{T}(ε::Vector{T})

    εt_val = (ε[1], ε[3]/2, 0.0, ε[2], 0.0, 0.0)

    return SymmetricTensor{2,3,T}(εt_val)

end

# Hadamard product
function hadamard(x::Vector,y::Vector)
    size(x) == size(y) || throw(ArgumentError("x and y need to have the same length."))
    return x.*y
end

const ∘ = hadamard

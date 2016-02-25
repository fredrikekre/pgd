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

function create_edof(gFunc::PGDFunction)
    gnEl = gFunc.mesh.nEl # Number of elements in gov. mesh
    edof = zeros(Int64,8,gnEl)
    edof[1:4,:] = hcat([gFunc.components[1].mesh.edof for i in 1:gFunc.components[2].mesh.nEl]...)
    edof[5:8,:] = reshape(vcat([gFunc.components[2].mesh.edof for i in 1:gFunc.components[1].mesh.nEl]...),(4,gnEl))
    edof[5:8,:] += gFunc.components[1].mesh.nDofs
    return edof
end

function evaluate_at_gauss_point!(fe_v::JuAFEM.FEValues, ξ::Vector, x::Matrix, N::Vector, dNdx::Matrix)
    # Evaluates N and dNdx at a speciefied Gauss-point

    dNdξ = zeros(dNdx)
    JuAFEM.value!(fe_v.function_space,N,ξ)
    JuAFEM.derivative!(fe_v.function_space,dNdξ,ξ)
    J = dNdξ * x'
    Jinv = JuAFEM.inv_spec(J)

    dNdx[:,:] = Jinv*dNdξ
    return N, dNdx
end
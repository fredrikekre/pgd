function calc_globres{T}(an::Vector{T},a::Matrix,U::PGDFunction,D::Matrix,edof::Matrix,λ_dofs::Vector,b::Vector,free)
    # Calculate global residual, g_glob
    ndofs = maximum(edof)+length(λ_dofs)
    g_glob = zeros(T,ndofs)

    for i = 1:U.mesh.nEl
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        JuAFEM.reinit!(U.fev,x)
        m = edof[:,i]
        ge = intf(an[m],a[m,:],x,U,D,b)

        g_glob[m] += ge
    end

    #########################################
    # Added for Lagrange multiplier for Ux
    for i =1:U.components[1].mesh.nEl
        x = U.components[1].mesh.ex[:,i]

        JuAFEM.reinit!(U.components[1].fev,x')
        edofcomponent = U.components[1].mesh.edof # Dangerous but works for first component (Need to add some connection between compnent mesh and global dofs)
        m = edofcomponent[:,i]
        mλ = [m[1:4]; λ_dofs] # Not very nice maybe, only works for first component

        gUλe = intfUλ(an[mλ],U.components[1].fev)
        g_glob[mλ] += gUλe
    end
    g_glob[λ_dofs] -= 0.5 # Since the norm of those modes should be 1
    #########################################

    return g_glob[free]
end

function calc_globK{T}(an::Vector{T},a::Matrix,U::PGDFunction,D::Matrix,edof::Matrix,λ_dofs,b::Vector, free)
    # Calculate global tangent stiffness matrix, K

    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:U.mesh.nEl
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        JuAFEM.reinit!(U.fev,x)
        m = edof[:,i]

        intf_closure(an) = intf(an,a[m,:],x,U,D,b)

        kefunc = ForwardDiff.jacobian(intf_closure, cache=cache)
        Ke = kefunc(an[m])

        JuAFEM.assemble(m,_K,Ke)
    end

    ###########################################
    # Added for Lagrange multiplier for Ux
    for i =1:U.components[1].mesh.nEl
        x = U.components[1].mesh.ex[:,i]
        JuAFEM.reinit!(U.components[1].fev,x')

        edofcomponent = U.components[1].mesh.edof # Dangerous but works for first component (Need to add some connection between compnent mesh and global dofs)
        m = edofcomponent[:,i]
        mλ = [m[1:4]; λ_dofs] # Not very nice maybe, only works for first component

        intfUλ_closure(an) = intfUλ(an,U.components[1].fev)

        kefuncUλ = ForwardDiff.jacobian(intfUλ_closure, cache=cache)
        KeUλ = kefuncUλ(an[mλ])

        JuAFEM.assemble(mλ,_K,KeUλ)
    end
    ###########################################

    K = JuAFEM.end_assemble(_K)

    return K[free, free]
end
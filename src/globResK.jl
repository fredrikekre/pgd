function calc_globres{T}(an::Vector{T},a::Matrix,U::PGDFunction,D::Matrix,edof::Matrix,b::Vector,free)
    # Calculate global residual, g_glob
    ndofs = maximum(edof)
    g_glob = zeros(T,ndofs)

    for i = 1:U.mesh.nEl
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        JuAFEM.reinit!(U.fev,x)
        m = edof[:,i]
        ge = intf_AmplitudeFormulation(an[m],a[m,:],x,U,D,b)

        g_glob[m] += ge
    end

    # Not needed now that we integrate over Ω and have conditions on that
    # #########################################
    # # Added for Lagrange multiplier for Ux
    # for i =1:U.components[1].mesh.nEl
    #     x = U.components[1].mesh.ex[:,i]

    #     JuAFEM.reinit!(U.components[1].fev,x')
    #     edofcomponent = U.components[1].mesh.edof # Dangerous but works for first component (Need to add some connection between compnent mesh and global dofs)
    #     m = edofcomponent[:,i]
    #     mλ = [m[1:4]; λ_dofs[1:2]] # Not very nice, only works for first component

    #     gUλe = intfUλ(an[mλ],U.components[1].fev)
    #     g_glob[mλ] += gUλe
    # end
    # for i =1:U.components[2].mesh.nEl
    #     x = U.components[2].mesh.ex[:,i]

    #     JuAFEM.reinit!(U.components[2].fev,x')
    #     edofcomponent = U.components[2].mesh.edof # Dangerous but works for first component (Need to add some connection between compnent mesh and global dofs)
    #     m = edofcomponent[:,i] + U.components[1].mesh.nDofs
    #     mλ = [m[1:4]; λ_dofs[3:4]] # Not very nice, only works for first component

    #     gUλe = intfUλ(an[mλ],U.components[2].fev)
    #     g_glob[mλ] += gUλe
    # end
    # g_glob[λ_dofs] -= 1 # Since the norm of those modes should be 1
    # #########################################

    # Right hand side of λ-equation
    g_glob[end] -= 0.5
    g_glob[end-1] -= 0.5
    return g_glob[free]
end

function calc_globK{T}(an::Vector{T},a::Matrix,U::PGDFunction,D::Matrix,edof::Matrix,b::Vector, free)
    # Calculate global tangent stiffness matrix, K
    println("Inne i calc_globK")

    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    for i = 1:U.mesh.nEl
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        JuAFEM.reinit!(U.fev,x)
        m = edof[:,i]
        println(m)

        intf_AmplitudeFormulation_closure(an) = intf_AmplitudeFormulation(an,a[m,:],x,U,D,b)
        println("Done making the closure func.")

        kefunc = ForwardDiff.jacobian(intf_AmplitudeFormulation_closure, cache=cache)
        println("Done making kefunc")
        Ke = kefunc(an[m])
        println(Ke)

        JuAFEM.assemble(m,_K,Ke)
    end

    # Same reason as above
    # ###########################################
    # # Added for Lagrange multiplier for Ux
    # for i =1:U.components[1].mesh.nEl
    #     x = U.components[1].mesh.ex[:,i]
    #     JuAFEM.reinit!(U.components[1].fev,x')

    #     edofcomponent = U.components[1].mesh.edof # Dangerous but works for first component (Need to add some connection between compnent mesh and global dofs)
    #     m = edofcomponent[:,i]
    #     mλ = [m[1:4]; λ_dofs[1:2]] # Not very nice maybe, only works for first component

    #     intfUλ_closure(an) = intfUλ(an,U.components[1].fev)

    #     kefuncUλ = ForwardDiff.jacobian(intfUλ_closure)#, cache=cache)
    #     KeUλ = kefuncUλ(an[mλ])

    #     JuAFEM.assemble(mλ,_K,KeUλ)
    # end
    # for i =1:U.components[2].mesh.nEl
    #     x = U.components[2].mesh.ex[:,i]
    #     JuAFEM.reinit!(U.components[2].fev,x')

    #     edofcomponent = U.components[2].mesh.edof # Dangerous but works for first component (Need to add some connection between compnent mesh and global dofs)
    #     m = edofcomponent[:,i] + U.components[1].mesh.nDofs
    #     mλ = [m[1:4]; λ_dofs[3:4]] # Not very nice maybe, only works for first component

    #     intfUλ_closure(an) = intfUλ(an,U.components[2].fev)

    #     kefuncUλ = ForwardDiff.jacobian(intfUλ_closure)#, cache=cache)
    #     KeUλ = kefuncUλ(an[mλ])

    #     JuAFEM.assemble(mλ,_K,KeUλ)
    # end
    # ###########################################

    K = JuAFEM.end_assemble(_K)
    println("Determinanten av K[free,free] = $(det(K[free,free]))")
    error()

    return K[free, free]
end
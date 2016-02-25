function calc_globres{T}(an::Vector{T},a::Matrix,U::PGDFunction,D::Matrix,edof::Matrix,b::Vector)
    # Calculate global residual, g_glob

    g_glob = zeros(T,maximum(edof))

    for i = 1:U.mesh.nEl
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        JuAFEM.reinit!(U.fev,x)
        m = edof[:,i]
        ge = intf(an[m],a[m,:],x,U,D,b)

        g_glob[m] += ge
    end
    return g_glob
end

function calc_globK{T}(an::Vector{T},a::Matrix,U::PGDFunction,D::Matrix,edof::Matrix,b::Vector)
    # Calculate global tangent stiffness matrix

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

    K = JuAFEM.end_assemble(_K)
    return K
end
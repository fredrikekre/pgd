function calc_globres{T}(an::Vector{T},a::Matrix,U::PGDFunction,D::Matrix,edof::Matrix,b::Vector,free)
    # Calculate global residual, g_glob
    ndofs = maximum(edof)
    g_glob = zeros(T,ndofs)

    # stiffel = [collect((50*15+16):(50*15+35));
    #            collect((50*16+16):(50*16+35));
    #            collect((50*17+16):(50*17+35));
    #            collect((50*18+16):(50*18+35));
    #            collect((50*19+16):(50*19+35));
    #            collect((50*20+16):(50*20+35));
    #            collect((50*21+16):(50*21+35));
    #            collect((50*22+16):(50*22+35));
    #            collect((50*23+16):(50*23+35));
    #            collect((50*24+16):(50*24+35));
    #            collect((50*25+16):(50*25+35));
    #            collect((50*26+16):(50*26+35));
    #            collect((50*27+16):(50*27+35));
    #            collect((50*28+16):(50*28+35));
    #            collect((50*29+16):(50*29+35));
    #            collect((50*30+16):(50*30+35));
    #            collect((50*31+16):(50*31+35));
    #            collect((50*32+16):(50*32+35));
    #            collect((50*33+16):(50*33+35));
    #            collect((50*34+16):(50*34+35));
    #            collect((50*35+16):(50*35+35))]

    elstiff = ones(U.mesh.nEl)
    # elstiff[stiffel] = 100


    for i = 1:U.mesh.nEl
        # println("Residual element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        JuAFEM.reinit!(U.fev,x)
        m = edof[:,i]
        ge = intf(an[m],a[m,:],x,U,D*elstiff[i],b)

        g_glob[m] += ge
    end

    return g_glob[free]
end

function calc_globK{T}(an::Vector{T},a::Matrix,U::PGDFunction,D::Matrix,edof::Matrix,b::Vector, free)
    # Calculate global tangent stiffness matrix, K

    _K = JuAFEM.start_assemble()
    cache = ForwardDiffCache()

    # stiffel = [collect((50*15+16):(50*15+35));
    #            collect((50*16+16):(50*16+35));
    #            collect((50*17+16):(50*17+35));
    #            collect((50*18+16):(50*18+35));
    #            collect((50*19+16):(50*19+35));
    #            collect((50*20+16):(50*20+35));
    #            collect((50*21+16):(50*21+35));
    #            collect((50*22+16):(50*22+35));
    #            collect((50*23+16):(50*23+35));
    #            collect((50*24+16):(50*24+35));
    #            collect((50*25+16):(50*25+35));
    #            collect((50*26+16):(50*26+35));
    #            collect((50*27+16):(50*27+35));
    #            collect((50*28+16):(50*28+35));
    #            collect((50*29+16):(50*29+35));
    #            collect((50*30+16):(50*30+35));
    #            collect((50*31+16):(50*31+35));
    #            collect((50*32+16):(50*32+35));
    #            collect((50*33+16):(50*33+35));
    #            collect((50*34+16):(50*34+35));
    #            collect((50*35+16):(50*35+35))]

    elstiff = ones(U.mesh.nEl)
    # elstiff[stiffel] = 100

    for i = 1:U.mesh.nEl
        # println("Stiffness element #$i")
        x = [U.mesh.ex[:,i] U.mesh.ey[:,i]]'
        JuAFEM.reinit!(U.fev,x)
        m = edof[:,i]

        intf_closure(an) = intf(an,a[m,:],x,U,D*elstiff[i],b)

        kefunc = ForwardDiff.jacobian(intf_closure, cache=cache)
        Ke = kefunc(an[m])

        JuAFEM.assemble(m,_K,Ke)
    end

    K = JuAFEM.end_assemble(_K)

    return K[free, free]
end
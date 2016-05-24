##########################
# 3D with 1D integration #
##########################
function mode_solver{T}(U1::PGDComponent, a1::Vector{Vector{T}},
                        U2::PGDComponent, a2::Vector{Vector{T}},
                        U3::PGDComponent, a3::Vector{Vector{T}},
                        E::Tensor{4,3},f,
                        bc::Vector{Int},::Val{1})
    # input check
    length(a1) == length(a2) == length(a3) || throw(ArgumentError("Something is wrong."))

    # Pre integrate other dimensions
    Evec = Tensor{4,3}[E for i in 1:length(a1)]
    E3, f3 = integrate_one_dimension(a3,U3,Evec,f)
    E32, f32 = integrate_one_dimension(a2,U2,E3,f3)

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(U1.mesh.edof)
    nElDofs = U1.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    N, dN = getN(U1, Val{3}()) # Shapefunctions (Also reinits!)

    for el in 1:U1.mesh.nEl
        m = U1.mesh.edof[:,el]
        fe, Ke = ele_3D_1D(fe,Ke,m,U1,a1,N,dN,E32,f32)
        assemble(m,_K,Ke)
        fext[m] += fe
    end

    K = end_assemble(_K)

    free = setdiff(1:nDofs,bc)
    a1new = zeros(nDofs)
    a1new[free] += K[free,free]\fext[free]

    return a1new
end

##########################
# 2D with 1D integration #
##########################
function mode_solver{T}(U1::PGDComponent, a1::Vector{Vector{T}},
                        U2::PGDComponent, a2::Vector{Vector{T}},
                        E::Tensor{4,2},f,
                        bc::Vector{Int},::Val{1})
    # input check
    length(a1) == length(a2) || throw(ArgumentError("Something is wrong."))

    # Pre integrate other dimensions
    Evec = typeof(E)[E for i in 1:length(a1)]
    E2, f2 = integrate_one_dimension(a2,U2,Evec,f)

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(U1.mesh.edof)
    nElDofs = U1.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    N, dN = getN(U1, Val{2}()) # Shapefunctions (Also reinits!)

    for el in 1:U1.mesh.nEl
        m = U1.mesh.edof[:,el]
        fe, Ke = ele_2D_1D(fe,Ke,m,U1,a1,N,dN,E2,f2)
        assemble(m,_K,Ke)
        fext[m] += fe
    end


    K = end_assemble(_K)

    free = setdiff(1:nDofs,bc)
    a1new = zeros(nDofs)
    a1new[free] += K[free,free]\fext[free]

    return a1new
end

##########################
# 3D with 3D integration #
##########################
function mode_solver{T}(U1::PGDComponent, a1::Vector{Vector{T}},
                        U2::PGDComponent, a2::Vector{Vector{T}},
                        U3::PGDComponent, a3::Vector{Vector{T}},
                        E,f,
                        bc::Vector{Int},::Val{3})
    # input check
    length(a1) == length(a2) == length(a3) || throw(ArgumentError("Something is wrong."))

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(U1.mesh.edof)
    nElDofs = U1.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    N1, dN1 = getN(U1, Val{3}()) # Shapefunctions (Also reinits!)
    N2, dN2 = getN(U2, Val{3}())
    N3, dN3 = getN(U3, Val{3}())

    for el1 in 1:U1.mesh.nEl, el2 in 1:U2.mesh.nEl, el3 in 1:U3.mesh.nEl
        m1 = U1.mesh.edof[:,el1]
        m2 = U2.mesh.edof[:,el2]
        m3 = U3.mesh.edof[:,el3]

        fe, Ke = ele_3D_3D(fe,Ke,U1,extract_eldofs(a1,m1),N1,dN1,
                                 U2,extract_eldofs(a2,m2),N2,dN2,
                                 U3,extract_eldofs(a3,m3),N3,dN3,
                                 E,f)
        assemble(m1,_K,Ke)
        fext[m1] += fe
    end

    K = end_assemble(_K)

    free = setdiff(1:nDofs,bc)
    a1new = zeros(nDofs)
    a1new[free] += K[free,free]\fext[free]

    return a1new
end

##########################
# 2D with 2D integration #
##########################
function mode_solver{T}(U1::PGDComponent, a1::Vector{Vector{T}},
                        U2::PGDComponent, a2::Vector{Vector{T}},
                        E,f,
                        bc::Vector{Int},::Val{2})
    # input check
    length(a1) == length(a2) || throw(ArgumentError("Something is wrong."))

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(U1.mesh.edof)
    nElDofs = U1.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    N1, dN1 = getN(U1, Val{2}()) # Shapefunctions (Also reinits!)
    N2, dN2 = getN(U2, Val{2}())

    for el1 in 1:U1.mesh.nEl, el2 in 1:U2.mesh.nEl
        m1 = U1.mesh.edof[:,el1]
        m2 = U2.mesh.edof[:,el2]

        fe, Ke = ele_2D_2D(fe,Ke,U1,extract_eldofs(a1,m1),N1,dN1,
                                 U2,extract_eldofs(a2,m2),N2,dN2,
                                 E,f)
        assemble(m1,_K,Ke)
        fext[m1] += fe
    end

    K = end_assemble(_K)

    free = setdiff(1:nDofs,bc)
    a1new = zeros(nDofs)
    a1new[free] += K[free,free]\fext[free]

    return a1new
end
###################################################
# Damage with displacement 3D with 3D integration #
###################################################
# x
function du_x_mode_solver{T}(Dx::PGDComponent, d_ax::Vector{Vector{T}},
                             Dy::PGDComponent, d_ay::Vector{Vector{T}},
                             Dz::PGDComponent, d_az::Vector{Vector{T}},
                             DN::d_shapefunctions{3,Float64},global_d_fe_values::FEValues,
                             dmp::damage_params,d_xbc::Vector{Int},
                             Ψ::Vector{Vector{Float64}})
    # input check
    length(d_ax) == length(d_ay) == length(d_az) || throw(ArgumentError("Something is wrong."))

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(Dx.mesh.edof)
    nElDofs = Dx.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    global_el = 0
    for el1 in 1:Dx.mesh.nEl, el2 in 1:Dy.mesh.nEl, el3 in 1:Dz.mesh.nEl
        global_el += 1

        d_m_x = Dx.mesh.edof[:,el1]
        d_m_y = Dy.mesh.edof[:,el2]
        d_m_z = Dz.mesh.edof[:,el3]

        fe, Ke = du_x_ele_3D_3D(fe,Ke,Dx,extract_eldofs(d_ax,d_m_x),
                                      Dy,extract_eldofs(d_ay,d_m_y),
                                      Dz,extract_eldofs(d_az,d_m_z),
                                      DN,global_d_fe_values,
                                      dmp,Ψ[global_el])
        assemble(d_m_x,_K,Ke)
        fext[d_m_x] += fe
    end

    K_x = end_assemble(_K)

    free = setdiff(1:nDofs,d_xbc)
    d_ax_new = zeros(nDofs)
    d_ax_new[free] += K_x[free,free]\fext[free]

    return d_ax_new
end

# y
function du_y_mode_solver{T}(Dx::PGDComponent, d_ax::Vector{Vector{T}},
                             Dy::PGDComponent, d_ay::Vector{Vector{T}},
                             Dz::PGDComponent, d_az::Vector{Vector{T}},
                             DN::d_shapefunctions{3,Float64},global_d_fe_values::FEValues,
                             dmp::damage_params,d_ybc::Vector{Int},
                             Ψ::Vector{Vector{Float64}})
    # input check
    length(d_ax) == length(d_ay) == length(d_az) || throw(ArgumentError("Something is wrong."))

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(Dy.mesh.edof)
    nElDofs = Dy.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    global_el = 0
    for el1 in 1:Dx.mesh.nEl, el2 in 1:Dy.mesh.nEl, el3 in 1:Dz.mesh.nEl
        global_el += 1

        d_m_x = Dx.mesh.edof[:,el1]
        d_m_y = Dy.mesh.edof[:,el2]
        d_m_z = Dz.mesh.edof[:,el3]

        fe, Ke = du_y_ele_3D_3D(fe,Ke,Dx,extract_eldofs(d_ax,d_m_x),
                                      Dy,extract_eldofs(d_ay,d_m_y),
                                      Dz,extract_eldofs(d_az,d_m_z),
                                      DN,global_d_fe_values,
                                      dmp,Ψ[global_el])
        assemble(d_m_y,_K,Ke)
        fext[d_m_y] += fe
    end

    K_y = end_assemble(_K)

    free = setdiff(1:nDofs,d_ybc)
    d_ay_new = zeros(nDofs)
    d_ay_new[free] += K_y[free,free]\fext[free]

    return d_ay_new
end

# z
function du_z_mode_solver{T}(Dx::PGDComponent, d_ax::Vector{Vector{T}},
                             Dy::PGDComponent, d_ay::Vector{Vector{T}},
                             Dz::PGDComponent, d_az::Vector{Vector{T}},
                             DN::d_shapefunctions{3,Float64},global_d_fe_values::FEValues,
                             dmp::damage_params,d_zbc::Vector{Int},
                             Ψ::Vector{Vector{Float64}})
    # input check
    length(d_ax) == length(d_ay) == length(d_az) || throw(ArgumentError("Something is wrong."))

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(Dz.mesh.edof)
    nElDofs = Dz.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    global_el = 0
    for el1 in 1:Dx.mesh.nEl, el2 in 1:Dy.mesh.nEl, el3 in 1:Dz.mesh.nEl
        global_el += 1

        d_m_x = Dx.mesh.edof[:,el1]
        d_m_y = Dy.mesh.edof[:,el2]
        d_m_z = Dz.mesh.edof[:,el3]

        fe, Ke = du_z_ele_3D_3D(fe,Ke,Dx,extract_eldofs(d_ax,d_m_x),
                                      Dy,extract_eldofs(d_ay,d_m_y),
                                      Dz,extract_eldofs(d_az,d_m_z),
                                      DN,global_d_fe_values,
                                      dmp,Ψ[global_el])
        assemble(d_m_z,_K,Ke)
        fext[d_m_z] += fe
    end

    K_z = end_assemble(_K)

    free = setdiff(1:nDofs,d_zbc)
    d_az_new = zeros(nDofs)
    d_az_new[free] += K_z[free,free]\fext[free]

    return d_az_new
end

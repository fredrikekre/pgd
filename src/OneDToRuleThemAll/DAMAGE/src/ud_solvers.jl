###################################################
# Displacement with damage 3D with 3D integration #
###################################################
# x
function ud_x_mode_solver{T}(Ux::PGDComponent, u_ax::Vector{Vector{T}},
                             Uy::PGDComponent, u_ay::Vector{Vector{T}},
                             Uz::PGDComponent, u_az::Vector{Vector{T}},
                             UN::u_shapefunctions{3,Float64,9},global_u_fe_values::FEValues,
                             E::Tensor{4,3,Float64,81},b::Tensor{1,3,Float64,3},u_xbc::Vector{Int},
                             Dx::PGDComponent, d_ax::Vector{Vector{T}},
                             Dy::PGDComponent, d_ay::Vector{Vector{T}},
                             Dz::PGDComponent, d_az::Vector{Vector{T}},
                             DN::d_shapefunctions{3,Float64},
                             Ψ_new::Vector{Vector{Float64}})
    # input check
    length(u_ax) == length(u_ay) == length(u_az) || throw(ArgumentError("Something is wrong."))
    length(d_ax) == length(d_ay) == length(d_az) || throw(ArgumentError("Something is wrong."))

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(Ux.mesh.edof)
    nElDofs = Ux.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    global_el = 0
    for el1 in 1:Ux.mesh.nEl, el2 in 1:Uy.mesh.nEl, el3 in 1:Uz.mesh.nEl
        global_el += 1
        u_m_x = Ux.mesh.edof[:,el1]
        u_m_y = Uy.mesh.edof[:,el2]
        u_m_z = Uz.mesh.edof[:,el3]

        d_m_x = Dx.mesh.edof[:,el1]
        d_m_y = Dy.mesh.edof[:,el2]
        d_m_z = Dz.mesh.edof[:,el3]

        fe, Ke, Ψe = ud_x_ele_3D_3D(fe,Ke,Ux,extract_eldofs(u_ax,u_m_x),
                                          Uy,extract_eldofs(u_ay,u_m_y),
                                          Uz,extract_eldofs(u_az,u_m_z),
                                          UN,global_u_fe_values,
                                          E,b,
                                          Dx,extract_eldofs(d_ax,d_m_x),
                                          Dy,extract_eldofs(d_ay,d_m_y),
                                          Dz,extract_eldofs(d_az,d_m_z),
                                          DN)
        assemble(u_m_x,_K,Ke)
        fext[u_m_x] += fe
        Ψ_new[global_el] = Ψe
    end

    K_x = end_assemble(_K)

    free = setdiff(1:nDofs,u_xbc)
    u_ax_new = zeros(nDofs)
    u_ax_new[free] += K_x[free,free]\fext[free]

    return u_ax_new, Ψ_new
end
# y
function ud_y_mode_solver{T}(Ux::PGDComponent, u_ax::Vector{Vector{T}},
                             Uy::PGDComponent, u_ay::Vector{Vector{T}},
                             Uz::PGDComponent, u_az::Vector{Vector{T}},
                             UN::u_shapefunctions{3,Float64,9},global_u_fe_values::FEValues,
                             E::Tensor{4,3,Float64,81},b::Tensor{1,3,Float64,3},u_ybc::Vector{Int},
                             Dx::PGDComponent, d_ax::Vector{Vector{T}},
                             Dy::PGDComponent, d_ay::Vector{Vector{T}},
                             Dz::PGDComponent, d_az::Vector{Vector{T}},
                             DN::d_shapefunctions{3,Float64},
                             Ψ_new::Vector{Vector{Float64}})
    # input check
    length(u_ax) == length(u_ay) == length(u_az) || throw(ArgumentError("Something is wrong."))
    length(d_ax) == length(d_ay) == length(d_az) || throw(ArgumentError("Something is wrong."))

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(Uy.mesh.edof)
    nElDofs = Uy.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    global_el = 0
    for el1 in 1:Ux.mesh.nEl, el2 in 1:Uy.mesh.nEl, el3 in 1:Uz.mesh.nEl
        global_el += 1
        u_m_x = Ux.mesh.edof[:,el1]
        u_m_y = Uy.mesh.edof[:,el2]
        u_m_z = Uz.mesh.edof[:,el3]

        d_m_x = Dx.mesh.edof[:,el1]
        d_m_y = Dy.mesh.edof[:,el2]
        d_m_z = Dz.mesh.edof[:,el3]

        fe, Ke, Ψe = ud_y_ele_3D_3D(fe,Ke,Ux,extract_eldofs(u_ax,u_m_x),
                                          Uy,extract_eldofs(u_ay,u_m_y),
                                          Uz,extract_eldofs(u_az,u_m_z),
                                          UN,global_u_fe_values,
                                          E,b,
                                          Dx,extract_eldofs(d_ax,d_m_x),
                                          Dy,extract_eldofs(d_ay,d_m_y),
                                          Dz,extract_eldofs(d_az,d_m_z),
                                          DN)
        assemble(u_m_y,_K,Ke)
        fext[u_m_y] += fe
        Ψ_new[global_el] = Ψe
    end

    K_y = end_assemble(_K)

    free = setdiff(1:nDofs,u_ybc)
    u_ay_new = zeros(nDofs)
    u_ay_new[free] += K_y[free,free]\fext[free]

    return u_ay_new, Ψ_new
end

# z
function ud_z_mode_solver{T}(Ux::PGDComponent, u_ax::Vector{Vector{T}},
                             Uy::PGDComponent, u_ay::Vector{Vector{T}},
                             Uz::PGDComponent, u_az::Vector{Vector{T}},
                             UN::u_shapefunctions{3,Float64,9},global_u_fe_values::FEValues,
                             E::Tensor{4,3,Float64,81},b::Tensor{1,3,Float64,3},u_zbc::Vector{Int},
                             Dx::PGDComponent, d_ax::Vector{Vector{T}},
                             Dy::PGDComponent, d_ay::Vector{Vector{T}},
                             Dz::PGDComponent, d_az::Vector{Vector{T}},
                             DN::d_shapefunctions{3,Float64},
                             Ψ_new::Vector{Vector{Float64}})
    # input check
    length(u_ax) == length(u_ay) == length(u_az) || throw(ArgumentError("Something is wrong."))
    length(d_ax) == length(d_ay) == length(d_az) || throw(ArgumentError("Something is wrong."))

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(Uz.mesh.edof)
    nElDofs = Uz.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    global_el = 0
    for el1 in 1:Ux.mesh.nEl, el2 in 1:Uy.mesh.nEl, el3 in 1:Uz.mesh.nEl
        global_el += 1
        u_m_x = Ux.mesh.edof[:,el1]
        u_m_y = Uy.mesh.edof[:,el2]
        u_m_z = Uz.mesh.edof[:,el3]

        d_m_x = Dx.mesh.edof[:,el1]
        d_m_y = Dy.mesh.edof[:,el2]
        d_m_z = Dz.mesh.edof[:,el3]

        fe, Ke, Ψe = ud_z_ele_3D_3D(fe,Ke,Ux,extract_eldofs(u_ax,u_m_x),
                                          Uy,extract_eldofs(u_ay,u_m_y),
                                          Uz,extract_eldofs(u_az,u_m_z),
                                          UN,global_u_fe_values,
                                          E,b,
                                          Dx,extract_eldofs(d_ax,d_m_x),
                                          Dy,extract_eldofs(d_ay,d_m_y),
                                          Dz,extract_eldofs(d_az,d_m_z),
                                          DN)
        assemble(u_m_z,_K,Ke)
        fext[u_m_z] += fe
        Ψ_new[global_el] = Ψe
    end

    K_z = end_assemble(_K)

    free = setdiff(1:nDofs,u_zbc)
    u_az_new = zeros(nDofs)
    u_az_new[free] += K_z[free,free]\fext[free]

    return u_az_new, Ψ_new
end

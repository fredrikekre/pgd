###################################################
# Displacement with damage 2D with 2D integration #
###################################################
# x
function ud_x_mode_solver{T}(Ux::PGDComponent, u_ax::Vector{Vector{T}},
                             Uy::PGDComponent, u_ay::Vector{Vector{T}},
                             UN::u_shapefunctions{2,Float64,4},global_u_fe_values::FEValues,
                             E::Tensor{4,2,Float64,16},b::Tensor{1,2,Float64,2},u_xbc::Vector{Int},
                             Dx::PGDComponent, d_ax::Vector{Vector{T}},
                             Dy::PGDComponent, d_ay::Vector{Vector{T}},
                             DN::d_shapefunctions{2,Float64},
                             Ψ_new::Vector{Vector{Float64}})
    # input check
    length(u_ax) == length(u_ay) || throw(ArgumentError("Something is wrong."))
    length(d_ax) == length(d_ay) || throw(ArgumentError("Something is wrong."))

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(Ux.mesh.edof)
    nElDofs = Ux.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    global_el = 0
    for el1 in 1:Ux.mesh.nEl, el2 in 1:Uy.mesh.nEl
        global_el += 1
        u_m_x = Ux.mesh.edof[:,el1]
        u_m_y = Uy.mesh.edof[:,el2]

        d_m_x = Dx.mesh.edof[:,el1]
        d_m_y = Dy.mesh.edof[:,el2]

        fe, Ke, Ψe = ud_x_ele_2D_2D(fe,Ke,Ux,extract_eldofs(u_ax,u_m_x),
                                          Uy,extract_eldofs(u_ay,u_m_y),
                                          UN,global_u_fe_values,
                                          E,b,
                                          Dx,extract_eldofs(d_ax,d_m_x),
                                          Dy,extract_eldofs(d_ay,d_m_y),
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
                             UN::u_shapefunctions{2,Float64,4},global_u_fe_values::FEValues,
                             E::Tensor{4,2,Float64,16},b::Tensor{1,2,Float64,2},u_ybc::Vector{Int},
                             Dx::PGDComponent, d_ax::Vector{Vector{T}},
                             Dy::PGDComponent, d_ay::Vector{Vector{T}},
                             DN::d_shapefunctions{2,Float64},
                             Ψ_new::Vector{Vector{Float64}})
    # input check
    length(u_ax) == length(u_ay) || throw(ArgumentError("Something is wrong."))
    length(d_ax) == length(d_ay) || throw(ArgumentError("Something is wrong."))

    # Set up integration of stiffness matrix and force vector
    nDofs = maximum(Uy.mesh.edof)
    nElDofs = Uy.mesh.nElDofs

    fext = zeros(nDofs)
    _K = start_assemble()

    fe = zeros(nElDofs)
    Ke = zeros(nElDofs,nElDofs)

    global_el = 0
    for el1 in 1:Ux.mesh.nEl, el2 in 1:Uy.mesh.nEl
        global_el += 1
        u_m_x = Ux.mesh.edof[:,el1]
        u_m_y = Uy.mesh.edof[:,el2]

        d_m_x = Dx.mesh.edof[:,el1]
        d_m_y = Dy.mesh.edof[:,el2]

        fe, Ke, Ψe = ud_y_ele_2D_2D(fe,Ke,Ux,extract_eldofs(u_ax,u_m_x),
                                          Uy,extract_eldofs(u_ay,u_m_y),
                                          UN,global_u_fe_values,
                                          E,b,
                                          Dx,extract_eldofs(d_ax,d_m_x),
                                          Dy,extract_eldofs(d_ay,d_m_y),
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

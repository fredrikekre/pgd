############################
# Solves displacement mode #
############################

function U_ModeSolver(U_a::Matrix,U_a_old::Matrix,U::PGDFunction,U_bc::PGDBC,U_edof::Matrix{Int},
                      D_mat::Matrix{Int},b::Vector,modeItr::Int)

    # Set up initial stuff
    full_solution = U_a_old[:,modeItr] # Reuse last loadstep's mode as initial guess
    trial_solution = zeros(number_of_dofs(U_edof))

    Δan_0 = 0.1*ones(Float64, number_of_dofs(U_edof)) # Initial guess
    Δan_0[fixed_dofs(U_bc)] = 0.0
    Δan_0[prescr_dofs(U_bc)] = 0.0


    # Total solution need to satisfy boundary conditions so we enforce that here
    full_solution[prescr_dofs(U_bc)] = 0.0
    full_solution[fixed_dofs(U_bc)] = 0.0

    Δan = copy(Δan_0)
    # Δan_y = copy(Δan_y_0)
    i = -1; TOL = 1e-6; maxofg = 1
    while true; i += 1
        # tic()
        copy!(trial_solution,full_solution)
        trial_solution[free_dofs(U_bc)] += Δan[free_dofs(U_bc)]

        g = calc_globres_U(trial_solution,U_a,U,D_mat,U_edof,b,free_dofs(U_bc))

        maxofg = maximum(abs(g))
        println("Residual is now maxofg = $maxofg")
        if maxofg < TOL # converged

            U_atemp = copy(U_a)
            U_atemp[:,modeItr] = trial_solution
            tot_norms = norm_of_mode(U,U_atemp[:,1:U.modes+1])
            this_norms = norm_of_mode(U,trial_solution)
            ratios = (this_norms[1]/tot_norms[1], this_norms[2]/tot_norms[2])

            println("Converged mode #$modeItr in $i iterations. Ratios = $(ratios)")
            break
        else # do steps

            # Step in x-dir
            g_x = calc_globres_U(trial_solution,U_a,U,D_mat,U_edof,b,free_dofs(U_bc,1))
            K_x = calc_globK_U(trial_solution,U_a,U,D_mat,U_edof,b,free_dofs(U_bc,1))
            ΔΔan = cholfact(Symmetric(K_x, :U))\g_x
            Δan[free_dofs(U_bc,1)] -= ΔΔan

            # Update trial solution? Yes, should help when solving for y-dir
            trial_solution[free_dofs(U_bc,1)] -= ΔΔan # or trial_solution[free_dofs(bc_U,1)] = full_solution[free_dofs(bc_U,1)] + Δan[free_dofs(bc_U,1)]

            # Step in y-dir
            g_y = calc_globres_U(trial_solution,U_a,U,D_mat,U_edof,b,free_dofs(U_bc,2))
            K_y = calc_globK_U(trial_solution,U_a,U,D_mat,U_edof,b,free_dofs(U_bc,2))
            ΔΔan = cholfact(Symmetric(K_y, :U))\g_y
            Δan[free_dofs(U_bc,2)] -= ΔΔan

            # trial_solution[free_dofs(bc_U,2)] -= ΔΔan # or trial_solution[free_dofs(bc_U,2)] = full_solution[free_dofs(bc_U,2)] + Δan[free_dofs(bc_U,2)]

            if i > 149
                warn("Exited loop after $i iterations with residual $maxofg")
                break
            end
        end
        # toc()
    end

    full_solution[free_dofs(U_bc)] += Δan[free_dofs(U_bc)]

    return full_solution
end


#############################################
# Solves displacementmode with given damage #
#############################################
function UD_ModeSolver(U_a::Matrix,U_a_old::Matrix,U::PGDFunction,U_bc::PGDBC,U_edof::Matrix{Int},
                       D_a::Matrix,D_a_old::Matrix,D::PGDFunction,D_bc::PGDBC,D_edof::Matrix{Int},
                       U_mp_tangent::Matrix,b::Vector,modeItr::Int)

    # Set up initial stuff
    full_solution = U_a_old[:,modeItr] # Reuse last loadstep's mode as initial guess
    full_solution = zeros(Float64,number_of_dofs(U_edof))
    trial_solution = zeros(Float64,number_of_dofs(U_edof))

    Δan_0 = 0.1*ones(Float64, number_of_dofs(U_edof)) # Initial guess
    Δan_0[fixed_dofs(U_bc)] = 0.0
    Δan_0[prescr_dofs(U_bc)] = 0.0


    # Total solution need to satisfy boundary conditions so we enforce that here
    full_solution[prescr_dofs(U_bc)] = 0.0
    full_solution[fixed_dofs(U_bc)] = 0.0

    Δan = copy(Δan_0)
    # Δan_y = copy(Δan_y_0)
    i = -1; TOL = 1e-6; maxofg = 1.0
    while true; i += 1
        # tic()
        copy!(trial_solution,full_solution)
        trial_solution[free_dofs(U_bc)] += Δan[free_dofs(U_bc)]

        g = calc_globres_UD(trial_solution,U_a,U,U_edof,free_dofs(U_bc),
                                           D_a,D,D_edof,
                                           U_mp_tangent,b)
        maxofg = maximum(abs(g))
        # println("Residual is now maxofg = $maxofg")
        if maxofg < TOL # converged

            U_atemp = copy(U_a)
            U_atemp[:,modeItr] = trial_solution
            tot_norms = norm_of_mode(U,U_atemp[:,1:U.modes+1])
            this_norms = norm_of_mode(U,trial_solution)
            ratios = (this_norms[1]/tot_norms[1], this_norms[2]/tot_norms[2])

            println("Converged mode #$modeItr in $i iterations. Ratios = $(ratios)")
            break
        else # do steps

            # Step in x-dir
            g_x = calc_globres_UD(trial_solution,U_a,U,U_edof,free_dofs(U_bc,1),
                                                 D_a,D,D_edof,
                                                 U_mp_tangent,b)
            K_x = calc_globK_UD(trial_solution,U_a,U,U_edof,free_dofs(U_bc,1),
                                               D_a,D,D_edof,
                                               U_mp_tangent,b)
            ΔΔan = cholfact(Symmetric(K_x, :U))\g_x
            Δan[free_dofs(U_bc,1)] -= ΔΔan

            # Update trial solution? Yes, should help when solving for y-dir
            trial_solution[free_dofs(U_bc,1)] -= ΔΔan # or trial_solution[free_dofs(bc_U,1)] = full_solution[free_dofs(bc_U,1)] + Δan[free_dofs(bc_U,1)]

            # Step in y-dir
            g_y = calc_globres_UD(trial_solution,U_a,U,U_edof,free_dofs(U_bc,2),
                                                 D_a,D,D_edof,
                                                 U_mp_tangent,b)
            K_y = calc_globK_UD(trial_solution,U_a,U,U_edof,free_dofs(U_bc,2),
                                               D_a,D,D_edof,
                                               U_mp_tangent,b)
            ΔΔan = cholfact(Symmetric(K_y, :U))\g_y
            Δan[free_dofs(U_bc,2)] -= ΔΔan

            # trial_solution[free_dofs(bc_U,2)] -= ΔΔan # or trial_solution[free_dofs(bc_U,2)] = full_solution[free_dofs(bc_U,2)] + Δan[free_dofs(bc_U,2)]

            if i > 149
                warn("Exited loop after $i iterations with residual $maxofg")
                break
            end
        end
        # toc()
    end

    full_solution[free_dofs(U_bc)] += Δan[free_dofs(U_bc)]

    return full_solution
end


##############################################
# Solves damagemode with given displacements #
##############################################

function DU_ModeSolver(D_a::Matrix,D_a_old::Matrix,D::PGDFunction,D_bc::PGDBC,D_edof::Matrix{Int},
                       U_a::Matrix,U_a_old::Matrix,U::PGDFunction,U_bc::PGDBC,U_edof::Matrix{Int},
                       D_mp::PhaseFieldDamage,Ψ::Float64,modeItr::Int)

    # Set up initial stuff
    full_solution = D_a_old[:,modeItr] # Reuse last loadstep's mode as initial guess
    trial_solution = zeros(number_of_dofs(D_edof))

    Δan_0 = 0.1*ones(Float64, number_of_dofs(D_edof)) # Initial guess
    Δan_0[fixed_dofs(D_bc)] = 0.0
    Δan_0[prescr_dofs(D_bc)] = 0.0


    # Total solution need to satisfy boundary conditions so we enforce that here
    full_solution[prescr_dofs(D_bc)] = 0.0
    full_solution[fixed_dofs(D_bc)] = 0.0

    Δan = copy(Δan_0)
    # Δan_y = copy(Δan_y_0)
    i = -1; TOL = 1e-6; maxofg = 1.0
    while true; i += 1
        # tic()
        copy!(trial_solution,full_solution)
        trial_solution[free_dofs(D_bc)] += Δan[free_dofs(D_bc)]

        g = calc_globres_DU(trial_solution,D_a,D,D_edof,free_dofs(D_bc),
                                           U_a,U,U_edof,
                                           D_mp,Ψ)

        maxofg = maximum(abs(g))
        println("Residual is now maxofg = $maxofg")
        if maxofg < TOL # converged

            # D_atemp = copy(D_a)
            # D_atemp[:,modeItr+1] = trial_solution
            # tot_norms = norm_of_mode(D,D_atemp[:,1:D.modes+1])
            # this_norms = norm_of_mode(D,trial_solution)
            # ratios = (this_norms[1]/tot_norms[1], this_norms[2]/tot_norms[2])

            # println("Converged mode #$modeItr in $i iterations. Ratios = $(ratios)")
            println("Converged mode #$modeItr in $i iterations.")
            break
        else # do steps

            #################
            # Step in x-dir #
            #################
            g_x = calc_globres_DU(trial_solution,D_a,D,D_edof,free_dofs(D_bc,1),
                                                 U_a,U,U_edof,
                                                 D_mp,Ψ)

            K_x = calc_globK_DU(trial_solution,D_a,D,D_edof,free_dofs(D_bc,1),
                                                 U_a,U,U_edof,
                                                 D_mp,Ψ)

            ΔΔan = cholfact(Symmetric(K_x, :U))\g_x
            Δan[free_dofs(D_bc,1)] -= ΔΔan

            # Update trial solution? Yes, should help when solving for y-dir
            trial_solution[free_dofs(D_bc,1)] -= ΔΔan # or trial_solution[free_dofs(bc_U,1)] = full_solution[free_dofs(bc_U,1)] + Δan[free_dofs(bc_U,1)]

            #################
            # Step in y-dir #
            #################
            g_y = calc_globres_DU(trial_solution,D_a,D,D_edof,free_dofs(D_bc,2),
                                                 U_a,U,U_edof,
                                                 D_mp,Ψ)
            K_y = calc_globK_DU(trial_solution,D_a,D,D_edof,free_dofs(D_bc,2),
                                                 U_a,U,U_edof,
                                                 D_mp,Ψ)

            ΔΔan = cholfact(Symmetric(K_y, :U))\g_y
            Δan[free_dofs(D_bc,2)] -= ΔΔan

            # trial_solution[free_dofs(bc_U,2)] -= ΔΔan # or trial_solution[free_dofs(bc_U,2)] = full_solution[free_dofs(bc_U,2)] + Δan[free_dofs(bc_U,2)]

            if i > 149
                warn("Exited loop after $i iterations with residual $maxofg")
                break
            end
        end
        # toc()
    end

    full_solution[free_dofs(D_bc)] += Δan[free_dofs(D_bc)]

    return full_solution
end

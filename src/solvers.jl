function displacementModeSolver(a,a_old,U,bc_U,ndofs,D,edof,b,modeItr)

    # Set up initial stuff
    full_solution = a_old[:,modeItr] # Reuse last loadstep's mode as initial guess
    full_solution = a_old[:,modeItr]*0.0
    trial_solution = zeros(ndofs)

    Δan_0 = 1.0*ones(Float64, ndofs) # Initial guess
    Δan_0[fixed_dofs(bc_U)] = 0.0
    Δan_0[prescr_dofs(bc_U)] = 0.0


    # Total solution need to satisfy boundary conditions so we enforce that here
    full_solution[prescr_dofs(bc_U)] = 0.0
    full_solution[fixed_dofs(bc_U)] = 0.0

    Δan = copy(Δan_0)
    # Δan_y = copy(Δan_y_0)
    i = -1; TOL = 1e-7; maxofg = 1
    while true; i += 1
        # tic()
        copy!(trial_solution,full_solution)
        trial_solution[free_dofs(bc_U)] += Δan[free_dofs(bc_U)]

        g = calc_globres(trial_solution,a,U,D,edof,b,free_dofs(bc_U))

        maxofg = maximum(abs(g))
        println("Residual is now maxofg = $maxofg")
        if maxofg < TOL # converged
            println("Converged for displacement mode #$modeItr in $i iterations. max(g) = $(maxofg)")
            break
        else # do steps

            # Step in x-dir
            g_x = calc_globres(trial_solution,a,U,D,edof,b,free_dofs(bc_U,1))
            K_x = calc_globK(trial_solution,a,U,D,edof,b,free_dofs(bc_U,1))
            ΔΔan = cholfact(Symmetric(K_x, :U))\g_x
            Δan[free_dofs(bc_U,1)] -= ΔΔan

            # Update trial solution? Yes, should help when solving for y-dir
            trial_solution[free_dofs(bc_U,1)] -= ΔΔan

            # Step in y-dir
            g_y = calc_globres(trial_solution,a,U,D,edof,b,free_dofs(bc_U,2))
            K_y = calc_globK(trial_solution,a,U,D,edof,b,free_dofs(bc_U,2))
            ΔΔan = cholfact(Symmetric(K_y, :U))\g_y
            Δan[free_dofs(bc_U,2)] -= ΔΔan


            # trial_solution[free_dofs(bc_U,2)] = full_solution[free_dofs(bc_U,2)] + Δan[free_dofs(bc_U,2)]

            if i > 99
                println("Iteration is now $i")
            end
        end
        # toc()
    end

    full_solution[free_dofs(bc_U)] += Δan[free_dofs(bc_U)]

    return full_solution
end
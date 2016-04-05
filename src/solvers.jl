function displacementModeSolver(a,U,ndofs,bc,bc_x,bc_y,D,edof,b,free,free_x,free_y,modeItr)
    full_solution = zeros(ndofs)
    trial_solution = zeros(ndofs)

    Δan_0 = 1.0*ones(Float64, ndofs)
    Δan_0[bc[1]] = zeros(bc[1])


    # Total solution need to satisfy boundary conditions so we enforce that here
    full_solution[bc_x[1]] = bc_x[2]
    full_solution[bc_y[1]] = bc_y[2]

    Δan = copy(Δan_0)
    # Δan_y = copy(Δan_y_0)
    i = -1; TOL = 1e-7; maxofg = 1;
    while true; i += 1
        # tic()
        trial_solution[free] = full_solution[free] + Δan[free]

        g = calc_globres(trial_solution,a,U,D,edof,b,free)

        maxofg = maximum(abs(g))
        # println("Residual is now maxofg = $maxofg")
        if maxofg < TOL # converged
            println("Converged for displacement mode #$modeItr in $i iterations. max(g) = $(maxofg)")
            break
        else # do steps

            # Step in x-dir
            g_x = calc_globres(trial_solution,a,U,D,edof,b,free_x)
            K_x = calc_globK(trial_solution,a,U,D,edof,b,free_x)
            ΔΔan = cholfact(Symmetric(K_x, :U))\g_x
            Δan[free_x] -= ΔΔan

            # Update trial solution? Yes, should help when solving for y-dir
            trial_solution[free_x] = full_solution[free_x] + Δan[free_x]

            # Step in y-dir
            g_y = calc_globres(trial_solution,a,U,D,edof,b,free_y)
            K_y = calc_globK(trial_solution,a,U,D,edof,b,free_y)
            ΔΔan = cholfact(Symmetric(K_y, :U))\g_y
            Δan[free_y] -= ΔΔan

            # trial_solution[free_y] = full_solution[free_y] + Δan[free_y]

            if i > 99
                println("Iteration is now $i")
            end
        end
        # toc()
    end

    full_solution[free] += Δan[free]

    return full_solution
end
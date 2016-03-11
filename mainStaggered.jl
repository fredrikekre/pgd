import JuAFEM
using InplaceOps
using ForwardDiff
using NLsolve

include("src/meshgenerator.jl")
include("src/PGDmodule.jl")
include("src/utilities.jl")
include("src/elementFunctions.jl")
include("src/globResK.jl")
include("src/visualize.jl")

function mainNewton()
    xStart = 0; yStart = 0
    xEnd = 1; yEnd = 1
    xnEl = 50; ynEl = 50
    xnElNodes = 2; ynElNodes = 2
    xnNodeDofs = 2; ynNodeDofs = 2
    xmesh = create_mesh1D(xStart,xEnd,xnEl,xnElNodes,xnNodeDofs)
    ymesh = create_mesh1D(yStart,yEnd,ynEl,ynElNodes,ynNodeDofs)
    xymesh = create_mesh2D(xStart,xEnd,yStart,yEnd,xnEl,ynEl,2)

    # Set up function components
    function_space = JuAFEM.Lagrange{1,JuAFEM.Square,1}()
    q_rule = JuAFEM.get_gaussrule(JuAFEM.Dim{1},JuAFEM.Square(),1)
    fevx = JuAFEM.FEValues(Float64,q_rule,function_space)
    fevy = JuAFEM.FEValues(Float64,q_rule,function_space)

    Ux = PGDComponent(1,xmesh,fevx)
    Uy = PGDComponent(1,ymesh,fevy)

    # Set up PGDfunction
    function_space = JuAFEM.Lagrange{2,JuAFEM.Square,1}()
    q_rule = JuAFEM.get_gaussrule(JuAFEM.Dim{2},JuAFEM.Square(),2)
    fevxy = JuAFEM.FEValues(Float64,q_rule,function_space)

    U = PGDFunction(2,2,xymesh,fevxy,[Ux, Uy])
    edof = create_edof(U)

    x_mode_dofs = collect(1:U.components[1].mesh.nDofs)
    y_mode_dofs = collect((U.components[1].mesh.nDofs+1):(U.components[1].mesh.nDofs+U.components[2].mesh.nDofs))

    # Material stiffness
    E = 1; ν = 0.3
    D = JuAFEM.hooke(2,E,ν); D = D[[1,2,4],[1,2,4]]

    # Boundary conditions
    bc = Vector[[x_mode_dofs[[1,2]];
                 x_mode_dofs[[end-1,end]];
                 y_mode_dofs[[1,2]];
                 y_mode_dofs[[end-1,end]]],
                 0.0*[x_mode_dofs[[1,2]];
                 x_mode_dofs[[end-1,end]];
                 y_mode_dofs[[1,2]];
                 y_mode_dofs[[end-1,end]]]]

    # bc_1: Lock y_mode_dofs
    bc_x = Vector[[x_mode_dofs[[1,2]];
                   x_mode_dofs[[end-1,end]];
                   y_mode_dofs],
                   0.0*[x_mode_dofs[[1,2]];
                   x_mode_dofs[[end-1,end]];
                   y_mode_dofs]]

    # bc_2: Lock x_mode_dofs
    bc_y = Vector[[x_mode_dofs;
                   y_mode_dofs[[1,2]];
                   y_mode_dofs[[end-1,end]]],
                   0.0*[x_mode_dofs;
                   y_mode_dofs[[1,2]];
                   y_mode_dofs[[end-1,end]]]]

    ndofs = maximum(edof)
    free = setdiff(1:ndofs,bc[1])
    free_x = setdiff(1:ndofs,bc_x[1])
    free_y = setdiff(1:ndofs,bc_y[1])

    n_free = length(free)
    n_free_dofs_x = length(free_x)
    n_free_dofs_y = length(free_y)

    b = [1.0,1.0] # Body force

    n_modes = 5
    a = zeros(ndofs, n_modes)
    for modeItr = 1:n_modes
        tic()
        # Current full solution and trial
        full_solution = zeros(ndofs)
        trial_solution = zeros(ndofs)

        Δan_0 = 1.0*ones(Float64, ndofs)
        Δan_0[bc[1]] = zeros(bc[1])

        # Δan_x_0 = 1.0*ones(Float64, n_free_dofs_x)
        # Δan_y_0 = 1.0*ones(Float64, n_free_dofs_y)

        # # Test with modes from 1D as starting guesses
        # amps = readdlm("amplitude_fp.txt",',',Float64)[:]
        # Tx = readdlm("Tx_fp.txt",',',Float64)
        # Ty = readdlm("Ty_fp.txt",',',Float64)
        # Δan_0 = Float64[]
        # push!(Δan_0,amps[modeItr])
        # append!(Δan_0,Tx[:,modeItr])
        # append!(Δan_0,Ty[:,modeItr])
        # push!(Δan_0,1.0) # λ
        # Δan_1_0 = Δan_0[free_1]
        # Δan_2_0 = Δan_0[free_2]


        # Total solution need to satisfy boundary conditions so we enforce that here
        full_solution[bc_x[1]] = bc_y[2]
        full_solution[bc_y[1]] = bc_x[2]

        Δan = copy(Δan_0)
        # Δan_y = copy(Δan_y_0)
        i = 0; tol = 1e-5; maxofg = 1; tol_fixpoint = 1e-10; hej = 0;
        while true; i += 1
            # tic()
            trial_solution[free] = full_solution[free] + Δan[free]
            # trial_solution[free_y] = full_solution[free_y] + Δan_y

            if maxofg > tol_fixpoint && hej ==0
                # Step in x-dir
                g_x = calc_globres(trial_solution,a,U,D,edof,b,free_x)
                K_x = calc_globK(trial_solution,a,U,D,edof,b,free_x)
                ΔΔan = -K_x\g_x
                Δan[free_x] += ΔΔan

                # Update trial solution? Yes, should help when solving for y-dir
                trial_solution[free_x] = full_solution[free_x] + Δan[free_x]

                # Step in y-dir
                g_y = calc_globres(trial_solution,a,U,D,edof,b,free_y)
                K_y = calc_globK(trial_solution,a,U,D,edof,b,free_y)
                ΔΔan = -K_y\g_y
                Δan[free_y] += ΔΔan

                trial_solution[free_y] = full_solution[free_y] + Δan[free_y]

            else
                # Full newton
                hej += 1
                println("Switched to Newton solver")
                g = calc_globres(trial_solution,a,U,D,edof,b,free)
                K = calc_globK(trial_solution,a,U,D,edof,b,free)
                ΔΔan = -K\g
                Δan[free] += ΔΔan

                trial_solution[free] = full_solution[free] + Δan[free]

            end

            # Check residuals
            # trial_solution[free_1] = full_solution[free_1] + Δan_1
            # trial_solution[free_2] = full_solution[free_2] + Δan_2

            g = calc_globres(trial_solution,a,U,D,edof,b,free)
            # g1 = calc_globres(trial_solution,a,U,D,edof,b,free_1)
            # g2 = calc_globres(trial_solution,a,U,D,edof,b,free_2)

            maxofg = maximum(abs(g))
            println("Residual is now maxofg = $maxofg")
            if maxofg < tol
                println("Converged for mode #$modeItr in $i iterations. max(g) = $(maxofg)")
                break
            elseif i > 99
                println("Iteration is now $i")
            end
            # toc()
        end

        # The total solution after the Newtons step is the previous plus
        # the solution to the Newton iterations.
        full_solution[free] += Δan[free]
        # full_solution[free_1] += Δan_1
        # full_solution[free_2] += Δan_2
        a[:,modeItr] = full_solution;
        U.modes += 1
        toc()
    end
    return a, U
end

o = mainNewton()



visualize(o...)










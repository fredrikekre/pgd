import JuAFEM
using InplaceOps
using ForwardDiff
using NLsolve

include("src/meshgenerator.jl")
include("src/PGDmodule.jl")
include("src/utilities.jl")
include("src/heatElementFunctions.jl")
# include("src/solidElementFunctions.jl")
include("src/globResK.jl")
include("src/visualize_heat.jl")

function mainNewtonAmplitudeFormulation()
    xStart = 0; yStart = 0
    xEnd = 2; yEnd = 2
    xnEl = 3; ynEl = 3
    xnElNodes = 2; ynElNodes = 2
    xnNodeDofs = 1; ynNodeDofs = 1
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
    edof = create_edof_heat(U)

    # Temporary, add amplitude dof at the top and λ-dofs at the bottom
    edof = [ones(edof[1,:]); # T amplitude (same for all elements)
            edof+1; # usual edof matrix but +2 since dof 1,2 = amplitude
            (maximum(edof)+2)*ones(edof[1,:])] # λ, same for all elements

    # Material stiffness
    E = 1; ν = 0.3
    D = JuAFEM.hooke(2,E,ν); D = D[[1,2,4],[1,2,4]]
    D = [1.0 0.0; 0.0 1.0]
    # Boundary conditions
    α_dofs = 1
    x_mode_dofs = collect(1:U.components[1].mesh.nDofs) + 1
    y_mode_dofs = collect(U.components[1].mesh.nDofs+1:U.components[1].mesh.nDofs+U.components[2].mesh.nDofs) + 1
    λ_dofs = maximum(y_mode_dofs) + 1

    # bc1: Lock y_mode_dofs
    bc_1 = Vector[[x_mode_dofs[1];
                 x_mode_dofs[end];
                 y_mode_dofs],
                 0.0*[x_mode_dofs[1];
                 x_mode_dofs[end];
                 y_mode_dofs]]

    # bc2: Lock x_mode_dofs
    bc_2 = Vector[[α_dofs;
                   x_mode_dofs;
                   y_mode_dofs[1];
                   y_mode_dofs[end];
                   λ_dofs],
                   0.0*[α_dofs;
                   x_mode_dofs;
                   y_mode_dofs[1];
                   y_mode_dofs[end];
                   λ_dofs]]

    ndofs = maximum(edof)
    free_1 = setdiff(1:ndofs,bc_1[1])
    free_2 = setdiff(1:ndofs,bc_2[1])

    # fixed = Vector[setdiff(1:ndofs, free)
    n_free_dofs_1 = length(free_1)
    n_free_dofs_2 = length(free_2)

    b = 1.0 # Body heat source

    n_modes = 1
    a = zeros(ndofs, n_modes)

    for modeItr in 1:n_modes

         # Current full solution and trial
        full_solution = zeros(ndofs)
        trial_solution = zeros(ndofs)

        Δan_1_0 = 1.0*ones(Float64, n_free_dofs_1)
        Δan_2_0 = 1.0*ones(Float64, n_free_dofs_2)

        # function f!(Δan, fvec) # For NLsolve
        #     # Update the trial solution with the current solution
        #     # plus the trial step
        #     trial_solution[free] = full_solution[free] + Δan
        #     globres = calc_globres_heat(trial_solution, a,U,D,edof,b, free)

        #     copy!(fvec, globres)
        # end

        # function g!(Δan, gjac) # For NLsolve
        #     trial_solution[free] = full_solution[free] + Δan
        #     K = calc_globK_heat(trial_solution,a,U,D,edof,b,free)

        #     # Workaround for copy! being slow on 0.4 for sparse matrices
        #     gjac.colptr = K.colptr
        #     gjac.rowval = K.rowval
        #     gjac.nzval = K.nzval
        # end

        # Test with modes from 1D as starting guesses
        amps = readdlm("amplitude_fp.txt",',',Float64)[:]
        Tx = readdlm("Tx_fp.txt",',',Float64)
        Ty = readdlm("Ty_fp.txt",',',Float64)
        Δan_0 = Float64[]
        push!(Δan_0,amps[modeItr])
        append!(Δan_0,Tx[:,modeItr])
        append!(Δan_0,Ty[:,modeItr])
        push!(Δan_0,1.0) # λ
        Δan_1_0 = Δan_0[free_1]
        Δan_2_0 = Δan_0[free_2]


        # Total solution need to satisfy boundary conditions so we enforce that here
        full_solution[bc_1[1]] = bc_1[2]
        full_solution[bc_2[1]] = bc_2[2]

        # df = DifferentiableSparseMultivariateFunction(f!,g!)

        # res = nlsolve(df, Δan_0, ftol = 1e-7, iterations=1000, show_trace = true, method = :trust_region)
        # if !converged(res)
        #     # error("Global equation did not converge")
        # end
        Δan_1 = copy(Δan_1_0)
        Δan_2 = copy(Δan_2_0)
        i = 0; tol = 1e-7
        while true; i += 1
            trial_solution[free_1] = full_solution[free_1] + Δan_1
            trial_solution[free_2] = full_solution[free_2] + Δan_2

            # Step in x-dir
            g1 = calc_globres_heat(trial_solution,a,U,D,edof,b,free_1)
            K1 = calc_globK_heat(trial_solution,a,U,D,edof,b,free_1)
            println("det(K1) = $(det(K1))")
            ΔΔan_1 = -K1\g1
            Δan_1 += ΔΔan_1
            # Update trial solution?
            trial_solution[free_1] = full_solution[free_1] + Δan_1

            # Step in y-dir
            g2 = calc_globres_heat(trial_solution,a,U,D,edof,b,free_2)
            K2 = calc_globK_heat(trial_solution,a,U,D,edof,b,free_2)
            println("det(K2) = $(det(K2))")
            ΔΔan_2 = -K2\g2
            Δan_2 += ΔΔan_2

            # Check residuals
            # trial_solution[free_1] = full_solution[free_1] + Δan_1
            trial_solution[free_2] = full_solution[free_2] + Δan_2

            g1 = calc_globres_heat(trial_solution,a,U,D,edof,b,free_1)
            g2 = calc_globres_heat(trial_solution,a,U,D,edof,b,free_2)
            maxofg = maximum(abs([g1;g2]))
            println("Residual is now maxofg = $maxofg")
            if maxofg < 1e-7
                println("Converged in $i iterations, max(g) = $(maxofg)")
                break
            end
        end

        # The total solution after the Newtons step is the previous plus
        # the solution to the Newton iterations.
        full_solution[free_1] += Δan_1
        full_solution[free_2] += Δan_2
        a[:,modeItr] = full_solution;
        U.modes += 1
    end

    return a, U
end

o = mainNewtonAmplitudeFormulation()



# visualize(o...)










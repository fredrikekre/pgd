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
    xnEl = 20; ynEl = 20
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
    bc = [1                             0;
          U.components[1].mesh.nDofs    0;
          U.components[1].mesh.nDofs+1  0;
          U.components[1].mesh.nDofs+U.components[2].mesh.nDofs 0]

    bc[:,1] += 1 # Since I added α as dof 1

    ndofs = maximum(edof)
    free = setdiff(1:ndofs,bc[:,1])
    fixed = setdiff(1:ndofs, free)
    n_free_dofs = length(free)

    b = 1.0 # Body heat source

    n_modes = 5
    a = zeros(ndofs, n_modes)
    for modeItr in 1:n_modes

         # Current full solution and trial
        full_solution = zeros(ndofs)
        trial_solution = zeros(ndofs)

        function f!(Δan, fvec) # For NLsolve
            # Update the trial solution with the current solution
            # plus the trial step
            trial_solution[free] = full_solution[free] + Δan
            globres = calc_globres_heat(trial_solution, a,U,D,edof,b, free)

            copy!(fvec, globres)
        end

        function g!(Δan, gjac) # For NLsolve
            trial_solution[free] = full_solution[free] + Δan
            K = calc_globK_heat(trial_solution,a,U,D,edof,b,free)
            # println(full(K))
            # println(det(K))
            # error()
            # Workaround for copy! being slow on 0.4 for sparse matrices
            gjac.colptr = K.colptr
            gjac.rowval = K.rowval
            gjac.nzval = K.nzval
        end

        # Initial guess
        # We only guess on the unconstrained nodes, the others will not be
        # changed during the newton iterations
        Δan_0 = 1.1*ones(Float64, n_free_dofs)

        # Test with modes from 1D as starting guesses
        amps = readdlm("amplitude_fp.txt",',',Float64)[:]
        Tx = readdlm("Tx_fp.txt",',',Float64)
        Ty = readdlm("Ty_fp.txt",',',Float64)
        Δan_0 = Float64[]
        push!(Δan_0,amps[modeItr])
        append!(Δan_0,Tx[:,modeItr])
        append!(Δan_0,Ty[:,modeItr])
        push!(Δan_0,0.0) # λ
        Δan_0 = Δan_0[free]


        # Total solution need to satisfy boundary conditions so we enforce that here
        full_solution[bc[:,1]] = bc[:,2]

        df = DifferentiableSparseMultivariateFunction(f!,g!)

        res = nlsolve(df, Δan_0, ftol = 1e-7, iterations=1000, show_trace = true, method = :trust_region)
        if !converged(res)
            # error("Global equation did not converge")
        end

        # The total solution after the Newtons step is the previous plus
        # the solution to the Newton iterations.
        full_solution[free] += res.zero
        a[:,modeItr] = full_solution;
        U.modes += 1
    end

    return a, U
end

o = mainNewtonAmplitudeFormulation()



visualize(o...)










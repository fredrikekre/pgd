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
    xnEl = 100; ynEl = 100
    xnElNodes = 2; ynElNodes = 2
    xnNodeDofs = 2; ynNodeDofs = 2
    xmesh = create_mesh1D(xStart,xEnd,xnEl,xnElNodes,xnNodeDofs)
    ymesh = create_mesh1D(yStart,yEnd,ynEl,ynElNodes,ynNodeDofs)
    xymesh = create_mesh2D(xStart,xEnd,yStart,yEnd,xnEl,ynEl,2)

    # Set up function components
    function_space = JuAFEM.Lagrange{1,JuAFEM.Line}()
    q_rule = JuAFEM.get_gaussrule(JuAFEM.Line(),1)
    fevx = JuAFEM.FEValues(Float64,q_rule,function_space)
    fevy = JuAFEM.FEValues(Float64,q_rule,function_space)

    Ux = PGDComponent(1,xmesh,fevx)
    Uy = PGDComponent(1,ymesh,fevy)

    # Set up PGDfunction
    function_space = JuAFEM.Lagrange{1,JuAFEM.Square}()
    q_rule = JuAFEM.get_gaussrule(JuAFEM.Square(),2)
    fevxy = JuAFEM.FEValues(Float64,q_rule,function_space)

    U = PGDFunction(2,2,xymesh,fevxy,[Ux, Uy])
    edof = create_edof(U)
    λ_dofs = collect((maximum(edof)+1):(maximum(edof)+4)) # Add two Lagrange multipliers

    # Material stiffness
    E = 1; ν = 0.3
    D = JuAFEM.hooke(2,E,ν); D = D[[1,2,4],[1,2,4]]

    # Boundary conditions
    bc = [1                             0;
          2                             0;
          U.components[1].mesh.nDofs-1  0;
          U.components[1].mesh.nDofs    0;
          U.components[1].mesh.nDofs+1  0;
          U.components[1].mesh.nDofs+2  0;
          U.components[1].mesh.nDofs+U.components[2].mesh.nDofs-1 0;
          U.components[1].mesh.nDofs+U.components[2].mesh.nDofs 0]

    ndofs = maximum(edof) + length(λ_dofs)
    free = setdiff(1:ndofs,bc[:,1])
    fixed = setdiff(1:ndofs, free)
    n_free_dofs = length(free)

    b = [1.0,1.0] # Body force

    n_modes = 2
    a = zeros(ndofs, n_modes)
    for modeItr = 1:n_modes

         # Current full solution and trial
        full_solution = zeros(ndofs)
        trial_solution = zeros(ndofs)

        function f!(Δan, fvec) # For NLsolve
            # Update the trial solution with the current solution
            # plus the trial step
            trial_solution[free] = full_solution[free] + Δan
            globres = calc_globres(trial_solution, a,U,D,edof,λ_dofs,b, free)

            copy!(fvec, globres)
        end

        function g!(Δan, gjac) # For NLsolve
            trial_solution[free] = full_solution[free] + Δan
            K = calc_globK(trial_solution,a,U,D,edof,λ_dofs,b,free)
            # Workaround for copy! being slow on 0.4 for sparse matrices
            gjac.colptr = K.colptr
            gjac.rowval = K.rowval
            gjac.nzval = K.nzval
        end

        # # Mode 1 from fixpoint solution (20x20 elements x ∈ [0,20], y ∈ [0,20])
        # aX_fix = readdlm("aX.txt")[:]
        # aY_fix = readdlm("aY.txt")[:]

        # u0_fix = [aX_fix; aY_fix]

        
        # g = calc_globres(u0_fix,a,U,D,edof,b,free)
        # println("Residual of fix-point solution = $(maximum(abs(g)))")

        # Initial guess
        # We only guess on the unconstrained nodes, the others will not be
        # changed during the newton iterations
        Δan_0 = 1.1*ones(Float64, n_free_dofs);

        # Total solution need to satisfy boundary conditions so we enforce that here
        full_solution[fixed] = bc[:,2]

        df = DifferentiableSparseMultivariateFunction(f!,g!)

        res = nlsolve(df, Δan_0, ftol = 1e-7, iterations=10000, show_trace = true, method = :trust_region)
        if !converged(res)
            error("Global equation did not converge")
        end

        # The total solution after the Newtons step is the previous plus
        # the solution to the Newton iterations.
        full_solution[free] += res.zero
        a[:,modeItr] = full_solution*0.5;
        U.modes += 1
    end
    a[:,2] *= 2.0
    return a, U
end

o = mainNewton()



visualize(o...)










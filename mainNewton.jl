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
    xEnd = 20; yEnd = 20
    xnEl = 20; ynEl = 20
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
    ndofs = maximum(edof)
    free = 1:ndofs
    free = setdiff(free,bc[:,1])

    b = [1.0,1.0] # Body force

    n_modes = 1

    a = rand(ndofs, n_modes)

    function f!(an,fvec) # For NLsolve
        globres = calc_globres(an,a,U,D,edof,b)
        globres[bc[:,1]] = 0.0*bc[:,1]
        copy!(globres,fvec)
    end

    function g!(an,gjac) # For NLsolve
        K = calc_globK(an,a,U,D,edof,b)
        copy!(gjac,K)
    end

    # Mode 1 from fixpoint solution (20x20 elements x ∈ [0,20], y ∈ [0,20])
    aX_fix = readdlm("aX.txt")[:]
    aY_fix = readdlm("aY.txt")[:]

    u0_fix = [aX_fix; aY_fix]


    g = calc_globres(u0_fix,a,U,D,edof,b)
    println("Residual of fix-point solution = $(maximum(abs(g)))")



    # Initial guess
    u0 = 0.1*ones(Float64,ndofs); u0[bc[:,1]] = 0.0
    u = copy(u0)

    for modeItr = 1:n_modes # Mode iterations
        i = 0
        while true; i+=1 # Newton iterations
            g = calc_globres(u,a,U,D,edof,b) # Global residual
            tol = 1e-7
            if maximum(abs(g[free])) < tol
                println("Mode $modeItr converged after $i iterations, g = $(maximum(abs(g)))")
                break
            end
            println("Iteration $i, residual $(maximum(abs(g)))")
            K = calc_globK(u,a,U,D,edof,b)
            Δu = -K[free,free]\g[free]
            u[free] += Δu
        end
        a[:,modeItr] = u
        U.modes = modeItr
        copy!(u,u0)
    end

    return a, U

    # Δu0 = 0.1*ones(Float64,ndofs); Δu0[bc[:,1]] = 0.0
    # u = copy(Δu0)

    # df = DifferentiableSparseMultivariateFunction(f!,g!)

    # g = calc_globres(Δu0,a,U,D,edof,b)

    # println("Norm av residual: $(maximum(abs(g)))")

    # res = nlsolve(df,Δu0, ftol = 1e-7, iterations=20, show_trace = true, method = :newton)
    # if !converged(res)
    #     error("Global equation did not converge")
    # end
    # return 1
    # return u, U
end

o = mainNewton()



visualize(o...)










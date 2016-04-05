import JuAFEM
import CALFEM
using InplaceOps
using ForwardDiff
using ContMechTensors

include("src/meshgenerator.jl")
include("src/PGDmodule.jl")
include("src/utilities.jl")
include("src/elementFunctions.jl")
include("src/globResK.jl")
include("src/visualize.jl")
include("src/solvers.jl")
include("src/boundaryConditions.jl")

function main()
    xStart = 0; yStart = 0
    xEnd = 1; yEnd = 1
    xnEl = 50; ynEl = 50
    xnElNodes = 2; ynElNodes = 2
    xnNodeDofs = 2; ynNodeDofs = 2
    xmesh = create_mesh1D(xStart,xEnd,xnEl,xnElNodes,xnNodeDofs)
    ymesh = create_mesh1D(yStart,yEnd,ynEl,ynElNodes,ynNodeDofs)
    xymesh = create_mesh2D(xStart,xEnd,yStart,yEnd,xnEl,ynEl,2)

    # Set up function components
    function_space = JuAFEM.Lagrange{1,JuAFEM.RefCube,1}()
    q_rule = JuAFEM.GaussQuadrature(JuAFEM.Dim{1},JuAFEM.RefCube(),1)
    fevx = JuAFEM.FEValues(Float64,q_rule,function_space)
    fevy = JuAFEM.FEValues(Float64,q_rule,function_space)

    Ux = PGDComponent(1,xmesh,fevx)
    Uy = PGDComponent(1,ymesh,fevy)

    # Set up PGDfunction
    function_space = JuAFEM.Lagrange{2,JuAFEM.RefCube,1}()
    q_rule = JuAFEM.GaussQuadrature(JuAFEM.Dim{2},JuAFEM.RefCube(),2)
    fevxy = JuAFEM.FEValues(Float64,q_rule,function_space)

    U = PGDFunction(2,2,xymesh,fevxy,[Ux, Uy])
    edof_U = create_edof(U)


    # Material parameters
    E = 1; ν = 0.3
    D = CALFEM.hooke(2,E,ν); D = D[[1,2,4],[1,2,4]]

    # Boundary conditions
    bc_U, bc_Ux, bc_Uy = displacementBC(U)

    ndofs = maximum(edof_U)
    free = setdiff(1:ndofs,bc_U[1])
    free_x = setdiff(1:ndofs,bc_Ux[1])
    free_y = setdiff(1:ndofs,bc_Uy[1])

    n_free = length(free)
    n_free_dofs_x = length(free_x)
    n_free_dofs_y = length(free_y)

    b = [1.0,1.0] # Body force

    n_modes = 5
    aU = zeros(ndofs, n_modes)

    # Displacement solution
    for modeItr = 1:n_modes
        # tic()
        newMode = displacementModeSolver(aU,U,ndofs,bc_U,bc_Ux,bc_Uy,D,edof_U,b,free,free_x,free_y,modeItr)
        aU[:,modeItr] = newMode # Maybe change this to aU = [aU newMode] instead for arbitrary number of modes
        U.modes += 1
        # toc()
    end

    # # Damage solution
    # for modeItr = 1:n_modes
    #     tic()
    #     newMode = damageModeSolver(aU,U,ndofs,bc,bc_x,bc_y,D,edof,b,free,free_x,free_y,modeItr)
    #     aD[:,modeItr] = newMode # Maybe change this to aU = [aU newMode] instead for arbitrary number of modes
    #     D.modes += 1
    #     toc()
    # end

    return aU, U
end

tic()
o = main()
toc()

visualize(o...)

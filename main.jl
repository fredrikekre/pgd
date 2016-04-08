import JuAFEM
import CALFEM
# using InplaceOps
using ForwardDiff
using ContMechTensors
import PyPlot
using WriteVTK

include("src/meshgenerator.jl")
include("src/PGDmodule.jl")
include("src/utilities.jl")
include("src/elementFunctions.jl")
include("src/globResK.jl")
include("src/visualize.jl")
include("src/solvers.jl")
include("src/boundaryConditions.jl")
include("src/vtkwriter.jl")

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
    q_rule = JuAFEM.QuadratureRule(JuAFEM.Dim{1},JuAFEM.RefCube(),1)
    fevx = JuAFEM.FEValues(Float64,q_rule,function_space)
    fevy = JuAFEM.FEValues(Float64,q_rule,function_space)

    Ux = PGDComponent(1,xmesh,fevx)
    Uy = PGDComponent(1,ymesh,fevy)

    # Set up PGDfunction
    function_space = JuAFEM.Lagrange{2,JuAFEM.RefCube,1}()
    q_rule = JuAFEM.QuadratureRule(JuAFEM.Dim{2},JuAFEM.RefCube(),2)
    fevxy = JuAFEM.FEValues(Float64,q_rule,function_space)

    U = PGDFunction(2,2,xymesh,fevxy,[Ux, Uy])
    edof_U = create_edof(U)


    # Material parameters
    E = 1; ν = 0.3
    D = CALFEM.hooke(2,E,ν); D = D[[1,2,4],[1,2,4]]

    # Boundary conditions
    bc_U, dirichletmode = displacementBC(U)
    U.modes += 1
    n_modes = 2
    ndofs = maximum(edof_U)
    aU = [dirichletmode zeros(ndofs, n_modes)]
    # aU = zeros(ndofs, n_modes)
    aU_old = copy(aU)

    # Write output
    pvd = paraview_collection("./resultfiles/result")

    b = [0.0,-0.5] # Body force
    max_displacement = 0.5

    n_loadsteps = 2
    for loadstep in 0:n_loadsteps
        println("Loadstep #$loadstep of $(n_loadsteps)")
        controlled_displacement = max_displacement*(loadstep/n_loadsteps)
        aU[:,1] = sqrt(controlled_displacement) * dirichletmode # Since `dirichletmode` is squared

        # Displacement solution
        for modeItr = 1:n_modes
            # tic()
            newMode = displacementModeSolver(aU,aU_old,U,bc_U,ndofs,D,edof_U,b,modeItr)
            aU[:,modeItr+1] = newMode # Maybe change this to aU = [aU newMode] instead for arbitrary number of modes
            U.modes += 1
            # toc()
        end # of mode iterations


        # # # Damage solution
        # # for modeItr = 1:n_modes
        # #     tic()
        # #     newMode = damageModeSolver(aU,U,ndofs,bc,bc_x,bc_y,D,edof,b,free,free_x,free_y,modeItr)
        # #     aD[:,modeItr] = newMode # Maybe change this to aU = [aU newMode] instead for arbitrary number of modes
        # #     D.modes += 1
        # #     toc()
        # # end

        # Write to file
        vtkwriter(pvd,aU,U,loadstep)
        copy!(aU_old,aU)
        U.modes = 1

    end # of loadstepping
    vtk_save(pvd)
    return aU, U
end

tic()
o = main()
toc()

visualize(o...)

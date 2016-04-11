import JuAFEM
import CALFEM
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
include("src/material_params.jl")

function main()

    ############
    # Geometry #
    ############
    xStart = 0; yStart = 0
    xEnd = 1; yEnd = 1
    xnEl = 50; ynEl = 50


    ###################
    # Displacement, U #
    ###################
    # Mesh
    xnElNodes = 2; ynElNodes = 2
    xnNodeDofs = 2; ynNodeDofs = 2
    xynNodeDofs = 2
    xmesh = create_mesh1D(xStart,xEnd,xnEl,xnElNodes,xnNodeDofs)
    ymesh = create_mesh1D(yStart,yEnd,ynEl,ynElNodes,ynNodeDofs)
    xymesh = create_mesh2D(xStart,xEnd,yStart,yEnd,xnEl,ynEl,xynNodeDofs)

    # Set up the two components
    function_space = JuAFEM.Lagrange{1,JuAFEM.RefCube,1}()
    q_rule = JuAFEM.QuadratureRule(JuAFEM.Dim{1},JuAFEM.RefCube(),1)
    fevx = JuAFEM.FEValues(Float64,q_rule,function_space)
    fevy = JuAFEM.FEValues(Float64,q_rule,function_space)

    Ux = PGDComponent(1,xmesh,fevx)
    Uy = PGDComponent(1,ymesh,fevy)

    # Combine the 2 components
    function_space = JuAFEM.Lagrange{2,JuAFEM.RefCube,1}()
    q_rule = JuAFEM.QuadratureRule(JuAFEM.Dim{2},JuAFEM.RefCube(),2)
    fevxy = JuAFEM.FEValues(Float64,q_rule,function_space)

    U = PGDFunction(2,2,xymesh,fevxy,[Ux, Uy])
    U_edof = create_edof(U,xynNodeDofs)


    #############
    # Damage, D #
    #############
    # Mesh
    xnElNodes = 2; ynElNodes = 2
    xnNodeDofs = 1; ynNodeDofs = 1
    xynNodeDofs = 1
    xmesh = create_mesh1D(xStart,xEnd,xnEl,xnElNodes,xnNodeDofs)
    ymesh = create_mesh1D(yStart,yEnd,ynEl,ynElNodes,ynNodeDofs)
    xymesh = create_mesh2D(xStart,xEnd,yStart,yEnd,xnEl,ynEl,xynNodeDofs)

    # Set up the two components
    function_space = JuAFEM.Lagrange{1,JuAFEM.RefCube,1}()
    q_rule = JuAFEM.QuadratureRule(JuAFEM.Dim{1},JuAFEM.RefCube(),1)
    fevx = JuAFEM.FEValues(Float64,q_rule,function_space)
    fevy = JuAFEM.FEValues(Float64,q_rule,function_space)

    Dx = PGDComponent(1,xmesh,fevx)
    Dy = PGDComponent(1,ymesh,fevy)

    # Combine the 2 components
    function_space = JuAFEM.Lagrange{2,JuAFEM.RefCube,1}()
    q_rule = JuAFEM.QuadratureRule(JuAFEM.Dim{2},JuAFEM.RefCube(),2)
    fevxy = JuAFEM.FEValues(Float64,q_rule,function_space)

    D = PGDFunction(2,2,xymesh,fevxy,[Dx, Dy])
    D_edof = create_edof(D,xynNodeDofs)


    #########################
    # Simulation parameters #
    #########################
    U_n_modes = 5
    D_n_modes = 5

    n_loadsteps = 10
    max_displacement = 0.5

    #######################
    # Material parameters #
    #######################
    E = 1; ν = 0.3
    D_mat = CALFEM.hooke(2,E,ν); D_mat = D_mat[[1,2,4],[1,2,4]]


    #######################
    # Boundary conditions #
    #######################
    U_bc, U_dirichletmode = displacementBC(U)
    U.modes += 1 # Add the first mode as a dirichlet mode
    U_a = [U_dirichletmode repmat(zeros(U_dirichletmode),1,U_n_modes)]
    U_a_old = copy(U_a)

    # D_bc, D_dirichletmode = damageBC(D)
    # U.modes += 1 # Add the first mode as a dirichlet mode
    # D_a = [D_dirichletmode repmat(zeros(D_dirichletmode),1,D_n_modes)]
    # D_a_old = copy(D_a)

    # Body force
    b = [0.0, 0.0]

    ################
    # Write output #
    ################
    pvd = paraview_collection("./resultfiles/result")

    ####################
    # Start simulation #
    ####################
    for loadstep in 0:n_loadsteps
        println("Loadstep #$loadstep of $(n_loadsteps)")
        controlled_displacement = max_displacement*(loadstep/n_loadsteps)
        U_a[:,1] = sqrt(controlled_displacement) * U_dirichletmode # Since `dirichletmode` is squared

        # Displacement solution
        for modeItr = 1:U_n_modes
            # tic()
            newMode = displacementModeSolver(U_a,U_a_old,U,U_bc,D_mat,U_edof,b,modeItr)
            U_a[:,modeItr+1] = newMode # Maybe change this to aU = [aU newMode] instead for arbitrary number of modes
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
        vtkwriter(pvd,U_a,U,loadstep)
        copy!(U_a_old,U_a)r
        U.modes = 1

    end # of loadstepping
    vtk_save(pvd)
    return U_a, U
end

tic()
o = main()
toc()

# visualize(o...)

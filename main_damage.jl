import JuAFEM
import CALFEM
using ForwardDiff
using ContMechTensors
import PyPlot
using WriteVTK

include("src/material_params.jl")
include("src/meshgenerator.jl")
include("src/PGDmodule.jl")
include("src/utilities.jl")
include("src/elementFunctions.jl")
include("src/globResK.jl")
include("src/visualize.jl")
include("src/boundaryConditions.jl")
include("src/solvers.jl")
include("src/vtkwriter.jl")


############################################
# Main file for PGD elasticity with damage #
############################################

function main()

    ############
    # Geometry #
    ############
    xStart = 0; yStart = 0
    xEnd = 1; yEnd = 1
    xnEl = 10; ynEl = 10


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
    U_n_modes = 1
    D_n_modes = 1

    n_loadsteps = 10
    max_displacement = 0.2

    #######################
    # Material parameters #
    #######################
    E = 1.0; ν = 0.3
    U_mp = LinearElastic(:E,E,:ν,ν)
    U_mp_tangent = TangentStiffness(U_mp)

    gc = 0.01
    l = 0.5
    D_mp = PhaseFieldDamage(gc,l)


    #######################
    # Boundary conditions #
    #######################
    U_bc, U_dirichletmode = U_BC(U)
    U.modes += 1 # Add the first mode as a dirichlet mode
    U_a = [U_dirichletmode repmat(zeros(U_dirichletmode),1,U_n_modes)]
    U_a_old = copy(U_a)

    D_bc, D_dirichletmode = D_BC(D)
    D.modes += 1 # Add the first mode as a dirichlet mode
    D_a = [D_dirichletmode repmat(zeros(D_dirichletmode),1,D_n_modes)]
    D_a_old = copy(D_a)

    # Body force
    b = [0.0, -0.5]

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

        # Displacement as function of damage
        for modeItr = 2:(U_n_modes + 1)
            # tic()
            newMode = UD_ModeSolver(U_a,U_a_old,U,U_bc,U_edof,
                                    D_a,D_a_old,D,D_bc,D_edof,
                                    U_mp_tangent,b,modeItr)

            U_a[:,modeItr] = newMode
            U.modes = modeItr
            # toc()
        end # of mode iterations

        # Damage as function of the displacement
        for modeItr = 2:(D_n_modes + 1)
            # tic()
            Ψ = zeros(4)
            newMode = DU_ModeSolver(D_a,D_a_old,D,D_bc,D_edof,
                                    U_a,U_a_old,U,U_bc,U_edof,
                                    D_mp,Ψ,modeItr)
            D_a[:,modeItr] = newMode
            D.modes = modeItr
            # toc()
        end

        # Write to file
        vtkwriter(pvd,U_a,U,D_a,D,loadstep)
        # vtkwriter(pvd,U_a,U,loadstep)
        copy!(U_a_old,U_a)
        copy!(D_a_old,D_a)
        U.modes = 1
        D.modes = 1

    end # of loadstepping
    vtk_save(pvd)
    return U_a, U, D_a, D
end

tic()
o = main()
toc()

visualize(o...)

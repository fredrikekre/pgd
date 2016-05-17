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
include("src/pretty_printing.jl")


################################
# Main file for PGD elasticity #
################################

function main_elastic()

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


    #########################
    # Simulation parameters #
    #########################
    U_n_modes = 5

    n_loadsteps = 1
    max_displacement = 0.1

    #######################
    # Material parameters #
    #######################
    E = 1.0; ν = 0.3
    U_mp = LinearElastic(:E,E,:ν,ν)
    U_mp_tangent = TangentStiffness(U_mp)
    U_mp = LinearElastic_withTangent(U_mp,U_mp_tangent)


    #######################
    # Boundary conditions #
    #######################
    U_bc, U_dirichletmode = U_BC(U)
    U.modes += 1 # Add the first mode as a dirichlet mode
    U_a = [U_dirichletmode 0.1*repmat(ones(U_dirichletmode),1,U_n_modes)]
    U_a_old = copy(U_a)

    # Body force
    b = [0.0, 0.0]

    ################
    # Write output #
    ################
    pvd = paraview_collection("./vtkfiles_elastic/vtkoutfile")

    ####################
    # Start simulation #
    ####################
    for loadstep in 0:n_loadsteps
    print_loadstep(loadstep,n_loadsteps)
        controlled_displacement = max_displacement*(loadstep/n_loadsteps)
        U_a[:,1] = sqrt(controlled_displacement) * U_dirichletmode # Since `dirichletmode` is squared

        # Displacement
        for modeItr = 2:(U_n_modes + 1)
            print_modeitr(modeItr-1,U_n_modes,"displacement")
            newMode = U_ModeSolver(U_a,U_a_old,U,U_bc,U_edof,
                                       U_mp,b,modeItr,loadstep)

            U_a[:,modeItr] = newMode
            U.modes = modeItr
            println("done!")
        end # of mode iterations

        # Write to file
        vtkwriter(pvd,U_a,U,loadstep)
        if loadstep > 0 # Since first loadstep is a 0-mode
            copy!(U_a_old,U_a)
        end
        U.modes = 1

    end # of loadstepping
    vtk_save(pvd)
    return U_a, U
end

@time o = main_elastic()

visualize(o...)

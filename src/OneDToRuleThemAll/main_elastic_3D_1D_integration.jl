using JuAFEM
using ContMechTensors

using TimerOutputs

include("src/meshgenerator.jl")
include("src/PGDmodule.jl")
include("src/elements.jl")
include("src/hadamard.jl")
include("src/material.jl")
include("src/solvers.jl")
include("src/utilities.jl")
include("src/write_to_vtk.jl")
include("src/postprocesser.jl")


################################
# Main file for PGD elasticity #
################################

function main_elastic_3D_1D_integration()
    # const to = TimerOutput()

    ############
    # Geometry #
    ############
    xStart = 0; yStart = 0; zStart = 0
    xEnd = 1.0; yEnd = 1.0; zEnd = 1.0
    xnEl = 100; ynEl = 100; znEl = 100


    ###################
    # Displacement, U #
    ###################
    # Mesh
    xnElNodes = 2; ynElNodes = 2; znElNodes = 2
    xnNodeDofs = 3; ynNodeDofs = 3; znNodeDofs = 3

    xmesh = create_mesh1D(xStart,xEnd,xnEl,xnElNodes,xnNodeDofs)
    ymesh = create_mesh1D(yStart,yEnd,ynEl,ynElNodes,ynNodeDofs)
    zmesh = create_mesh1D(zStart,zEnd,znEl,znElNodes,znNodeDofs)


    # Set up the components
    function_space = Lagrange{1,RefCube,1}()
    q_rule = QuadratureRule(Dim{1},RefCube(),2)
    fevx = FEValues(Float64,q_rule,function_space)
    fevy = FEValues(Float64,q_rule,function_space)
    fevz = FEValues(Float64,q_rule,function_space)

    Ux = PGDComponent(xmesh,fevx,1,3)
    Uy = PGDComponent(ymesh,fevy,2,3)
    Uz = PGDComponent(zmesh,fevz,3,3)

    nxdofs = maximum(xmesh.edof)
    aX = Vector{Float64}[]

    nydofs = maximum(ymesh.edof)
    aY = Vector{Float64}[]

    nzdofs = maximum(zmesh.edof)
    aZ = Vector{Float64}[]

    ############
    # Material #
    ############
    Emod = 1; ν = 0.3
    E = get_E_tensor(Emod, ν, 3)

    #########################
    # Simulation parameters #
    #########################
    n_modes = 20
    n_loadsteps = 1
    TOL = 1e-7
    # max_displacement = 0.1*0.5/4


    #######################
    # Boundary conditions #
    #######################
    xbc = [1:3;(nxdofs-2):nxdofs]
    ybc = [1:3;(nydofs-2):nydofs]
    zbc = [1:3;(nzdofs-2):nzdofs]
    xbc = [1:3;]; #xbc = Int[1, 2, 3, nxdofs-2, nxdofs-1, nxdofs]
    ybc = [1:3;]; #ybc = Int[1, 2, 3, nydofs-2, nydofs-1, nydofs]
    zbc = [1:3;]; #zbc = Int[]

    # aXd = ones(nxdofs); aXd[xbc] = 0.0
    # aYd = ones(nydofs); aYd[ybc] = 0.0
    # aZd = ones(nzdofs); aZd[zbc] = 0.0

    # push!(aX,aXd); push!(aY,aYd); push!(aZ,aZd); push!(Es,E)

    # Dirichlet mode
    aXd = ones(nxdofs)
    aXd[2:3:(end-1)] = 0.0
    aXd[3:3:end] = 0.0

    aYd = ones(nydofs)
    aYd[2:3:(end-1)] = 0.0
    aYd[3:3:end] = 0.0

    aZd = ones(nzdofs)
    aZd[1:3] = 0.0; aZd[(end-2):end] = [1.0,0.0,0.0]
    aZd[1:3:end-2] = linspace(0,1,length(aZd[1:3:end-2]))

    # push!(aX,aXd); push!(aY,aYd); push!(aZ,aZd); push!(Es,E)

    # Body force
    b = Vec{3}((1.0, 1.0, 1.0))

    # ################
    # # Write output #
    # ################
    pvd = paraview_collection("./vtkfiles/vtkoutfile")
    # vtkwriter(pvd,0,Ux,aX,Uy,aY,Uz,aZ)
    # return vtk_save(pvd)
    ####################
    # Start simulation #
    ####################
    for loadstep in 1:n_loadsteps
        # controlled_displacement = max_displacement*(loadstep/n_loadsteps)
        # U_a[:,1] = sqrt(controlled_displacement) * U_dirichletmode # Since `dirichletmode` is squared

        for modeItr = 2:(n_modes + 1)
            iterations = 0

            # Initial guess
            aX0 = 0.1*ones(nxdofs); #aX0[xbc] = 0.0
            push!(aX,aX0)
            aY0 = 0.1*ones(nydofs); #aY0[ybc] = 0.0
            push!(aY,aY0)
            aZ0 = 0.1*ones(nzdofs); #aZ0[zbc] = 0.0
            push!(aZ,aZ0)
            compsold = IterativeFunctionComponents(aX0,aY0,aZ0)

            while true; iterations += 1

                newXmode = mode_solver(Ux,aX,Uy,aY,Uz,aZ,E,b,xbc,Val{1}())
                aX[end] = newXmode

                newYmode = mode_solver(Uy,aY,Uz,aZ,Ux,aX,E,b,ybc,Val{1}())
                aY[end] = newYmode

                newZmode = mode_solver(Uz,aZ,Ux,aX,Uy,aY,E,b,zbc,Val{1}())
                aZ[end] = newZmode

                # println("Done with iteration $(iterations) for mode $(modeItr).")

                compsnew = IterativeFunctionComponents(newXmode,newYmode,newZmode)
                xdiff,ydiff,zdiff = iteration_difference(compsnew,compsold)
                # println("xdiff = $(xdiff), ydiff = $(ydiff), zdiff = $(zdiff)")
                compsold = compsnew

                if (xdiff < TOL && ydiff < TOL && zdiff < TOL) || iterations > 100
                    println("Converged for mode $(modeItr) after $(iterations) iterations.")
                    println("xdiff = $(xdiff), ydiff = $(ydiff), zdiff = $(zdiff)")
                    vtkwriter(pvd,modeItr,Ux,aX,Uy,aY,Uz,aZ)
                    break
                end

            end
        end # of mode iterations

        # Write to file
        # vtkwriter(pvd,1,Ux,aX,Uy,aY,Uz,aZ)
        # if loadstep > 0 # Since first loadstep is a 0-mode
            # copy!(U_a_old,U_a)
        # end
        # U.modes = 1

    end # of loadstepping

    vtk_save(pvd)
    return aX, aY, aZ, Ux, Uy, Uz
end

@time o = main_elastic_3D_1D_integration();

postprocesser_main_elastic_3D_1D_integration(o...)

using JuAFEM
using WriteVTK
using ContMechTensors

include("src/meshgenerator.jl")
include("src/material.jl")
include("src/solvers.jl")
include("src/elements.jl")
include("src/vtk_writer.jl")



function main_elastic_3D()

    ############
    # Geometry #
    ############
    xStart = 0.0; yStart = 0.0; zStart = 0.0
    xs = Tensor{1,3}((xStart, yStart, zStart))

    xEnd = 1.0; yEnd = 1.0; zEnd = 1.0
    xe = Tensor{1,3}((xEnd, yEnd, zEnd))

    xnEl = 10; ynEl = 10; znEl = 10
    nel = [xnEl, ynEl, znEl]

    ########
    # Mesh #
    ########
    mesh = generate_mesh(xs, xe, nel)

    #######################
    # Boundary conditions #
    #######################
    u_prescr = Int[mesh.side_dofs[6][1]; mesh.side_dofs[6][2]]
    u_fixed = [mesh.side_dofs[1][1]; mesh.side_dofs[1][2]; mesh.side_dofs[1][3]; mesh.side_dofs[6][3]]
    u_free = setdiff(1:number_of_dofs(mesh),[u_prescr; u_fixed])

    # Load
    b = Vec{3}((0.0001, 0.0001, 0.0001))

    ###################
    # Function spaces #
    ###################
    function_space = Lagrange{3, JuAFEM.RefCube, 1}()
    quad_rule = QuadratureRule(Dim{3}, RefCube(), 2)
    u_fe_values = FEValues(Float64, quad_rule, function_space)


    ############
    # Material #
    ############
    E = 1; ν = 0.3
    EE = get_E_tensor(E, ν, 3)

    ###############################
    # Set up iteration parameters #
    ###############################
    u = zeros(number_of_dofs(mesh))
    n_loadsteps = 1

    for loadstep in 1:n_loadsteps

        # Prescribed conditions
        u_controlled = 0.5
        u[u_prescr] = u_controlled

        Δu = u_solver(u,u_fe_values,mesh,u_free,EE,b)
        u[u_free] += Δu

        vtk_writer_3D(xs,xe,nel,u)
    end
    return u
end

@time o = main_elastic_3D()

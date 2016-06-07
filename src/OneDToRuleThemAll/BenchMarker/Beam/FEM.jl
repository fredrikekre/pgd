using JuAFEM
using WriteVTK
using ContMechTensors

include("beam_src/meshgenerator.jl")
include("beam_src/material.jl")
include("beam_src/solvers.jl")
include("beam_src/elements.jl")
include("beam_src/vtk_writer.jl")

function bench_beam_FEM(sidenel::Int)
    t_FEM = 0.0

    ############
    # Geometry #
    ############
    xStart = 0.0; yStart = 0.0; zStart = 0.0
    xs = Tensor{1,3}((xStart, yStart, zStart))

    xEnd = 1.0; yEnd = 0.1; zEnd = 0.1
    xe = Tensor{1,3}((xEnd, yEnd, zEnd))

    xnEl = 10*sidenel; ynEl = sidenel; znEl = sidenel
    nel = [xnEl, ynEl, znEl]

    ########
    # Mesh #
    ########
    mesh = generate_mesh(xs, xe, nel)

    #######################
    # Boundary conditions #
    #######################
    u_prescr = Int[]
    u_fixed = [mesh.side_dofs[5][1]; mesh.side_dofs[5][2]; mesh.side_dofs[5][3]]
    u_free = setdiff(1:number_of_dofs(mesh),[u_prescr; u_fixed])

    # Load
    b = Vec{3}((0.0, 0.0, -1.0))

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
        u_controlled = 0.1
        u[u_prescr] = u_controlled

        tic()
        Δu = u_solver(u,u_fe_values,mesh,u_free,EE,b)
        u[u_free] += Δu
        t_FEM += toq();

        vtk_writer_3D(xs,xe,nel,u)
    end

    uu = u[1:3:(end-2)]
    uu = reshape(uu,(xnEl+1, ynEl+1, znEl +1))

    vv = u[2:3:(end-1)]
    vv = reshape(vv,(xnEl+1, ynEl+1, znEl +1))

    ww = u[3:3:(end-0)]
    ww = reshape(ww,(xnEl+1, ynEl+1, znEl +1))

    return uu, vv, ww, t_FEM
end

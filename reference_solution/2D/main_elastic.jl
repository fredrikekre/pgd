using JuAFEM
using ForwardDiff
using WriteVTK
# using CALFEM

include("../../src/material_params.jl")
include("../../src/meshgenerator.jl")
include("src/solvers.jl")
include("src/gandK.jl")
include("src/element_functions.jl")
include("src/vtkwriter.jl")

function main()

    # Write output
    pvd = paraview_collection("./vtkfiles_elastic/vtkoutfile")

    # Problem parameters
    xStart = 0.0; xEnd = 1.0; nElx = 100
    yStart = 0.0; yEnd = 1.0; nEly = 100
    u_nNodeDofs = 2
    u_mesh = create_mesh2D(xStart,xEnd,yStart,yEnd,nElx,nEly,u_nNodeDofs)
    b = zeros(2)

    # Function spaces
    function_space = Lagrange{2, JuAFEM.RefCube, 1}()
    quad_rule = QuadratureRule(Dim{2}, RefCube(), 2)
    u_fe_values = FEValues(Float64, quad_rule, function_space)

    # Material
    E = 1; ν = 0.3;
    u_mp = LinearElastic(:E,E,:ν,ν)

    # Boundary conditions
    u_prescr = u_mesh.b3[2,:][:]
    u_fixed = [u_mesh.b1[1,:][:]; u_mesh.b1[2,:][:]; u_mesh.b3[1,:][:]]
    u_free = setdiff(1:u_mesh.nDofs,[u_prescr; u_fixed])


    nTimeSteps = 100
    u_prescr_max = 0.1*0.5/4

    u = zeros(u_mesh.nDofs)

    for i in 1:nTimeSteps
        println("Timestep $i of $nTimeSteps")
        u_prescribed_value = (i-1)*u_prescr_max/nTimeSteps

        u[u_prescr] = u_prescribed_value

        # Solve for displacements
        Δu = U_solver(u,u_mesh,u_free,u_fe_values,u_mp,b)
        u[u_free] += Δu

        # Write to VTK
        vtkwriter(pvd,i,u_mesh,u)
    end
    vtk_save(pvd)
    return u
end

@time o = main()

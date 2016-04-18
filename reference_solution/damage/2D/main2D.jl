using JuAFEM
using ForwardDiff
using WriteVTK
# using CALFEM

include("src/phasefield.jl")
include("../../../src/meshgenerator.jl")
include("src/solvers.jl")
include("src/gandK.jl")
include("src/vtkwriter.jl")

function main()

    # Write output
    pvd = paraview_collection("./vtkfiles/vtkoutfile")

    # Problem parameters
    xStart = 0.0; xEnd = 1.0; nElx = 10
    yStart = 0.0; yEnd = 1.0; nEly = 10
    u_nNodeDofs = 2; d_nNodeDofs = 1
    u_mesh = create_mesh2D(xStart,xEnd,yStart,yEnd,nElx,nEly,u_nNodeDofs)
    d_mesh = create_mesh2D(xStart,xEnd,yStart,yEnd,nElx,nEly,d_nNodeDofs)
    b = zeros(2)

    # Function spaces
    function_space = Lagrange{2, JuAFEM.RefCube, 1}()
    quad_rule = QuadratureRule(Dim{2}, RefCube(), 2)
    u_fe_values = FEValues(Float64, quad_rule, function_space)
    d_fe_values = FEValues(Float64, quad_rule, function_space)

    # Material
    u_mp = LEmtrl()
    # u_mp = LEmtrl(1.0,0.3)
    Ψ = [zeros(length(JuAFEM.points(quad_rule))) for i in 1:u_mesh.nEl]


    d_mp = DamageParams()

    # Boundary conditions
    u_prescr = u_mesh.b3[2,:][:]
    u_fixed = [u_mesh.b1[1,1]; u_mesh.b1[2,:][:]]
    u_free = setdiff(1:u_mesh.nDofs,[u_prescr; u_fixed])

    # d_prescr = [collect((101*49 + 1):(101*49 + 41));
    #             collect((101*50 + 1):(101*50 + 41));
    #             collect((101*51 + 1):(101*51 + 41))]
    d_prescr = collect(((nElx+1)*div(nEly,2) + 1):((nElx+1)*div(nEly,2) + Int(round(nElx/2))))
    # d_prescr = []

    d_fixed = []
    d_free = setdiff(1:d_mesh.nDofs,[d_prescr; d_fixed])

    nTimeSteps = 100
    u_prescr_max = 0.1*0.5

    u = zeros(u_mesh.nDofs)
    d = zeros(d_mesh.nDofs)

    for i in 1:nTimeSteps
    # tic()
        println("Timestep $i of $nTimeSteps")
        u_prescribed_value = (i-1)*u_prescr_max/nTimeSteps
        d_prescribed_value = 1.0

        u[u_prescr] = u_prescribed_value
        d[d_prescr] = d_prescribed_value

        # Solve for displacements
        # Δu = solveDisplacementField(u,u_mesh,u_free,u_fe_values,d,d_mesh,d_fe_values,u_mp,b)
        # u[u_free] += Δu
        for j = 1:3 # Do some iterations

            # Solve for displacements
            Δu = solveDisplacementField(u,u_mesh,u_free,u_fe_values,d,d_mesh,d_fe_values,u_mp,b)
            u[u_free] += Δu

            # Calculate free energy
            Ψ, _, _ = calculateFreeEnergy(u,u_mesh,u_free,u_fe_values,d,d_mesh,d_fe_values,u_mp,Ψ)

            # Solve for damage field
            Δd = solveDamageField(d,d_mesh,d_free,d_fe_values,u,u_mesh,u_fe_values,d_mp,Ψ)
            d[d_free] += Δd

        end

        _, Ψ_plot, σe_plot = calculateFreeEnergy(u,u_mesh,u_free,u_fe_values,d,d_mesh,d_fe_values,u_mp,Ψ)

        # Write to VTK
        # toc()
        # tic()
        vtkwriter(pvd,i,u_mesh,u,d,Ψ_plot,σe_plot)
        # toc()
    end
    vtk_save(pvd)
    return u, d, Ψ
end

o = main()

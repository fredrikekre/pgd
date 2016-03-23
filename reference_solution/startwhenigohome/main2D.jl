using JuAFEM
using ForwardDiff
using WriteVTK

include("phasefield.jl")
include("../../src/meshgenerator.jl")
include("solvers.jl")
include("gandK.jl")
include("vtkwriter.jl")

function main()

    # Write output
    pvd = paraview_collection("./vtkfiles/vtkoutfile")

    # Problem parameters
    xStart = 0.0; xEnd = 0.001; nElx = 150
    yStart = 0.0; yEnd = 0.001; nEly = 150
    u_nNodeDofs = 2; d_nNodeDofs = 1
    u_mesh = create_mesh2D(xStart,xEnd,yStart,yEnd,nElx,nEly,u_nNodeDofs)
    d_mesh = create_mesh2D(xStart,xEnd,yStart,yEnd,nElx,nEly,d_nNodeDofs)
    b = zeros(2)

    # Function spaces
    function_space = Lagrange{2, JuAFEM.Square, 1}()
    quad_rule = get_gaussrule(Dim{2}, JuAFEM.Square(), 2)
    u_fe_values = FEValues(Float64, quad_rule, function_space)
    d_fe_values = FEValues(Float64, quad_rule, function_space)

    # Material
    u_mp = LEmtrl()
    # u_mp = LEmtrl(1.0,0.3)
    Ψ = [zeros(length(JuAFEM.points(quad_rule))) for i in 1:u_mesh.nEl]

    # brokenEls = [collect((49*100 + 1):(49*100 + 50));
    #              collect((50*100 + 1):(50*100 + 50))]

    # for i in brokenEls
    #     Ψ[i] += [1.0,1.0,1.0,1.0]*10
    # end

    d_mp = DamageParams()

    # Boundary conditions
    u_prescr = u_mesh.b3[1,:][:]
    u_fixed = [u_mesh.b1[1,:][:]; u_mesh.b1[2,:][:]; u_mesh.b2[2,:][:]; u_mesh.b3[2,:][:]; u_mesh.b4[2,:][:]]
    u_free = setdiff(1:u_mesh.nDofs,[u_prescr; u_fixed])

    # d_prescr = [collect((101*49 + 1):(101*49 + 51));
    #             collect((101*50 + 1):(101*50 + 51));
    #             collect((101*51 + 1):(101*51 + 51))]
    d_prescr = collect((151*75 + 1):(151*75 + 75))
    # d_prescr = []

    d_fixed = []
    d_free = setdiff(1:d_mesh.nDofs,[d_prescr; d_fixed])

    nTimeSteps = 1000
    u_prescr_max = 11e-6

    u = zeros(u_mesh.nDofs)
    d = zeros(d_mesh.nDofs)

    for i in 1:nTimeSteps
        println("Timestep $i of $nTimeSteps")
        u_prescribed_value = (i-1)*u_prescr_max/nTimeSteps
        d_prescribed_value = 1.0

        u[u_prescr] = u_prescribed_value
        d[d_prescr] = d_prescribed_value

        # Solve for displacements
        # Δu = solveDisplacementField(u,u_mesh,u_free,u_fe_values,d,d_mesh,d_fe_values,u_mp,b)
        # u[u_free] += Δu
        for j = 1:3
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
        # Ψ_plot = reshape(Ψ_plot,(50,50))
        # σe_plot = reshape(σe_plot,(50,50))

        # Write to VTK
        vtkwriter(pvd,i,u_mesh,u,d,Ψ_plot,σe_plot)
    end
    vtk_save(pvd)
    return u, d, Ψ
end

o = main()

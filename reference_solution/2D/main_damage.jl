using JuAFEM
using ForwardDiff
using WriteVTK

include("../../src/material_params.jl")
include("../../src/meshgenerator.jl")
include("src/solvers.jl")
include("src/gandK.jl")
include("src/element_functions.jl")
include("src/vtkwriter.jl")
include("../../src/pretty_printing.jl")

function main_damage()

    # Write output
    pvd = paraview_collection("./vtkfiles_damage/vtkoutfile")

    # Problem parameters
    xStart = 0.0; xEnd = 1.0; nElx = 50
    yStart = 0.0; yEnd = 1.0; nEly = 50
    u_nNodeDofs = 2; d_nNodeDofs = 1
    u_mesh = create_mesh2D(xStart,xEnd,yStart,yEnd,nElx,nEly,u_nNodeDofs)
    d_mesh = create_mesh2D(xStart,xEnd,yStart,yEnd,nElx,nEly,d_nNodeDofs)
    b = [0.0, 0.0]

    # Function spaces
    function_space = Lagrange{2, JuAFEM.RefCube, 1}()
    quad_rule = QuadratureRule(Dim{2}, RefCube(), 2)
    u_fe_values = FEValues(Float64, quad_rule, function_space)
    d_fe_values = FEValues(Float64, quad_rule, function_space)

    # Material
    E = 1; ν = 0.3
    u_mp = LinearElastic(:E,E,:ν,ν)

    Ψ = [zeros(length(JuAFEM.points(quad_rule))) for i in 1:u_mesh.nEl]

    damagedElementsBelow = collect((nElx*div(nEly,2)+1):(nElx*(div(nEly,2))+div(nElx,2)))
    damagedElementsAbove = collect((nElx*(div(nEly,2)+1)+1):(nElx*(div(nEly,2)+1)+div(nElx,2)))

    Ψ_d = 0.01*100
    Ψ[damagedElementsBelow] = [Ψ_d*[0.0,0.0,1.0,1.0] for i in 1:length(damagedElementsBelow)]
    Ψ[damagedElementsAbove] = [Ψ_d*[1.0,1.0,0.0,0.0] for i in 1:length(damagedElementsAbove)]

    gc = 0.01/100
    l = 0.1
    d_mp = PhaseFieldDamage(gc,l)

    # Boundary conditions
    u_prescr = u_mesh.b3[1,:][:]
    u_fixed = [u_mesh.b1[1,:][:]; u_mesh.b1[2,:][:]; u_mesh.b3[2,:][:]]
    u_free = setdiff(1:u_mesh.nDofs,[u_prescr; u_fixed])

    # d_prescr = [collect((101*49 + 1):(101*49 + 41));
    #             collect((101*50 + 1):(101*50 + 41));
    #             collect((101*51 + 1):(101*51 + 41))]
    # d_prescr = collect(((nElx+1)*div(nEly,2) + 1):((nElx+1)*div(nEly,2) + Int(ceil(nElx/2))))
    d_prescr = []

    d_fixed = []
    d_free = setdiff(1:d_mesh.nDofs,[d_prescr; d_fixed])

    n_loadsteps = 40
    u_prescr_max = 0.1

    u = zeros(u_mesh.nDofs)
    d = zeros(d_mesh.nDofs)

    for loadstep in 0:n_loadsteps
        print_loadstep(loadstep,n_loadsteps)
        u_prescribed_value = u_prescr_max*(loadstep/n_loadsteps)
        d_prescribed_value = 1.0

        u[u_prescr] = u_prescribed_value
        d[d_prescr] = d_prescribed_value

        for j in 1:3 # Do some iterations
            # Solve for displacements
            Δu, Ψ_new = UD_solver(u,u_mesh,u_free,u_fe_values,d,d_mesh,d_fe_values,u_mp,b)
            u[u_free] += Δu

            # Calculate free energy
            for ele in 1:length(Ψ)
                Ψ[ele] = max(Ψ[ele],Ψ_new[ele])
            end

            # Solve for damage field
            Δd = DU_solver(d,d_mesh,d_free,d_fe_values,d_mp,Ψ)
            d[d_free] += Δd

        end

        # if loadstep == 89
        #     elem = 1
        #     return Ψ[elem], d[d_mesh.edof[:,elem]]
        # end

        # Write to VTK
        vtkwriter(pvd,loadstep,u_mesh,u,d)
    end
    vtk_save(pvd)
    return u, d, Ψ
end

@time o = main_damage()

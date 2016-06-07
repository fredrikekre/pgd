using JuAFEM
using ContMechTensors

include("src/meshgenerator.jl")
include("src/material.jl")
include("src/PGDmodule.jl")
include("src/element_header.jl")
include("src/du_elements.jl")
include("src/ud_elements.jl")
include("src/hadamard.jl")
include("src/ud_solvers.jl")
include("src/du_solvers.jl")
include("src/utilities.jl")
include("src/write_to_vtk.jl")
include("src/postprocesser.jl")


################################
# Main file for PGD elasticity #
################################

function main_elastic_3D_3D_integration()

    ############
    # Geometry #
    ############
    xStart = 0; yStart = 0; zStart = 0
    xEnd = 1.0; yEnd = 1.0; zEnd = 1.0
    xnEl = 10; ynEl = 10; znEl = 10


    ###################
    # Displacement, U #
    ###################
    # Mesh
    u_xnElNodes = 2; u_ynElNodes = 2; u_znElNodes = 2
    u_xnNodeDofs = 3; u_ynNodeDofs = 3; u_znNodeDofs = 3

    u_xmesh = create_mesh1D(xStart,xEnd,xnEl,u_xnElNodes,u_xnNodeDofs)
    u_ymesh = create_mesh1D(yStart,yEnd,ynEl,u_ynElNodes,u_ynNodeDofs)
    u_zmesh = create_mesh1D(zStart,zEnd,znEl,u_znElNodes,u_znNodeDofs)

    # Set up the components
    u_function_space = Lagrange{1,RefCube,1}()
    u_q_rule = QuadratureRule(Dim{1},RefCube(),2)
    u_fevx = FEValues(Float64,u_q_rule,u_function_space)
    u_fevy = FEValues(Float64,u_q_rule,u_function_space)
    u_fevz = FEValues(Float64,u_q_rule,u_function_space)

    Ux = PGDComponent(u_xmesh,u_fevx,1,3)
    Uy = PGDComponent(u_ymesh,u_fevy,2,3)
    Uz = PGDComponent(u_zmesh,u_fevz,3,3)

    u_nxdofs = maximum(u_xmesh.edof)
    u_ax = Vector{Float64}[]

    u_nydofs = maximum(u_ymesh.edof)
    u_ay = Vector{Float64}[]

    u_nzdofs = maximum(u_zmesh.edof)
    u_az = Vector{Float64}[]

    #############
    # Damage, D #
    #############
    # Mesh
    d_xnElNodes = 2; d_ynElNodes = 2; d_znElNodes = 2
    d_xnNodeDofs = 1; d_ynNodeDofs = 1; d_znNodeDofs = 1

    d_xmesh = create_mesh1D(xStart,xEnd,xnEl,d_xnElNodes,d_xnNodeDofs)
    d_ymesh = create_mesh1D(yStart,yEnd,ynEl,d_ynElNodes,d_ynNodeDofs)
    d_zmesh = create_mesh1D(zStart,zEnd,znEl,d_znElNodes,d_znNodeDofs)

    # Set up the components
    d_function_space = Lagrange{1,RefCube,1}()
    d_q_rule = QuadratureRule(Dim{1},RefCube(),2)
    d_fevx = FEValues(Float64,d_q_rule,d_function_space)
    d_fevy = FEValues(Float64,d_q_rule,d_function_space)
    d_fevz = FEValues(Float64,d_q_rule,d_function_space)

    Dx = PGDComponent(d_xmesh,d_fevx,1,3)
    Dy = PGDComponent(d_ymesh,d_fevy,2,3)
    Dz = PGDComponent(d_zmesh,d_fevz,3,3)

    d_nxdofs = maximum(d_xmesh.edof)
    d_ax = Vector{Float64}[]

    d_nydofs = maximum(d_ymesh.edof)
    d_ay = Vector{Float64}[]

    d_nzdofs = maximum(d_zmesh.edof)
    d_az = Vector{Float64}[]

    ###################
    # Global FEValues #
    ###################
    global_function_space = Lagrange{3,RefCube,1}()
    global_q_rule = QuadratureRule(Dim{3},RefCube(),2)
    global_u_fe_values = FEValues(Float64,global_q_rule,global_function_space)
    global_d_fe_values = FEValues(Float64,global_q_rule,global_function_space)

    ##########
    # Energy #
    ##########
    n_global_els = u_xmesh.nEl*u_ymesh.nEl*u_zmesh.nEl
    Ψ = [zeros(Float64,length(points(global_q_rule))) for i in 1:n_global_els]
    Ψ_new = copy(Ψ)

    ############
    # Material #
    ############
    Emod = 1; ν = 0.3
    E = get_E_tensor(Emod, ν, 3)

    l = 0.01; Gc = 0.1
    dmp = damage_params(l,Gc)

    #########################
    # Simulation parameters #
    #########################
    u_n_modes = 5
    d_n_modes = 5
    n_loadsteps = 1
    TOL = 1e-7
    # max_displacement = 0.1*0.5/4


    #######################
    # Boundary conditions #
    #######################
    u_xbc = [1:3;(u_nxdofs-2):u_nxdofs]
    u_ybc = [1:3;(u_nydofs-2):u_nydofs]
    u_zbc = [1:3;(u_nzdofs-2):u_nzdofs]
    u_xbc = [1:3;];
    u_ybc = [1:3;];
    u_zbc = [1:3;];

    # Body force
    b = Vec{3,Float64,3}((1.0, 1.0, 1.0))

    d_xbc = Int[]; d_ybc = Int[]; d_zbc = Int[]

    ######################
    # Set up old vectors #
    ######################
    u_ax0 = 0.1*ones(u_nxdofs); u_ax0[u_xbc] = 0.0
    u_ax_old = [u_ax0 for i in 1:(u_n_modes+1)]
    u_ay0 = 0.1*ones(u_nydofs); u_ay0[u_ybc] = 0.0
    u_ay_old = [u_ay0 for i in 1:(u_n_modes+1)]
    u_az0 = 0.1*ones(u_nzdofs); u_az0[u_zbc] = 0.0
    u_az_old = [u_az0 for i in 1:(u_n_modes+1)]

    d_ax_old = [0.1*ones(d_nxdofs) for i in 1:d_n_modes]
    d_ay_old = [0.1*ones(d_nydofs) for i in 1:d_n_modes]
    d_az_old = [0.1*ones(d_nzdofs) for i in 1:d_n_modes]

    ###########################################################
    # Shape functions (Setting them up here once and for all) #
    ###########################################################
    UN, global_u_fe_values = get_u_N_3D(Ux, Uy, Uz, global_u_fe_values)
    DN, global_d_fe_values = get_d_N_3D(Dx, Dy, Dz, global_d_fe_values)

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
        u_ax_Dirichlet = zeros(u_nxdofs)
        u_ay_Dirichlet = zeros(u_nydofs)
        u_az_Dirichlet = zeros(u_nzdofs)

        # Reset solution vectors
        u_ax = Vector{Float64}[]; push!(u_ax,u_ax_Dirichlet)
        u_ay = Vector{Float64}[]; push!(u_ay,u_ay_Dirichlet)
        u_az = Vector{Float64}[]; push!(u_az,u_az_Dirichlet)
        d_ax = Vector{Float64}[]
        d_ay = Vector{Float64}[]
        d_az = Vector{Float64}[]

        for j in 1:1 # Staggered iterations

            ########################
            # Solving displacement #
            ########################
            for modeItr = 2:(u_n_modes + 1)
                iterations = 0

                # Initial guess
                push!(u_ax,u_ax_old[modeItr])
                push!(u_ay,u_ay_old[modeItr])
                push!(u_az,u_az_old[modeItr])
                u_compsold = IterativeFunctionComponents(u_ax_old[modeItr],u_ay_old[modeItr],u_az_old[modeItr])

                while true; iterations += 1

                    u_ax_new, Ψ_new = ud_x_mode_solver(Ux,u_ax,Uy,u_ay,Uz,u_az,UN,global_u_fe_values,
                                                       E,b,u_xbc,
                                                       Dx,d_ax,Dy,d_ay,Dz,d_az,DN,
                                                       Ψ_new)
                    u_ax[end] = u_ax_new

                    u_ay_new, Ψ_new = ud_y_mode_solver(Ux,u_ax,Uy,u_ay,Uz,u_az,UN,global_u_fe_values,
                                                       E,b,u_ybc,
                                                       Dx,d_ax,Dy,d_ay,Dz,d_az,DN,
                                                       Ψ_new)
                    u_ay[end] = u_ay_new

                    u_az_new, Ψ_new = ud_z_mode_solver(Ux,u_ax,Uy,u_ay,Uz,u_az,UN,global_u_fe_values,
                                                       E,b,u_zbc,
                                                       Dx,d_ax,Dy,d_ay,Dz,d_az,DN,
                                                       Ψ_new)
                    u_az[end] = u_az_new

                    # println("Done with iteration $(iterations) for mode $(modeItr).")

                    u_compsnew = IterativeFunctionComponents(u_ax_new,u_ay_new,u_az_new)
                    u_xdiff, u_ydiff, u_zdiff = iteration_difference(u_compsnew,u_compsold)
                    # println("u_xdiff = $(u_xdiff), u_ydiff = $(u_ydiff), u_zdiff = $(u_zdiff)")
                    u_compsold = u_compsnew

                    if (u_xdiff < TOL && u_ydiff < TOL && u_zdiff < TOL) || iterations > 100
                        println("Converged for u mode $(modeItr) after $(iterations) iterations.")
                        println("u_xdiff = $(u_xdiff), u_ydiff = $(u_ydiff), u_zdiff = $(u_zdiff)")
                        # vtkwriter(pvd,modeItr,Ux,u_ax,Uy,u_ay,Uz,u_az)
                        break
                    end

                end
            end # of mode iterations
            u_ax_old = copy(u_ax)
            u_ay_old = copy(u_ay)
            u_az_old = copy(u_az)

            ########################
            # Calculate max energy #
            ########################
            for el in 1:length(Ψ)
                Ψ[el] = max(Ψ[el],Ψ_new[el])
            end

            ##################
            # Solving damage #
            ##################
            for modeItr = 1:d_n_modes
                iterations = 0

                # Initial guess
                push!(d_ax,d_ax_old[modeItr])
                push!(d_ay,d_ay_old[modeItr])
                push!(d_az,d_az_old[modeItr])
                d_compsold = IterativeFunctionComponents(d_ax_old[modeItr],d_ay_old[modeItr],d_az_old[modeItr])

                while true; iterations += 1

                    d_ax_new = du_x_mode_solver(Dx,d_ax,Dy,d_ay,Dz,d_az,DN,global_d_fe_values,dmp,d_xbc,Ψ)
                    d_ax[end] = d_ax_new

                    d_ay_new = du_y_mode_solver(Dx,d_ax,Dy,d_ay,Dz,d_az,DN,global_d_fe_values,dmp,d_ybc,Ψ)
                    d_ay[end] = d_ay_new

                    d_az_new = du_z_mode_solver(Dx,d_ax,Dy,d_ay,Dz,d_az,DN,global_d_fe_values,dmp,d_zbc,Ψ)
                    d_az[end] = d_az_new

                    # println("Done with iteration $(iterations) for mode $(modeItr).")

                    d_compsnew = IterativeFunctionComponents(d_ax_new,d_ay_new,d_az_new)
                    d_xdiff, d_ydiff, d_zdiff = iteration_difference(d_compsnew,d_compsold)
                    # println("d_xdiff = $(d_xdiff), d_ydiff = $(d_ydiff), d_zdiff = $(d_zdiff)")
                    d_compsold = d_compsnew

                    if (d_xdiff < TOL && d_ydiff < TOL && d_zdiff < TOL) || iterations > 100
                        println("Converged for d mode $(modeItr) after $(iterations) iterations.")
                        println("d_xdiff = $(d_xdiff), d_ydiff = $(d_ydiff), d_zdiff = $(d_zdiff)")
                        break
                    end

                end
            end # of mode iterations
            d_ax_old = copy(d_ax)
            d_ay_old = copy(d_ay)
            d_az_old = copy(d_az)
        end

        # Write to file
        vtkwriter(pvd,loadstep,Ux,u_ax,Uy,u_ay,Uz,u_az,Dx,d_ax,Dy,d_ay,Dz,d_az)

    end # of loadstepping

    vtk_save(pvd)
    return Ux,u_ax,Uy,u_ay,Uz,u_az,Dx,d_ax,Dy,d_ay,Dz,d_az
end

@time o = main_elastic_3D_3D_integration();

# postprocesser_main_elastic_3D_3D_integration(o...)

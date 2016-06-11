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

function main_damage_2D_2D_integration()

    ############
    # Geometry #
    ############
    xStart = 0; yStart = 0
    xEnd = 1.0; yEnd = 1.0
    xnEl = 50; ynEl = 50


    ###################
    # Displacement, U #
    ###################
    # Mesh
    u_xnElNodes = 2; u_ynElNodes = 2
    u_xnNodeDofs = 2; u_ynNodeDofs = 2

    u_xmesh = create_mesh1D(xStart,xEnd,xnEl,u_xnElNodes,u_xnNodeDofs)
    u_ymesh = create_mesh1D(yStart,yEnd,ynEl,u_ynElNodes,u_ynNodeDofs)

    # Set up the components
    u_function_space = Lagrange{1,RefCube,1}()
    u_q_rule = QuadratureRule(Dim{1},RefCube(),2)
    u_fevx = FEValues(Float64,u_q_rule,u_function_space)
    u_fevy = FEValues(Float64,u_q_rule,u_function_space)

    Ux = PGDComponent(u_xmesh,u_fevx,1,2)
    Uy = PGDComponent(u_ymesh,u_fevy,2,2)

    u_nxdofs = maximum(u_xmesh.edof)
    u_ax = Vector{Float64}[]

    u_nydofs = maximum(u_ymesh.edof)
    u_ay = Vector{Float64}[]


    #############
    # Damage, D #
    #############
    # Mesh
    d_xnElNodes = 2; d_ynElNodes = 2
    d_xnNodeDofs = 1; d_ynNodeDofs = 1

    d_xmesh = create_mesh1D(xStart,xEnd,xnEl,d_xnElNodes,d_xnNodeDofs)
    d_ymesh = create_mesh1D(yStart,yEnd,ynEl,d_ynElNodes,d_ynNodeDofs)

    # Set up the components
    d_function_space = Lagrange{1,RefCube,1}()
    d_q_rule = QuadratureRule(Dim{1},RefCube(),2)
    d_fevx = FEValues(Float64,d_q_rule,d_function_space)
    d_fevy = FEValues(Float64,d_q_rule,d_function_space)

    Dx = PGDComponent(d_xmesh,d_fevx,1,2)
    Dy = PGDComponent(d_ymesh,d_fevy,2,2)

    d_nxdofs = maximum(d_xmesh.edof)
    d_ax = Vector{Float64}[]

    d_nydofs = maximum(d_ymesh.edof)
    d_ay = Vector{Float64}[]


    ###################
    # Global FEValues #
    ###################
    global_function_space = Lagrange{2,RefCube,1}()
    global_q_rule = QuadratureRule(Dim{2},RefCube(),2)
    global_u_fe_values = FEValues(Float64,global_q_rule,global_function_space)
    global_d_fe_values = FEValues(Float64,global_q_rule,global_function_space)

    ##########
    # Energy #
    ##########
    n_global_els = u_xmesh.nEl*u_ymesh.nEl
    Ψ = [zeros(Float64,length(points(global_q_rule))) for i in 1:n_global_els]
    Ψ_ref = 0.01*100

    # 50x50
    for el in 25:50:(25 + 50*16)
        Ψ[el] = Ψ_ref * [0.0, 1.0, 0.0, 1.0]
        Ψ[el+1] = Ψ_ref * [1.0, 0.0, 1.0, 0.0]
    end
    Ψ_new = copy(Ψ)

    ############
    # Material #
    ############
    Emod = 1; ν = 0.3
    E = get_E_tensor(Emod, ν, 2)

    l = 0.1; Gc = 0.01/100
    dmp = damage_params(l,Gc)

    #########################
    # Simulation parameters #
    #########################
    u_n_modes = 5
    d_n_modes = 5
    n_loadsteps = 40
    TOL = 1e-7
    max_displacement = 0.035


    #######################
    # Boundary conditions #
    #######################
    # Non-homogeneous Dirichlet mode for u
    # # Some type of shear
    # u_ax_Dirichlet = ones(u_nxdofs)
    # u_ay_Dirichlet = ones(u_nydofs)
    # u_az_Dirichlet = ones(u_nzdofs)
    # u_ax_Dirichlet[3:3:end] = 0.0
    # u_ay_Dirichlet[3:3:end] = 0.0
    # u_az_Dirichlet[1:3:(end-2)] = reinterpret(Float64,Uz.mesh.x,(length(Uz.mesh.x),))
    # u_az_Dirichlet[2:3:(end-1)] = reinterpret(Float64,Uz.mesh.x,(length(Uz.mesh.x),))
    # u_az_Dirichlet[3:3:(end-0)] = 0.0

    # Pulling up
    u_ax_Dirichlet = zeros(u_nxdofs)
    u_ay_Dirichlet = zeros(u_nydofs)
    u_ax_Dirichlet[2:2:(end-0)] = 1.0
    u_ay_Dirichlet[2:2:(end-0)] = reinterpret(Float64,Uy.mesh.x,(length(Uy.mesh.x),))

    # Homogeneous Dirichlet for u
    u_xbc = Int[]
    u_ybc = Int[1:2; (u_nydofs-1):u_nydofs]

    # Body force
    b = Vec{2,Float64,3}((0.00001, 0.00001))

    d_xbc = Int[]; d_ybc = Int[]

    ######################
    # Set up old vectors #
    ######################
    u_ax0 = 0.1*ones(u_nxdofs); u_ax0[u_xbc] = 0.0
    u_ax_old = [u_ax0 for i in 1:(u_n_modes+1)]
    u_ay0 = 0.1*ones(u_nydofs); u_ay0[u_ybc] = 0.0
    u_ay_old = [u_ay0 for i in 1:(u_n_modes+1)]

    d_ax_old = [0.1*ones(d_nxdofs) for i in 1:d_n_modes]
    d_ay_old = [0.1*ones(d_nydofs) for i in 1:d_n_modes]

    ###########################################################
    # Shape functions (Setting them up here once and for all) #
    ###########################################################
    UN, global_u_fe_values = get_u_N_2D(Ux, Uy, global_u_fe_values)
    DN, global_d_fe_values = get_d_N_2D(Dx, Dy, global_d_fe_values)

    # ################
    # # Write output #
    # ################
    pvd = paraview_collection("./vtkfiles/vtkoutfile")
    # u_ax = Vector{Float64}[]; push!(u_ax,u_ax_Dirichlet*(0.5)^(1/2))
    # u_ay = Vector{Float64}[]; push!(u_ay,u_ay_Dirichlet*(0.5)^(1/2))
    # d_ax = Vector{Float64}[]
    # d_ay = Vector{Float64}[]
    # vtkwriter(pvd,0,Ux,u_ax,Uy,u_ay,Dx,d_ax,Dy,d_ay)
    # return vtk_save(pvd)

    ####################
    # Start simulation #
    ####################
    for loadstep in 0:n_loadsteps
        println("Starting loadstep #$(loadstep).")
        println("##########################")
        controlled_displacement = max_displacement*(loadstep/n_loadsteps)

        for j in 1:3 # Staggered iterations

            ########################
            # Solving displacement #
            ########################
            # Reset solution vectors
            u_ax = Vector{Float64}[]; push!(u_ax,u_ax_Dirichlet*(controlled_displacement)^(1/2))
            u_ay = Vector{Float64}[]; push!(u_ay,u_ay_Dirichlet*(controlled_displacement)^(1/2))

            for modeItr = 2:(u_n_modes + 1)
                iterations = 0

                # Initial guess
                push!(u_ax,u_ax_old[modeItr])
                push!(u_ay,u_ay_old[modeItr])
                u_compsold = IterativeFunctionComponents(u_ax_old[modeItr],u_ay_old[modeItr])

                while true; iterations += 1

                    u_ax_new, Ψ_new = ud_x_mode_solver(Ux,u_ax,Uy,u_ay,UN,global_u_fe_values,
                                                       E,b,u_xbc,
                                                       Dx,d_ax,Dy,d_ay,DN,
                                                       Ψ_new)
                    u_ax[end] = u_ax_new*0.9

                    u_ay_new, Ψ_new = ud_y_mode_solver(Ux,u_ax,Uy,u_ay,UN,global_u_fe_values,
                                                       E,b,u_ybc,
                                                       Dx,d_ax,Dy,d_ay,DN,
                                                       Ψ_new)
                    u_ay[end] = u_ay_new*0.9


                    # println("Done with iteration $(iterations) for mode $(modeItr).")

                    u_compsnew = IterativeFunctionComponents(u_ax_new,u_ay_new)
                    u_xdiff, u_ydiff = iteration_difference(u_compsnew,u_compsold)
                    # println("u_xdiff = $(u_xdiff), u_ydiff = $(u_ydiff)")
                    u_compsold = u_compsnew

                    if (u_xdiff < TOL && u_ydiff < TOL) || iterations > 100
                        println("Converged for u mode $(modeItr) after $(iterations) iterations.")
                        println("u_xdiff = $(u_xdiff), u_ydiff = $(u_ydiff)")
                        # vtkwriter(pvd,modeItr,Ux,u_ax,Uy,u_ay)
                        break
                    end

                end

            end # of mode iterations
            if loadstep != 0
                u_ax_old = copy(u_ax)
                u_ay_old = copy(u_ay)
            end

            ########################
            # Calculate max energy #
            ########################
            for el in 1:length(Ψ)
                Ψ[el] = max(Ψ[el],Ψ_new[el])
            end

            ##################
            # Solving damage #
            ##################
            # Reset solution vectors
            d_ax = Vector{Float64}[]
            d_ay = Vector{Float64}[]
            for modeItr = 1:d_n_modes
                iterations = 0

                # Initial guess
                push!(d_ax,d_ax_old[modeItr])
                push!(d_ay,d_ay_old[modeItr])
                d_compsold = IterativeFunctionComponents(d_ax_old[modeItr],d_ay_old[modeItr])

                while true; iterations += 1

                    d_ax_new = du_x_mode_solver(Dx,d_ax,Dy,d_ay,DN,global_d_fe_values,dmp,d_xbc,Ψ)
                    d_ax[end] = d_ax_new*0.9

                    d_ay_new = du_y_mode_solver(Dx,d_ax,Dy,d_ay,DN,global_d_fe_values,dmp,d_ybc,Ψ)
                    d_ay[end] = d_ay_new*0.9


                    # println("Done with iteration $(iterations) for mode $(modeItr).")

                    d_compsnew = IterativeFunctionComponents(d_ax_new,d_ay_new)
                    d_xdiff, d_ydiff = iteration_difference(d_compsnew,d_compsold)
                    # println("d_xdiff = $(d_xdiff), d_ydiff = $(d_ydiff)")
                    d_compsold = d_compsnew

                    if (d_xdiff < TOL && d_ydiff < TOL) || iterations > 100
                        println("Converged for d mode $(modeItr) after $(iterations) iterations.")
                        println("d_xdiff = $(d_xdiff), d_ydiff = $(d_ydiff)")
                        break
                    end

                end
            end # of mode iterations
            if loadstep != 0
                d_ax_old = copy(d_ax)
                d_ay_old = copy(d_ay)
            end
        end

        # Write to file
        vtkwriter(pvd,loadstep,Ux,u_ax,Uy,u_ay,Dx,d_ax,Dy,d_ay)

    end # of loadstepping

    vtk_save(pvd)
    return Ux,u_ax,Uy,u_ay,Dx,d_ax,Dy,d_ay
end

@time o = main_damage_2D_2D_integration();

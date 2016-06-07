using TimerOutputs
using PyPlot

include("FEM.jl")
include("PGD_1D_integration.jl")
include("PGD_3D_integration.jl")



function beam_benchmarker()

    meshes = collect(5:1:20)

    times = zeros(4,length(meshes))
    times[1,:] = float(meshes)

    for (i,nel) in enumerate(meshes)
        println("nel = $nel")
        u_FEM, v_FEM, w_FEM, t_FEM = bench_beam_FEM(nel)
        times[2,i] = t_FEM

        # u_PGD_3D, t_PGD_3D = bench_beam_PGD_3D_integration(nel)
        # times[3,nel] = t_PGD_3D

        u_PGD_1D, t_PGD_1D = bench_beam_PGD_1D_integration(nel,u_FEM,v_FEM,w_FEM)
        times[4,i] = t_PGD_1D
    end

    times[2:end,:] /= times[2,1]

    plot(meshes, times[2,:],label="FEM")
    # plot(collect(2:N), times[3,2:end],label="PGD_3D")
    plot(meshes, times[4,:],label="PGD_1D")

    return times

end


# o = beam_benchmarker()

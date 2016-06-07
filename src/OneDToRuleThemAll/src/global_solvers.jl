function ud_solver()
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

        # Solving displacement
        while true; iterations += 1

            newXmode = mode_solver(Ux,aX,Uy,aY,Uz,aZ,E,b,xbc,Val{3}())
            aX[end] = newXmode

            newYmode = mode_solver(Uy,aY,Uz,aZ,Ux,aX,E,b,ybc,Val{3}())
            aY[end] = newYmode

            newZmode = mode_solver(Uz,aZ,Ux,aX,Uy,aY,E,b,zbc,Val{3}())
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

    return 
end
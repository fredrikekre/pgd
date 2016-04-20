function print_loadstep(step::Int,nsteps::Int)
    if step != 0
        println("")
    end
    infostring = "Loadstep #$(step) of $(nsteps)"
    println(infostring)
    dashstring = repeat("-",length(infostring))
    println(dashstring)
end

function print_modeitr(mode::Int,nmodes::Int,name::AbstractString)
    infostring = string("Solving ", name, "-mode #$(mode) of $(nmodes) ... ")
    print(infostring)
end

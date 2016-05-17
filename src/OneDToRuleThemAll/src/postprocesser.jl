function postprocesser_beam(aX, aY, aZ, Ux, Uy, Uz)

    writedlm("../../../../FinalReport/DataPlots/raw_data/double_beam/xmodes.txt", aX)
    xcoords = Ux.mesh.x
    xcoords = reinterpret(Float64,xcoords,(length(xcoords),))
    writedlm("../../../../FinalReport/DataPlots/raw_data/double_beam/xcoords.txt", xcoords)
    writedlm("../../../../FinalReport/DataPlots/raw_data/double_beam/ymodes.txt", aY)
    ycoords = Uy.mesh.x
    ycoords = reinterpret(Float64,ycoords,(length(ycoords),))
    writedlm("../../../../FinalReport/DataPlots/raw_data/double_beam/ycoords.txt", ycoords)
    writedlm("../../../../FinalReport/DataPlots/raw_data/double_beam/zmodes.txt", aZ)
    zcoords = Uz.mesh.x
    zcoords = reinterpret(Float64,zcoords,(length(zcoords),))
    writedlm("../../../../FinalReport/DataPlots/raw_data/double_beam/zcoords.txt", zcoords)

end
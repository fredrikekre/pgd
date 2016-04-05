function displacementBC(U)

    x_mode_dofs = collect(1:U.components[1].mesh.nDofs)
    y_mode_dofs = collect((U.components[1].mesh.nDofs+1):(U.components[1].mesh.nDofs+U.components[2].mesh.nDofs))

    # bc_U: Combined bc's
    bc_U = Vector[[x_mode_dofs[[1,2]];
                  x_mode_dofs[[end-1,end]];
                  y_mode_dofs[[1,2]];
                  y_mode_dofs[[end-1,end]]],
                  0.0*[x_mode_dofs[[1,2]];
                  x_mode_dofs[[end-1,end]];
                  y_mode_dofs[[1,2]];
                  y_mode_dofs[[end-1,end]]]]

    # bc_Ux: Lock y_mode_dofs
    bc_Ux = Vector[[x_mode_dofs[[1,2]];
                    x_mode_dofs[[end-1,end]];
                    y_mode_dofs],
                    0.0*[x_mode_dofs[[1,2]];
                    x_mode_dofs[[end-1,end]];
                    y_mode_dofs]]

    # bc_Uy: Lock x_mode_dofs
    bc_Uy = Vector[[x_mode_dofs;
                    y_mode_dofs[[1,2]];
                    y_mode_dofs[[end-1,end]]],
                    0.0*[x_mode_dofs;
                    y_mode_dofs[[1,2]];
                    y_mode_dofs[[end-1,end]]]]

    return bc_U, bc_Ux, bc_Uy
end
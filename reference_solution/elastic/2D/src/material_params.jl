immutable LEmtrl # Linear elastic material
    E::Float64
    ν::Float64
    G::Float64
    K::Float64
end

function LEmtrl()
    # μ = 80.77e9
    # G = μ
    # λ = 121.15e9
    # E = G * (3*λ + 2*G) / (λ + G)
    # ν = E/(2*G) - 1
    E = 1.0
    ν = 0.3
    G = E/(2*(1+ν))
    K = E / (3*(1 - 2*ν))
    return LEmtrl(E,ν,G,K)
end

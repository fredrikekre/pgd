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




immutable DamageParams # Phase-field parameters
    l::Float64
    gc::Float64
    rp::Float64
end

# function DamageParams()
#     l = 0.005
#     gc = 0.1/1000.0
#     rp = 1000.0 # Not used
#     return DamageParams(l,gc,rp)
# end

function DamageParams()
    l = 0.05
    gc = 0.01/1000
    rp = 1000.0 # Not used
    return DamageParams(l,gc,rp)
end

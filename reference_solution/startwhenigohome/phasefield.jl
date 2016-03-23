immutable LEmtrl # Linear elastic material
    E::Float64
    ν::Float64
    G::Float64
    K::Float64
    I_dev_sym::Matrix
    I::Vector
end

function LEmtrl()
    # μ = 80.77e9
    # G = μ
    # λ = 121.15e9
    # E = G * (3*λ + 2*G) / (λ + G)
    # ν = E/(2*G) - 1
    E = 210.0e9
    ν = 0.3
    G = E/(2*(1+ν))
    K = E / (3*(1 - 2*ν))
    I = [1,1,0]
    I_dev_sym = eye(3) - 1/3*I*I'
    return LEmtrl(E,ν,G,K,I_dev_sym,I)
end




immutable DamageParams # Phase-field parameters
    l::Float64
    gc::Float64
    rp::Float64
end

function DamageParams()
    l = 15.0e-6
    gc = 2700
    rp = 1000.0
    return DamageParams(l,gc,rp)
end

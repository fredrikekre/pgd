abstract MaterialParameters{T}

# Linear elastic material
immutable LinearElastic{T} <: MaterialParameters{T}
    E::T # Young's modulus
    ν::T # Poisson's ratio
    G::T # Shear modulus
    K::T # Bulk modulus
    μ::T # Lamé constants
    λ::T
end

function LinearElastic{T}(p1::Symbol,v1::T,p2::Symbol,v2::T)
    if p1 == :E && p2 == :ν
        E = v1; ν = v2
        G = E/(2.0*(1.0+ν))
        K = E/(3.0*(1.0-2.0*ν))
        μ = G
        λ = (E*ν)/((1.0+ν)*(1.0-2.0*ν))
        LinearElastic(E,ν,G,K,μ,λ)
    elseif p1 == :G && p2 == :K
        G = v1; K = v2
        E = (9.0*G*K)/(3.0*K+G)
        ν = (3.0*K-2.0*G)/(2.0*(3.0*K+G))
        μ = G
        λ = K-2.0*G/3.0
        LinearElastic(E,ν,G,K,μ,λ)
    else
        throw(ArgumentError("Parameter pair not supported."))
    end
end

function LinearElastic{T,Q}(p1::Symbol,v1::T,p2::Symbol,v2::Q)
    prom_par = promote(v1,v2)
    return LinearElastic(p1,prom_par[1],p2,prom_par[2])
end
LinearElastic(v1::Real,v2::Real) = LinearElastic(:E,v1,:ν,v2)

function TangentStiffness(mp::LinearElastic) # Voigt tangentstiffness
    E = CALFEM.hooke(2,mp.E,mp.ν)
    return E[[1,2,4],[1,2,4]]
    # return TangentStiffness(E)
end

immutable LinearElastic_withTangent{T}
    mp::LinearElastic{T}
    tangent::Matrix{T}
end


# Damage parameters
immutable PhaseFieldDamage{T} <: MaterialParameters{T}
    gc::T # Energy release rate
    l::T # Length scale
end

PhaseFieldDamage{T,Q}(gc::T,l::Q) = PhaseFieldDamage(promote(gc,l)...)

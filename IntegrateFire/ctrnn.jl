using Dates, Gadfly

set_default_plot_size(25cm, 35cm)

# simple Integrate and Fire Synapse Mode
type Synapse
    τ::Real # membrane time constant [ms]
    dt::DateTime # time stamp of last update
    v::Real # tracks current voltage
    function Synapse(v::Real;τ::Real=10., dt::DateTime=now())
        new(τ, dt, v)
    end
end

nextDt(dt::DateTime, addMs::Int64) = dt + Millisecond(addMs)

function update!(s::Synapse, iV::Real, dt::DateTime)
    Δdt = int(dt - s.dt)
    s.v += Δdt*(1.0/s.τ)*(-s.v + iV)
    return s.v
end

function getResponse(V::Vector{Float64}, Δτ::Vector{Int64})
    n = length(V)
    data = zeros(n)
    tm = zeros(Int64,n)
    s = Synapse(0,τ=10)
#     tm = ones(n)
    for i = 1:n
        v = update!(s, V[i], nextDt(s.dt, Δτ[i]))
        data[i] = v
        tm[i] = i == 1? Δτ[i] : tm[i-1] + Δτ[i]
    end
    return data,tm
end

n = 500
Δτ = rand(1:20,n)
# V = ones(n) - .0
V = max(randn(n) * 2 .+ 0.,0)
response, tm = getResponse(V, Δτ)
vstack(plot(x = tm,y=V, Geom.bar), plot(x=tm, y=response, Geom.line))



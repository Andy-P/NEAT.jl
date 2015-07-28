using Dates, Gadfly

# set_default_plot_size(15cm, 15cm)
# simple Integrate and Fire Synapse Mode
type Synapse
    Ω::Real # membrane resistance [MΩ]
    τ::Real # membrane time constant [ms]
    rV::Real # resting membrane potential [mV]
    thV::Real # spike threshold [mV]
    sV::Real # spike voltage [mV]
    rsV::Real # reset voltage to after a spike [mV]
    dt::DateTime # time stamp of last update
    v::Real # tracks current voltage
    function Synapse(;Ω::Real=10.,  τ::Real=10.,  rV::Real=-70.,
                     thV::Real=-55, sV::Real=-20, rsV::Real=-75,
                     dt::DateTime=now())
        new(Ω, τ, rV, thV, sV, rsV, dt,rV)
    end
end

nextDt(dt::DateTime, addMs::Int64) = dt + Millisecond(addMs)

function update!(s::Synapse, iV::Real, dt::DateTime)
    Δdt = int(dt - s.dt)
    V = s.rV + iV * s.Ω # current input voltage
    V₁ = V + (s.v - V) * exp(-Δdt/s.τ)
    s.v = V₁ >= s.thV? s.rsV:V₁
    return s.v == s.rsV? s.sV : s.v
end

Δτ = 0 # in millesec
n = 1000
s = Synapse(Ω=10,τ=10)
data = zeros(n)
# V = copy(data)
V = max(randn(n) * 1.5 + 0.5,0)
tm = ones(n)
t = 0
V[:] = 1.502
for i = 1:n
    Δτ = rand(1:1)
    v = update!(s, V[i], nextDt(s.dt, Δτ))
    data[i] = v
    tm[i] = i == 1? Δτ:tm[i-1] +  Δτ
end

vstack(plot(x=tm, y=data, Geom.line, Scale.y_continuous(minvalue=-80, maxvalue=10)), plot(x = tm,y=V, Geom.bar))

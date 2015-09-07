using Gadfly

set_default_plot_size(20cm, 15cm)

#     A spiking neuron model based on:
#     Izhikevich, E. M.
#     Simple Model of Spiking Neurons
#     IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 14, NO. 6, NOVEMBER 2003

#         a, b, c, d are the parameters of this model.
#         a: the time scale of the recovery variable.
#         b: the sensitivity of the recovery variable.
#         c: the after-spike reset value of the membrane potential.
#         d: after-spike reset of the recovery variable.

type IZNeuron
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    v::Float64 # membrane potential
    u::Float64 # membrane recovery variable
    has_fired::Bool
    bias::Float64
    current::Float64
    function IZNeuron(bias::Float64=0; a::Float64=0.02,b::Float64=0.2, c::Float64=-65.0, d::Float64=8.0)
       new(a, b, c, d, c, b*c, false, bias, bias)
    end
end


function update!(n::IZNeuron)
#         Advances time in 1 ms.
    n.v += 0.5 * (0.04 * n.v ^ 2. + 5. * n.v + 140. - n.u + n.current)
    n.v += 0.5 * (0.04 * n.v ^ 2. + 5. * n.v + 140. - n.u + n.current)
    n.u += n.a * (n.b * n.v - n.u)
    if n.v > 30
        n.has_fired = true
        n.v = n.c
        n.u += n.d
    else
        n.has_fired = false
        n.current = n.bias
    end
    return n.v
end

function reset!(n::IZNeuron)
#         'Resets all state variables.'
    n.v = n.c
    n.u = n.b * n.v
    n.has_fired = false
    n.current = n.bias
end

function getResponse(V::Vector{Float64}, s::IZNeuron)
    n = length(V)
    data = zeros(n)
    tm = zeros(Int64,n)
    for i = 1:n
        s.current = V[i]
        v = update!(s)
        data[i] = v
        tm[i] = i == 1? 1 : tm[i] = tm[i-1] + 1
    end
    return data,tm
end


# The following parameters produce some known spiking behaviors:
# Regular spiking:
s = IZNeuron(0.,a = 0.02, b = 0.2, c = -65., d = 8.)
# Intrinsically bursting:
s = IZNeuron(0.,a = 0.02, b = 0.2, c = -55., d = 4.)
# Chattering:
s = IZNeuron(0.,a = 0.02, b = 0.2, c = -50., d = 2.)
# Fast spiking:
s = IZNeuron(0.,a = 0.1, b = 0.2, c = -65., d = 2.)
# Thalamo-/cortical:
s = IZNeuron(0.,a = 0.02, b = 0.25, c = -65., d = 0.05)
# Resonator:
s = IZNeuron(0.,a = 0.1, b = 0.25, c = -65., d = 2.)
# Low-threshold spiking:
s = IZNeuron(0.,a = 0.02, b = 0.25, c = -65., d = 2.)

n = 2000
V = ones(n) * 5
response, tm = getResponse(V, s)
vstack(plot(x = tm, y=V, Geom.line), plot(x=tm, y=response, Geom.line))

response[1:10]
tm[1:10]


Ne = 800;                     Ni = 200;
re = rand(Ne);                ri = rand(Ni);
a = [0.02 * ones(Ne),         0.02 + 0.08 * ri]
b = [0.2  * ones(Ne),         0.25 - 0.05 * ri]
c = [-65. + 15.*re.^2,        -65. * ones(Ni)]
d = [  8. - 6. *re.^2,        2. * ones(Ni)]
S = hcat(0.5*rand(Ne+Ni,Ne),  -rand(Ne+Ni,Ni))

v =  -65*ones(Ne+Ni)
u =  b .* v

firings = zeros(Ne+Ni,1000)

for t =  1:1000
    I = [5.*randn(Ne), 2.*randn(Ni)]
    fired = find(x -> x >= 30,v)
    firings[fired,t] = v[fired]
    v[fired] = c[fired]
    u[fired] = u[fired] + d[fired]
    I = I + sum(S[:,fired],2)
    v = v .+ 0.5 * (0.04 * v .^ 2. + 5. .* v + 140. .- u .+ I)
    v = v .+ 0.5 * (0.04 * v .^ 2. + 5. .* v + 140. .- u .+ I)
    u = u .+ a .* (b .* v .- u)
end
firings
# plot(x=[1:1000], y=firings[3,:], Geom.line)
spy(firings)

# class Synapse:
#     """ A synapse indicates the connection strength between two neurons (or itself) """
#     def __init__(self, source, dest, weight):
#         self.__weight = weight
#         self.__source = source
#         self.__dest = dest

#     def advance(self):
#         'Advances time in 1 ms.'
#         if self.__source.has_fired:
#             self.__dest.current += self.__weight # dest.current or dest.__v ?

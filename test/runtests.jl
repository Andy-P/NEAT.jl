using NEAT
using Base.Test

# create 2 input nodes

# create 3 hidden nodes

# create 2 output nodes

# connect:
# I1 -> H1
# I2 -> H2
# H2 -> H3
# H2 -> H1
# H1 -> H3
 # H2 -> O1
# H1 -> O2
# H2 -> O2

# test forward

# test recursive
# change connections to creat loop
#  H2 -> H1 become  H1 -> H2
#  H1 -> H3 become  H3 -> H1


inputs =  [1:10]
i = 0
i += 1
function tick(i)
   i += 1
   println("ticking $i")
end
timer = Timer(i->tick(i) )
start_timer(timer, 1, 1)
stop_timer(timer)

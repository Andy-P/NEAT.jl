# type of forward prop
#    1. Forward - no recrusivity (straight forward approach) ✔︎
#    2. Forward - with self-recursive (only recursively connect to self) ✔︎
#    3. Forward - CTRnn recursivity allowed but shouldn't (?) create loops
#    4. Parallel - Standard but with no limits on structure
#    5. Parallel - CTRnn

# Problems/Requirements ✔︎
#   1.  If forward prop called, may get caught in loop
#   1.  Require some of the types to allow for backprop
#   2.  How to properly dispatch on node type
#       - need a subtype
#   3.  Parallel update cannot use chain pull method


# types: Forward, Forward -
# approach: Forward  = standard forward prop
#           Parallel = use previous output
#           Hyprid   = calls function based on type dispatch

# Clarity:  Inputs to a node are either...
#               Forward: Inputs that have just been calculated
#               Parallel: Inputs caluclated in previous step (AKA outputs)

abstract Connection
abstract Node

# calls each in function chain

# calls each in turn

# gene.f = () -> gene.output
# gene.update = () -> gene.output
type InputNode <: Node

end

type OutputNode <: Node

end

type HiddedNode <: Node


end

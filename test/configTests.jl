
examplePath =

@osx_only ls = split(readall(joinpath(dirname(@__FILE__),"../examples//doublePole","dp_config.txt")),"\n")

ls

ls2 = filter(l->length(l)>0 && l[1] != '#', ls)
ls3 = filter(l->length(l)>0 && l[1] != '#', ls)

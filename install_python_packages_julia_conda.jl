
using PyCall, Conda
# where are we installing the package
pyimport("sys").executable

#----------- Windows and Ubuntu----------------------
Conda.add("pytorch"; channel="pytorch")
Conda.add("torchvision"; channel="pytorch")
Conda.add("spyder")
Conda.add("scipy")
Conda.add("pandas")
Conda.add("pip")

pip = pyimport("pip")

pip.main(["install", "git+https://github.com/gbaydin/hypergradient-descent.git"])

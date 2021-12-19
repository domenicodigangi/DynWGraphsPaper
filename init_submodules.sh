git config --global user.email "emali@example.com"
git config --global user.name "digangidomenico"
cd DynWGraphsPaper
git submodule init
git submodule update
cd src/dynwgraphs
git submodule init
git submodule update
cd ../../
sudo apt-get update
sudo apt-get install bzip2 libxml2-dev
cd ..


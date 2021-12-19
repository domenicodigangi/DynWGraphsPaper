git config --global user.email "emali@example.com"
git config --global user.name "digangidomenico"
git clone https://github.com/domenicodigangi/DynWGraphsPaper.git
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


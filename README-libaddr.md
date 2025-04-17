sudo apt-get install curl autoconf automake libtool python-dev pkg-config
git clone https://github.com/openvenues/libpostal

mkdir $HOME/libpostal_data
cd libpostal

nano configure # Search for -mfpmath=sse => Comment out the line
    # CFLAGS="$CFLAGS -mfpmath=sse -msse2..."

./bootstrap.sh
./configure --datadir=$HOME/libpostal_data
make
sudo make install


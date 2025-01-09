if [ -d "build" ]; then
    rm -rf build
    mkdir build
else
    mkdir build
fi

cd build

cmake ..

make install -j8
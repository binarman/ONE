./nncc configure -DCMAKE_TOOLCHAIN_FILE=${HOME}/nnfw/infra/nnfw/cmake/buildtool/cross/toolchain_aarch64-android.cmake -DCMAKE_BUILD_TYPE=release -DNDK_DIR=/home/binarman/Downloads/android-ndk-r20b
./nncc build luci_run -j10

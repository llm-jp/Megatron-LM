# TSUBAME 4.0

## Environment Modules

デフォルトで用意されている module だけでは、実際のワークロードに不足することがよくある。
その際に、sudo を利用せずに自前の module を作成できるようになっておくことは重要である。
また、サーバー管理者として GPU server を管理するのに、CUdA Toolkit, cuDNN, NCCL, HPC-X などのライブラリを管理することは非常に重要であり、常に正しい形で管理されるようにしておく必要がある。

### CUDA Toolkit

通常以下のスクリプトを実行する際には、`sudo`が必要であるが、これは Driver も install できるようになっているからである。CUDA Toolkit のみを install する場合は、`sudo`は不要であるので、以下のコマンドにより任意の箇所に CUDA Toolkit を install する。

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit --toolkitpath=/gs/bs/tga-bayes-crest/fujii/modules/cuda/cuda-12.1
```

これにより、`/gs/bs/tga-bayes-crest/fujii/modules/cuda/cuda-12.1`に CUDA Toolkit が install される。

```bash
> ls -la /gs/bs/tga-bayes-crest/fujii/modules/cuda/cuda-12.1
total 148
drwxr-sr-x 18 ug02141 tga-bayes-crest  4096 Apr 21 23:17 .
drwxr-sr-x  3 ug02141 tga-bayes-crest  4096 Apr 21 23:11 ..
drwxr-sr-x  3 ug02141 tga-bayes-crest  4096 Apr 21 23:17 bin
drwxr-sr-x  5 ug02141 tga-bayes-crest  4096 Apr 21 23:16 compute-sanitizer
-rw-r--r--  1 ug02141 tga-bayes-crest   160 Apr 21 23:17 DOCS
-rw-r--r--  1 ug02141 tga-bayes-crest 61498 Apr 21 23:17 EULA.txt
drwxr-sr-x  5 ug02141 tga-bayes-crest  4096 Apr 21 23:17 extras
drwxr-sr-x  6 ug02141 tga-bayes-crest  4096 Apr 21 23:16 gds
drwxr-sr-x  2 ug02141 tga-bayes-crest  4096 Apr 21 23:16 gds-12.1
lrwxrwxrwx  1 ug02141 tga-bayes-crest    28 Apr 21 23:17 include -> targets/x86_64-linux/include
lrwxrwxrwx  1 ug02141 tga-bayes-crest    24 Apr 21 23:17 lib64 -> targets/x86_64-linux/lib
drwxr-sr-x  7 ug02141 tga-bayes-crest  4096 Apr 21 23:17 libnvvp
drwxr-sr-x  7 ug02141 tga-bayes-crest  4096 Apr 21 23:17 nsight-compute-2023.1.0
drwxr-sr-x  2 ug02141 tga-bayes-crest  4096 Apr 21 23:16 nsightee_plugins
drwxr-sr-x  6 ug02141 tga-bayes-crest  4096 Apr 21 23:17 nsight-systems-2023.1.2
drwxr-sr-x  3 ug02141 tga-bayes-crest  4096 Apr 21 23:16 nvml
drwxr-sr-x  7 ug02141 tga-bayes-crest  4096 Apr 21 23:17 nvvm
drwxr-sr-x  2 ug02141 tga-bayes-crest  4096 Apr 21 23:16 pkgconfig
-rw-r--r--  1 ug02141 tga-bayes-crest   524 Apr 21 23:17 README
drwxr-sr-x  3 ug02141 tga-bayes-crest  4096 Apr 21 23:16 share
drwxr-sr-x  2 ug02141 tga-bayes-crest  4096 Apr 21 23:16 src
drwxr-sr-x  3 ug02141 tga-bayes-crest  4096 Apr 21 23:16 targets
drwxr-sr-x  2 ug02141 tga-bayes-crest  4096 Apr 21 23:17 tools
-rw-r--r--  1 ug02141 tga-bayes-crest  2928 Apr 21 23:16 version.json
```

対応する modulefile は以下の書く。
なお、環境変数にセットしているだけなので、`export`を多用するれば、modulefile は不要である。

```bash
#%Module1.0
##
## CUDA 12.1 modulefile
##
proc ModulesHelp { } {
  puts stderr "This module adds CUDA 12.1 to your environment variables."
}
module-whatis "Sets up CUDA 12.1 in your environment"

set version 12.1
set cuda_home /gs/bs/tga-bayes-crest/fujii/modules/cuda/cuda-$version

prepend-path    PATH            $cuda_home/bin
prepend-path    LD_LIBRARY_PATH $cuda_home/lib64
prepend-path    MANPATH         $cuda_home/doc/man
setenv          CUDA_HOME       $cuda_home
```

### cuDNN

https://developer.nvidia.com/rdp/cudnn-archive から cuDNN のバージョンを選択し、ダウンロードする。

tar ファイルを解凍し、module を管理するディレクトリに移動する。

```bash
tar xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
mv cudnn-linux-x86_64-8.9.7.29_cuda12-archive /gs/bs/tga-bayes-crest/fujii/modules/cudnn/
```

その後、以下のコマンドで cudnn 関連のヘッダーファイルとライブラリファイルを CUDA Toolkit のディレクトリにコピーする。

```bash
cp /gs/bs/tga-bayes-crest/fujii/modules/cudnn/cudnn-8.9.7.29_cuda12/include/cudnn*.h /gs/bs/tga-bayes-crest/fujii/modules/cuda/cuda-12.1/include/
cp /gs/bs/tga-bayes-crest/fujii/modules/cudnn/cudnn-8.9.7.29_cuda12/lib64/libcudnn* /gs/bs/tga-bayes-crest/fujii/modules/cuda/cuda-12.1/lib64/
```

自分が管理者の場合は、以下のように all user に READ をつける。

```bash
chmod a+r /gs/bs/tga-bayes-crest/fujii/modules/cuda/cuda-12.1/include/cudnn*.h
chmod a+r /gs/bs/tga-bayes-crest/fujii/modules/cuda/cuda-12.1/lib64/libcudnn*
```

対応する modulefile は以下のように書く。

```bash
#%Module1.0
##
## cuDNN 8.9.7 modulefile
##
proc ModulesHelp { } {
    puts stderr "This module adds cuDNN 8.9.7 to your environment variables."
}
module-whatis "Sets up cuDNN 8.9.7 in your environment"

set version 8.9.7
set cudnn_root /gs/bs/tga-bayes-crest/fujii/modules/cuda/cuda-12.1

prepend-path  LD_LIBRARY_PATH     $cudnn_root/lib64
prepend-path 	LIBRARY_PATH        $cudnn_root/lib64
prepend-path 	CPATH               $cudnn_root/include
setenv        CUDNN_PATH          $cudnn_root
setenv        CUDNN_INCLUDE_DIR   $cudnn_root/include
setenv        CUDNN_LIBRARY_DIR   $cudnn_root/lib64
setenv	 	    CUDNN_ROOT_DIR	    $cudnn_root/
```

### NCCL

https://developer.nvidia.com/nccl/nccl-legacy-downloads から NCCL のバージョンを選択し、ダウンロードする。(Sign in が必要)

対象のバージョンの O/S agnostic local installer をクリックし、ダウンロードする。
(基本、自分が使用する CUDA Toolkit のバージョンに対応するものを選択する)

download したファイルを解凍し、module を管理するディレクトリに移動する。

```bash
> tar -xvf nccl_2.18.3-1+cuda12.1_x86_64.txz

nccl_2.18.3-1+cuda12.1_x86_64/include/
nccl_2.18.3-1+cuda12.1_x86_64/include/nccl.h
nccl_2.18.3-1+cuda12.1_x86_64/include/nccl_net.h
nccl_2.18.3-1+cuda12.1_x86_64/lib/
nccl_2.18.3-1+cuda12.1_x86_64/lib/libnccl.so.2
nccl_2.18.3-1+cuda12.1_x86_64/lib/libnccl.so.2.18.3
nccl_2.18.3-1+cuda12.1_x86_64/lib/libnccl.so
nccl_2.18.3-1+cuda12.1_x86_64/lib/libnccl_static.a
nccl_2.18.3-1+cuda12.1_x86_64/lib/pkgconfig/
nccl_2.18.3-1+cuda12.1_x86_64/lib/pkgconfig/nccl.pc
nccl_2.18.3-1+cuda12.1_x86_64/LICENSE.txt
```

modulefile は以下のように書く。

```bash
#%Module1.0

## NCCL 2.18.3 modulefile

proc ModulesHelp { } {
    puts stderr "NCCL 2.18.3"
}

module-whatis "Sets up NCCL 2.18.3 in your environment"

set version 2.18.3
set nccl_root /gs/bs/tga-bayes-crest/fujii/modules/nccl/nccl_2.18.3-1+cuda12.1_x86_64

prepend-path LD_LIBRARY_PATH $nccl_root/lib
prepend-path LIBRARY_PATH $nccl_root/lib
prepend-path CPATH $nccl_root/include
setenv NCCL_HOME $nccl_root
setenv NCCL_INCLUDE_DIR $nccl_root/include
setenv NCCL_LIBRARY_DIR $nccl_root/lib
setenv NCCL_VERSION $version
```

なお、NCCL は CUDA Toolkit の version に依存しているため、以下のように directory を切って管理すると良い。

```bash
/modulefiles/ylab/nccl/cuda-12.1/2.18.3
```

こうすることで、`module avail`をしたときに以下のようになる。

```bash
$ module avail
--------------------- /home/1/ug02141/modulefiles --------------------------
ylab/cuda/12.1  ylab/cudnn/8.9.7  ylab/cudnn/9.0.0  ylab/hpcx/2.17.1  ylab/nccl/cuda-12.0/2.19.3  ylab/nccl/cuda-12.1/2.18.3  ylab/nccl/cuda-12.3/2.19.3
```

### HPC-X

https://developer.nvidia.com/networking/hpc-x のページの下部に Resources Download があるので、そこからダウンロードする。

対応する CUDA Toolkit を選び、OS のディストリビューションを選択し、ダウンロードする。

- Version Current: 2.18.0-CUDA12.x
- MLNX_OFED/OFED: inbox
- MLNX_OFED/OFEDVer.: inbox
- OS Distro: RHEL/CentOS/Rucky
- OS Distp Ver.: 9.x
- Arch.: x86_64

のように選択すると http://www.mellanox.com/page/hpcx_eula?mrequest=downloads&mtype=hpc&mver=hpc-x&mname=v2.18/hpcx-v2.18-gcc-inbox-redhat9-cuda12-x86_64.tbz がダウンロードできる。

ダウンロードしたファイルを解凍し、module を管理するディレクトリに移動する。

```bash
tar -xvf hpcx-v2.17.1-gcc-inbox-ubuntu20.04-cuda12-x86_64.tbz
```

以下のように modulefile を書く。

```bash
#%Module

set fn      $ModulesCurrentModulefile
set fn      [file normalize $ModulesCurrentModulefile]

if {[file type $fn] eq "link"} {
set fn [ exec readlink -f $fn]
}

set hpcx_dir        /gs/bs/tga-bayes-crest/fujii/modules/hpcx/hpcx-v2.17.1-gcc-inbox-redhat9-cuda12-x86_64
set hpcx_mpi_dir    $hpcx_dir/ompi

module-whatis   NVIDIA HPC-X toolkit

setenv HPCX_DIR             $hpcx_dir
setenv HPCX_HOME            $hpcx_dir

setenv HPCX_MPI_DIR         $hpcx_mpi_dir
setenv HPCX_OSHMEM_DIR      $hpcx_mpi_dir
setenv HPCX_MPI_TESTS_DIR   $hpcx_mpi_dir/tests
setenv HPCX_OSU_DIR         $hpcx_mpi_dir/tests/osu-micro-benchmarks-7.2
setenv HPCX_OSU_CUDA_DIR    $hpcx_mpi_dir/tests/osu-micro-benchmarks-7.2-cuda
prepend-path    PATH    $hpcx_mpi_dir/tests/imb
prepend-path    CPATH   $hpcx_mpi_dir/include
prepend-path    PKG_CONFIG_PATH $hpcx_dir/ompi/lib/pkgconfig

prepend-path    MANPATH         $hpcx_mpi_dir/share/man

# Adding MPI runtime

setenv OPAL_PREFIX          $hpcx_mpi_dir
setenv PMIX_INSTALL_PREFIX  $hpcx_mpi_dir
setenv OMPI_HOME            $hpcx_mpi_dir
setenv MPI_HOME             $hpcx_mpi_dir
setenv OSHMEM_HOME          $hpcx_mpi_dir
setenv SHMEM_HOME           $hpcx_mpi_dir

prepend-path    PATH    $hpcx_mpi_dir/bin
prepend-path    LD_LIBRARY_PATH $hpcx_mpi_dir/lib
prepend-path    LIBRARY_PATH $hpcx_mpi_dir/lib
```

## NCCL Error

```bash
[r16n5:881490] Warning: could not find environment variable "/gs/bs/tga-bayes-crest/fujii/modules/hpcx/hpcx-v2.17.1-gcc-inbox-redhat9-cuda12-x86_64/ompi/lib:/gs/bs/tga-bayes-crest/fujii/modules/nccl/nccl_2.18.3-1+cuda12.1_x86_64/lib:/gs/bs/tga-bayes-crest/fujii/modules/cuda/cuda-12.1/lib64
```

のようなエラーが出ており、`LD_LIBRARY_PATH` が正しく設定されていなかった。

理由は調査したところ、1node では発生せず、2node 以上で発生していることが分かった。
結局、mpirun に以下を追加するだけで良かった。

```bash
  -x LD_LIBRARY_PATH \
  -x PATH \
```

これにより、`LD_LIBRARY_PATH` が正しく設定され、エラーが解消された。
ABCI では設定しなくても問題なかったが、おそらく自前の module だから発生したものと思われる。

let
  pkgs = import <nixpkgs> {
    config.allowUnfree = true;
  };
in pkgs.mkShell {
  buildInputs = with pkgs; [
      cudatoolkit
      python312
      python3Packages.virtualenv
      sumo
      gcc
      python312Packages.numpy
      git
      gitRepo
      gnupg
      autoconf
      curl
      procps
      gnumake
      util-linux
      m4
      gperf
      unzip
      linuxPackages.nvidia_x11
      libGLU
      libGL
      xorg.libXi
      xorg.libXmu
      freeglut
      xorg.libXext
      xorg.libX11
      xorg.libXv
      xorg.libXrandr
      zlib
      ncurses5
      stdenv.cc
      binutils
    ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
    export SUMO_HOME=$(readlink -f "${pkgs.sumo}/share/sumo")
    export CUDA_PATH=$(readlink -f "${pkgs.cudatoolkit}")
    # Create and activate a Python virtual environment
    if [ ! -d .venv ]; then
      virtualenv .venv
      . .venv/bin/activate
      pip install stable-baselines3 sumo-rl tensorflow tensorboard gym supersuit
    else
      . .venv/bin/activate
    fi
  '';
}

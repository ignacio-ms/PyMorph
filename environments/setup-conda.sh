if command -v curl &> /dev/null
then
    curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh
elif command -v wget &> /dev/null
then
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh
else
    echo
    echo "ERROR: Neither curl nor wget were available. Please install one of them to proceed."
    echo
    exit 1
fi

bash Mambaforge-$(uname)-$(uname -m).sh -b

export PATH="$HOME/mambaforge/bin:$PATH"
conda init
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --add channels mosaic
conda config --set channels morpheme

echo
echo "Setup completed. Please close and reopen your terminal for changes to take effect."
echo

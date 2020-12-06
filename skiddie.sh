source .venv/bin/activate
make clean
make
cd unittests
python3 -m unittest unittests -v
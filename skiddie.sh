source .venv/bin/activate
make
cd unittests
python3 -m unittest unittests.TestPow -v
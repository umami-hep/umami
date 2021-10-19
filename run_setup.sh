INSTDIR=python_install
rm -rf ${INSTDIR}
mkdir ${INSTDIR}
export PYTHONPATH=${PWD}:${PWD}/${INSTDIR}:${PYTHONPATH}
python setup.py develop --install-dir ${INSTDIR}
export PATH=${PWD}/${INSTDIR}:$PATH
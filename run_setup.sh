INSTDIR=python_install
rm -rf ${INSTDIR}
rm -rf *.egg-info
mkdir ${INSTDIR}
export PYTHONPATH=${PWD}:${PWD}/${INSTDIR}:${PYTHONPATH}
python -m pip install --prefix ${INSTDIR} -e .
export PATH=${PWD}/${INSTDIR}/bin:$PATH
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RES_DIR="${SCRIPT_DIR}/../pipeline_result/tests_out"
mkdir -p "${RES_DIR}"
cd "${ROOT_DIR}"

python -m pytest tests -q --junitxml="${RES_DIR}/pytest.xml" | tee "${RES_DIR}/pytest_out.txt"
TEST_RETURN=${PIPESTATUS[0]}

if [ "${TEST_RETURN}" -ne 0 ]; then
    echo "---------- Failed tests found ----------"
    cat "${RES_DIR}/pytest_out.txt"
    echo "------------------------------------------------------"
    echo "--------------- List of tests ---------------"
    python -m pytest tests --collect-only -q
    echo "-------------------------------------------"
    exit 1
fi

echo "---------- All tests were successful ---------"
cat "${RES_DIR}/pytest_out.txt"
echo "-------------------------------------------"
echo "--------------- List of tests ---------------"
python -m pytest tests --collect-only -q
echo "-------------------------------------------"

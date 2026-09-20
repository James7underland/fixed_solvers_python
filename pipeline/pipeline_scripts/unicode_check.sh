#!/bin/bash

error_list="pipeline/pipeline_result/encoding_error_list.txt"
mkdir -p pipeline/pipeline_result
: > "$error_list"

# Аналог C++ unicode_check.sh: там *.m *.cpp *.h, здесь Python-исходники.
while IFS= read -r file; do
    file_encoding=$(file -b --mime-encoding "$file")
    if [ "$file_encoding" != "utf-8" ] && [ "$file_encoding" != "us-ascii" ]; then
        echo "$file" >> "$error_list"
    fi
done < <(find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" -o -name "*.toml" \) \
    -not -path "./.git/*" -not -path "./.venv/*" -not -path "./venv/*" -not -path "./.pytest_cache/*")

echo "--------------- Result ---------------"

if [ -s "$error_list" ]; then
    echo "Error: some files are not in UTF-8 encoding. See the list of affected files in pipeline/pipeline_result/encoding_error_list.txt"
    cat "$error_list"
    echo "--------------------------------------"
    exit 1
else
    echo "All files are in Unicode encoding."
    echo "--------------------------------------"
fi

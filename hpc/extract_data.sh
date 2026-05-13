#!/bin/bash
# Extract DISC21 zip files to correct folders
# Run from HPC: bash hpc/extract_data.sh

cd /scratch/lustre/home/$USER/Kursinis/data

echo "=================================================="
echo "Extracting DISC21 Dataset"
echo "=================================================="
echo "Working directory: $(pwd)"
echo ""

# Clean up any partial extractions
rm -rf train_temp refs_temp queries_dev_temp queries_test_temp images

# Create target folders
mkdir -p train refs queries_dev queries_test

echo "[1/4] Extracting train_50k_0.zip..."
unzip -q train_50k_0.zip
# Move images to train folder (handles both 'images/' and direct extraction)
if [ -d "images" ]; then
    mv images/* train/
    rm -rf images
fi
echo "       Done. Files in train/: $(ls train/ 2>/dev/null | wc -l)"

echo ""
echo "[2/4] Extracting refs_50k_0.zip..."
unzip -q refs_50k_0.zip
if [ -d "images" ]; then
    mv images/* refs/
    rm -rf images
fi
echo "       Done. Files in refs/: $(ls refs/ 2>/dev/null | wc -l)"

echo ""
echo "[3/4] Extracting dev_queries_50k_0.zip..."
unzip -q dev_queries_50k_0.zip
if [ -d "images" ]; then
    mv images/* queries_dev/
    rm -rf images
fi
echo "       Done. Files in queries_dev/: $(ls queries_dev/ 2>/dev/null | wc -l)"

echo ""
echo "[4/4] Extracting test_queries_50k_0.zip..."
unzip -q test_queries_50k_0.zip
if [ -d "images" ]; then
    mv images/* queries_test/
    rm -rf images
fi
echo "       Done. Files in queries_test/: $(ls queries_test/ 2>/dev/null | wc -l)"

echo ""
echo "=================================================="
echo "Cleaning up zip files to save space..."
echo "=================================================="
rm -f train_50k_0.zip refs_50k_0.zip dev_queries_50k_0.zip test_queries_50k_0.zip

echo ""
echo "=================================================="
echo "EXTRACTION COMPLETE"
echo "=================================================="
echo ""
echo "Final structure:"
ls -la
echo ""
echo "Image counts:"
echo "  train/:       $(ls train/ 2>/dev/null | wc -l) images"
echo "  refs/:        $(ls refs/ 2>/dev/null | wc -l) images"
echo "  queries_dev/: $(ls queries_dev/ 2>/dev/null | wc -l) images"
echo "  queries_test/: $(ls queries_test/ 2>/dev/null | wc -l) images"
echo ""
echo "Groundtruth files:"
ls -la *.csv 2>/dev/null || echo "  (none found - upload via WinSCP)"


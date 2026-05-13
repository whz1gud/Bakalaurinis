#!/bin/bash
# Download DISC21 dataset on HPC
# Run from: /scratch/lustre/home/$USER/Kursinis

echo "=================================================="
echo "Downloading DISC21 Dataset"
echo "=================================================="

cd /scratch/lustre/home/$USER/Kursinis
mkdir -p data
cd data

echo ""
echo "[1/4] Downloading train_50k_0.zip..."
wget -O train_50k_0.zip "https://scontent-waw2-2.xx.fbcdn.net/m1/v/t6/An9Bmbe9vm85YAuM2kVgHyVfjBCwSkZIGIBHD4jmgNl-5sanAWdX6zHqD52QOqHuvDsaQpYvD-sr8vgH2MGWiCcX__B6rr2-kVEK24UniHa5dQ.zip?_nc_gid=Z32lFCwCg9r6MDYBeSj69Q&_nc_oc=Adk9hnOE2Lp2CS2TBDth2uPR6pPvCXlZ4v9LuBvZ0Ta5nsnifr_mGILaTrHZ5vnc-4k&ccb=10-5&oh=00_AfnqYeHjuFUn7IIi4bkj2Kd114iJbr024BsW954vvzhI8g&oe=6979B8D4&_nc_sid=e9ffdc"

echo ""
echo "[2/4] Downloading refs_50k_0.zip..."
wget -O refs_50k_0.zip "https://scontent-waw2-2.xx.fbcdn.net/m1/v/t6/An9vkgBxMWHzTS1GFIFsXZLCVPS6gErePMI75jrr3rG_1xFb2ZOkVTjpCD8vmyURTBYwWNejCDNxkyyt6oEl4Bm2UK7jgWVz0365-v-aUss.zip?_nc_gid=FIa_sFO2MhVV8N8Yh4xKXA&_nc_oc=AdmdqTfN3Er2qDk2sE3sHUog-UxqVcQqooPo3Z2NL1Ta__6zX1uowQW_3ObAok0Ia6Q&ccb=10-5&oh=00_Afkemoao_jMhtk6aBl-1UQEOivNkYefodiK2y-RlU9xeGw&oe=6979C145&_nc_sid=e9ffdc"

echo ""
echo "[3/4] Downloading dev_queries_50k_0.zip..."
wget -O dev_queries_50k_0.zip "https://scontent-waw2-2.xx.fbcdn.net/m1/v/t6/An_C-TfENWHx64fmYrDn5UBFxPBCERGf2FP7zwEkP_CFfGql3Dy_5E1surDPIxPltN6NECBrbkMwKFcRYuNjNEssGQjgc5zFr-w_hpFlc7QMUkCS.zip?_nc_gid=Z32lFCwCg9r6MDYBeSj69Q&_nc_oc=AdlclorpqPki-W0hTK_ISqp_npl7C_-M6FxHLAEjP10xMGi8vpZtogWKTfztFrIubV8&ccb=10-5&oh=00_AfktW4qOkjnqC9iBiR1ZKb9tHkO1iAdbFbMmju-daQPCxA&oe=6979E6B2&_nc_sid=e9ffdc"

echo ""
echo "[4/4] Downloading test_queries_50k_0.zip..."
wget -O test_queries_50k_0.zip "https://scontent-waw2-2.xx.fbcdn.net/m1/v/t6/An_7Dx-upUUmai5U-F2syWGc7u-VxIazLPI9PwfBX2TvZpY4h9n6Md9adhcUy3K75B807xB_CRYjlm3oUmzDClfU1m14jHvCYJCyyWprqQn0-FB2.zip?_nc_gid=w3-PJbfBjGNOeSC3njDBcA&_nc_oc=AdkGQ5CnRu3rt5a_4eR_QAEHvujoSykojs86yt5g-qf0H95Qg8X9boqYo-LHf9iPkL4&ccb=10-5&oh=00_Afl1rY5Yzhh2dV8P0eIZ8bkIgCtfZsg52wW2AKEmCFT0gw&oe=6979BA36&_nc_sid=e9ffdc"

echo ""
echo "=================================================="
echo "Extracting files..."
echo "=================================================="

echo "Extracting train..."
unzip -q train_50k_0.zip -d train_temp
mv train_temp/*/* train/ 2>/dev/null || mv train_temp/* train/ 2>/dev/null || mkdir -p train && mv train_temp/*/* train/
rm -rf train_temp

echo "Extracting refs..."
unzip -q refs_50k_0.zip -d refs_temp  
mv refs_temp/*/* refs/ 2>/dev/null || mv refs_temp/* refs/ 2>/dev/null || mkdir -p refs && mv refs_temp/*/* refs/
rm -rf refs_temp

echo "Extracting dev_queries..."
unzip -q dev_queries_50k_0.zip -d queries_dev_temp
mv queries_dev_temp/*/* queries_dev/ 2>/dev/null || mv queries_dev_temp/* queries_dev/ 2>/dev/null || mkdir -p queries_dev && mv queries_dev_temp/*/* queries_dev/
rm -rf queries_dev_temp

echo "Extracting test_queries..."
unzip -q test_queries_50k_0.zip -d queries_test_temp
mv queries_test_temp/*/* queries_test/ 2>/dev/null || mv queries_test_temp/* queries_test/ 2>/dev/null || mkdir -p queries_test && mv queries_test_temp/*/* queries_test/
rm -rf queries_test_temp

echo ""
echo "Cleaning up zip files..."
rm -f *.zip

echo ""
echo "=================================================="
echo "Download complete! Checking structure..."
echo "=================================================="
echo ""
ls -la
echo ""
echo "Train images: $(ls train/ 2>/dev/null | wc -l)"
echo "Refs images: $(ls refs/ 2>/dev/null | wc -l)"
echo "Dev queries: $(ls queries_dev/ 2>/dev/null | wc -l)"
echo "Test queries: $(ls queries_test/ 2>/dev/null | wc -l)"
echo ""
echo "Don't forget to upload groundtruth CSV via WinSCP!"
echo "Place it at: data/dev_queries_groundtruth.csv"


export AOKVQA_DIR=./aokvqa/
mkdir -p ${AOKVQA_DIR}

curl -fsSL https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz | tar xvz -C ${AOKVQA_DIR}

export COCO_DIR=./aokvqa/coco/
mkdir -p ${COCO_DIR}

for split in train val test; do
    aria2c -x 16 -s 16 -o "${split}2017.zip" "http://images.cocodataset.org/zips/${split}2017.zip"
    unzip "${split}2017.zip" -d "${COCO_DIR}"
    rm "${split}2017.zip"
done

aria2c -x 16 -s 16 -o "annotations_trainval2017.zip" "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
unzip "annotations_trainval2017.zip" -d "${COCO_DIR}"
rm "annotations_trainval2017.zip"

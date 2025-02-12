set -e

if [ -d "ok_vqa_data" ]; then
    echo "Directory ok_vqa_data already exists. Exiting."
    exit 1
fi

mkdir -p ok_vqa_data
cd ok_vqa_data

curl -O https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip
curl -O https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip
if command -v aria2c &> /dev/null; then
    aria2c -x 16 -s 16 http://images.cocodataset.org/zips/train2014.zip
else
    curl -O http://images.cocodataset.org/zips/train2014.zip
fi


if ! command -v unzip &> /dev/null; then
    echo "unzip command not found. Please install unzip and try again."
    exit 1
fi

unzip mscoco_train2014_annotations.json.zip
unzip OpenEnded_mscoco_train2014_questions.json.zip
unzip train2014.zip

rm mscoco_train2014_annotations.json.zip
rm OpenEnded_mscoco_train2014_questions.json.zip
rm train2014.zip
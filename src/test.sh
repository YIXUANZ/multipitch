model_file=${model_path}/${model_name}/${model_name}_${type}.model
echo $model_file

python -u test.py \
--test_list=../filelists/test/TIMIT_Jin/TIMIT_Jin_same_gender_0.lst \
--model_file=$model_file \
--model_name=$model_name \

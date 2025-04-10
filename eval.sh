python llama_infer_2shots.py --model_name meta-llama/Llama3-8B-Instruct --task qp --test_data ./Public_Test_A.json --save_data result.json --icl
python process.py --task qp
python llama_infer_2shots.py --model_name meta-llama/Llama3-8B-Instruct --task cp --test_data ./result.json --save_data result.json --icl
python process.py --task cp

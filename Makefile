# This is a Python template Makefile, do modification as you want
#
# Project:
# Author:
# Email :

HOST = 127.0.0.1
PYTHONPATH="$(shell printenv PYTHONPATH):$(PWD)"

## incremental Roshambo experiments
run-1-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 16 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-2-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 16 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-3-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 16 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-4-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 16 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-5-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 17 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-6-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 17 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-7-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 17 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-8-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 17 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-9-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 18 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-10-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 18 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-11-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 18 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-12-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 18 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-13-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 19 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-14-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 19 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-15-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 19 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-16-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 19 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-17-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 20 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-18-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 20 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-19-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 20 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-20-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 20 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-21-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 21 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-22-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 21 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-23-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 21 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-24-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 21 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-25-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 22 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-26-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 22 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-27-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 22 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-28-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 22 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-29-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 23 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-30-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 23 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-31-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 23 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-32-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 23 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-33-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 24 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-34-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 24 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-35-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 24 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-36-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 24 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-37-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 25 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-37-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 25 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-39-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 25 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-40-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 25 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-41-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 26 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-42-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 26 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-43-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 26 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-44-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 26 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-45-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 27 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-46-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 27 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-47-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 27 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-48-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 27 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-49-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 28 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-50-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 28 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-51-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 28 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-52-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 28 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-53-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 29 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-54-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 29 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-55-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 29 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-56-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 29 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-57-inc_roshambo:
	python main.py --exemplars_memory 10 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 30 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-58-inc_roshambo:
	python main.py --exemplars_memory 100 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 30 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-59-inc_roshambo:
	python main.py --exemplars_memory 1000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 30 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-60-inc_roshambo:
	python main.py --exemplars_memory 2000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 30 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-61-inc_roshambo:
	python main.py --exemplars_memory 4000 --base_epochs 0 --inc_epochs 100 --base_knowledge "big" --seed 21 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-62-inc_roshambo:
	python main.py --exemplars_memory 4000 --base_epochs 0 --inc_epochs 100 --base_knowledge "big" --seed 22 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-63-inc_roshambo:
	python main.py --exemplars_memory 4000 --base_epochs 0 --inc_epochs 100 --base_knowledge "big" --seed 24 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-64-inc_roshambo:
	python main.py --exemplars_memory 4000 --base_epochs 0 --inc_epochs 100 --base_knowledge "big" --seed 28 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-65-inc_roshambo:
	python main.py --exemplars_memory 4000 --base_epochs 0 --inc_epochs 100 --base_knowledge "big" --seed 30 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"
run-66-inc_roshambo:
	python main.py --exemplars_memory 4000 --base_epochs 0 --inc_epochs 5 --base_knowledge "big" --seed 22 --base_classes 4 --save_path "/scratch2/jlungu/event_based_icarl_results" --data_dir "./data"

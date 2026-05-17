python run_benchmark.py --model 'deepseek/deepseek-v4-flash:free' --output results/pedro_experiment_deepseek --max-questions 60
cp  results/pedro_experiment_deepseek/answers.csv results/pedro_experiment_deepseek/answers_before_ner.csv
python evaluate_results.py --results-file results/pedro_experiment_deepseek/answers_before_ner.csv --model gpt-5.4
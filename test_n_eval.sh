echo "Inference is started."
python infer.py
echo "Evalution is started."
python eval.py > output/results.txt
echo "Check results in 'output/results.txt'."
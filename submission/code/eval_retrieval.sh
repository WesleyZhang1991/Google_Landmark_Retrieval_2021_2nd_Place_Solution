set -e

# First extract features+extract raw sims, then try different rerank
python base_landmark.py
python convert_online_offline.py
cp submission.pkl ../eval/
cd ../eval/
python eval_cbir.py

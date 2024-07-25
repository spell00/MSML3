

i=0
for groupkfold in 1 0
do
	for binary in 0 1
	do
		for concs in na,h
		do
            cuda=$((i%1)) # Divide by the number of gpus available
            /usr/bin/env /home/simonp/anaconda3/bin/python \
                ~/MSML3/msml/ml/train_models_gp3.py --n_features=-1 \
                --binary=$binary --model_name=xgboost --groupkfold=$groupkfold \
                --ovr=0 --min_mz=0 --min_rt=0 --max_mz=10000 --max_rt=320 \
                --threshold=0.0 --concs=$concs
        done
    done
done
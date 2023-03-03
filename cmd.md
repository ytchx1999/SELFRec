

```bash
# garden
python3 util/preprocess_amazon.py --data garden  --path_name Patio_Lawn_and_Garden
# instruments
python3 util/preprocess_amazon.py --data instruments  --path_name Musical_Instruments

# 2018
# magazine
python3 util/preprocess_amazon_2018.py --data magazine  --path_name Magazine_Subscriptions
# software
python3 util/preprocess_amazon_2018.py --data software  --path_name Software
# pantry
python3 util/preprocess_amazon_2018.py --data pantry  --path_name Prime_Pantry
# scientific
python3 util/preprocess_amazon_2018.py --data scientific  --path_name Industrial_and_Scientific
# beauty
python3 util/preprocess_amazon_2018.py --data beauty  --path_name Luxury_Beauty
# office
python3 util/preprocess_amazon.py --data office --path_name Office_Products


# ml100k
python3 util/preprocess_ml100k.py
```
python whitebox.py --rank 4 --decomposition tucker --eps 8/255 --attack BPDA
python whitebox.py --rank 6 --decomposition tucker --eps 8/255 --attack BPDA
python whitebox.py --rank 8 --decomposition tucker --eps 8/255 --attack BPDA
python whitebox.py --rank 10 --decomposition tucker --eps 8/255 --attack BPDA
python whitebox.py --rank 12 --decomposition tucker --eps 8/255 --attack BPDA
python whitebox.py --rank 16 --decomposition tucker --eps 8/255 --attack BPDA

python whitebox.py --rank 4 --decomposition tucker --eps 8/255 --attack BPDA --autoencoder True
python whitebox.py --rank 6 --decomposition tucker --eps 8/255 --attack BPDA --autoencoder True
python whitebox.py --rank 8 --decomposition tucker --eps 8/255 --attack BPDA --autoencoder True
python whitebox.py --rank 10 --decomposition tucker --eps 8/255 --attack BPDA --autoencoder True
python whitebox.py --rank 12 --decomposition tucker --eps 8/255 --attack BPDA --autoencoder True
python whitebox.py --rank 16 --decomposition tucker --eps 8/255 --attack BPDA --autoencoder True


python whitebox.py --rank 4 --decomposition cp --eps 8/255 --attack BPDA
python whitebox.py --rank 6 --decomposition cp --eps 8/255 --attack BPDA
python whitebox.py --rank 8 --decomposition cp --eps 8/255 --attack BPDA
python whitebox.py --rank 10 --decomposition cp --eps 8/255 --attack BPDA
python whitebox.py --rank 12 --decomposition cp --eps 8/255 --attack BPDA
python whitebox.py --rank 16 --decomposition cp --eps 8/255 --attack BPDA

python whitebox.py --rank 4 --decomposition cp --eps 8/255 --attack BPDA --autoencoder True
python whitebox.py --rank 6 --decomposition cp --eps 8/255 --attack BPDA --autoencoder True
python whitebox.py --rank 8 --decomposition cp --eps 8/255 --attack BPDA --autoencoder True
python whitebox.py --rank 10 --decomposition cp --eps 8/255 --attack BPDA --autoencoder True
python whitebox.py --rank 12 --decomposition cp --eps 8/255 --attack BPDA --autoencoder True
python whitebox.py --rank 16 --decomposition cp --eps 8/255 --attack BPDA --autoencoder True


python blackbox.py --cp_rank 6 --tucker_rank 6 --eps 8/255 --attack FGSM
python blackbox.py --cp_rank 4 --tucker_rank 4 --eps 8/255 --attack FGSM
python blackbox.py --cp_rank 8 --tucker_rank 8 --eps 8/255 --attack FGSM
python blackbox.py --cp_rank 10 --tucker_rank 10 --eps 8/255 --attack FGSM
python blackbox.py --cp_rank 12 --tucker_rank 12 --eps 8/255 --attack FGSM
python blackbox.py --cp_rank 16 --tucker_rank 16 --eps 8/255 --attack FGSM

python blackbox.py --cp_rank 4 --tucker_rank 4 --eps 8/255 --attack PGD
python blackbox.py --cp_rank 6 --tucker_rank 6 --eps 8/255 --attack PGD
python blackbox.py --cp_rank 8 --tucker_rank 8 --eps 8/255 --attack PGD
python blackbox.py --cp_rank 10 --tucker_rank 10 --eps 8/255 --attack PGD
python blackbox.py --cp_rank 12 --tucker_rank 12 --eps 8/255 --attack PGD
python blackbox.py --cp_rank 16 --tucker_rank 16 --eps 8/255 --attack PGD

python blackbox.py --cp_rank 4 --tucker_rank 4 --eps 8/255 --attack DeepFool
python blackbox.py --cp_rank 6 --tucker_rank 6 --eps 8/255 --attack DeepFool
python blackbox.py --cp_rank 8 --tucker_rank 8 --eps 8/255 --attack DeepFool
python blackbox.py --cp_rank 10 --tucker_rank 10 --eps 8/255 --attack DeepFool
python blackbox.py --cp_rank 12 --tucker_rank 12 --eps 8/255 --attack DeepFool
python blackbox.py --cp_rank 16 --tucker_rank 16 --eps 8/255 --attack DeepFool

python blackbox.py --cp_rank 4 --tucker_rank 4 --eps 8/255 --attack EOT
python blackbox.py --cp_rank 6 --tucker_rank 6 --eps 8/255 --attack EOT
python blackbox.py --cp_rank 8 --tucker_rank 8 --eps 8/255 --attack EOT
python blackbox.py --cp_rank 10 --tucker_rank 10 --eps 8/255 --attack EOT
python blackbox.py --cp_rank 12 --tucker_rank 12 --eps 8/255 --attack EOT
python blackbox.py --cp_rank 16 --tucker_rank 16 --eps 8/255 --attack EOT
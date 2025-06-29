# conda init
# conda activate cqc_former
python preproc.py
bash dat_to_ins_dataset.sh
bash dat_to_img_dataset.sh
python3 postproc.py
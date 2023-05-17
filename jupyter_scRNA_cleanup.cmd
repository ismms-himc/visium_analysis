source ~/.bashrc

sh $HIMC/singularity_images/scripts/minerva-jupyter-web-darwin-edit.sh \
-i $HIMC/singularity_images/himc_rpy2_singlecell_2.1.sif \
-n 10 \
-W 6:00 \
-q premium \
-P acc_himc \
-M 20000



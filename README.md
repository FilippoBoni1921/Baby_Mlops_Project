# Baby_Mlops_Project

## DVC
- dvc init
- dvc remote add -d storage gdrive://19JK5AFbqOBlrFVwDHjTrf9uvQFtS0954
- dvc add ../models/best-checkpoint.ckpt --file trained_model.dvc
- dvc push trained_model.dvc
Now the final step is to commit the dvc files to git. Run the following commands:
- git add dvcfiles/trained_model.dvc ../models/.gitignore
- git commit -m "Added trained model to google drive using dvc"
- git push
- dvc pull trained_model.dvc

# Ignora i file pesanti, NON la cartella né i .dvc
models/*.pt
!models/
!models/*.dvc
dvc add models/best_model.pt
git add models/best_model.pt.dvc .gitignore
git commit -m "Traccia best_model.pt con DVC"
dvc remote add -d myremote gs://nome-del-bucket/dvc-store
dvc remote modify myremote credentialpath /percorso/chiave.json
git push origin main  # o il branch attuale

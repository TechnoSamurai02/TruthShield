# Free-GPU training

Open `truthshield_v4_lightning_colab.ipynb` in Lightning Studio or Google Colab. Configure a persistent directory before running any download or training cell.

The notebook is deliberately provider-neutral:

- Lightning: point `PERSISTENT_ROOT` at the Studio drive.
- Colab: mount Google Drive and point `PERSISTENT_ROOT` inside it.
- Interrupted runs resume from `checkpoint-*` automatically.
- Dataset/model caches remain in persistent storage.
- Only final weights, model registry entries, calibrated policies, and locked reports belong in a release package.
- When third-party manipulation weights cannot legally be redistributed, the notebook builds a paired local fallback with split-isolated edit families and localization masks. It is a candidate only until calibrated and locked-tested.

Never place the locked test CSV in the training or tuning command. The final evaluation cell is the only cell that should read it.

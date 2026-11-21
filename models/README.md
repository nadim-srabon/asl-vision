# Model weights

Weights are **not stored in git**. Each model is one file, `models/<model_id>.pt` (plain fp32 `state_dict`):

| model_id | size |
|---|---|
| swin3d_t_1000.pt | 129 MB |
| r3d18_dualstream_1000.pt | 141 MB |
| geometry_transformer_1000.pt | 7.6 MB |
| stgcn_1000.pt | 5.4 MB |
| geometry_transformer_300.pt / _500.pt / _1200.pt | 7-8 MB each |

* **Locally** they are produced from the Kaggle outputs by `python scripts/prepare_artifacts.py`.
* **In the cloud** the app downloads only the selected file from the Hugging Face model repo named by
  `ASL_HF_REPO` (Streamlit secret), on first use. See the main README, "Model storage".

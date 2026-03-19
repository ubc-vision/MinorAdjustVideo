# Evaluation instructions

## Test configs

- **Folder:** `config/natural_motions/leveltestD/`
- **Sample config:** `config/natural_motions/leveltestD/swan/ABASTCVX_556_765/prompt_0000/motion_0001.yaml`  
  (Same structure as other leveltestD configs: `.../leveltestD/<object>/<id>/prompt_<n>/motion_<m>.yaml`.)

## Running evaluation

Run box optimization over the leveltestD set using the command below (adjust `--output-path` / `--validate_dirname` as needed, or try a few samples by adding `--val_start 0 --val_stop 2`):

```bash
optim_timestamp="2026-03-01/00-00-00"
python3 -m bin.Fwd_CmdTrailBlazer --config config/natural_motions/leveltestD --run_config config/run_opt_fwd.yaml --validate --validate_dirname evalD --val_model_name optim --shared_config config/common_shared.yaml --timestamp ${optim_timestamp} --lr 1e-2 --n_opt_iterations 5 --bb_deviate_lambda 0.1 --outside_bbox_loss_scale 10 --width 320 --height 320 --sigma_strength 0.03 --off_normalize_mask --set_global_deterministic 
```

Run box trailblazer over the leveltestD set 
```bash
tbl_timestamp="2026-03-01/00-00-10"
python3 -m bin.Fwd_CmdTrailBlazer --config config/natural_motions/leveltestD --run_config '' --validate --validate_dirname evalD --val_model_name trailblazer_origin --shared_config config/common_shared.yaml --timestamp ${tbl_timestamp}  --width 320 --height 320 
```


Outputs are written under `output/<validate_dirname>/<model_type>/natural_motions/leveltestD/...` (videos, gifs, etc.).  

TODO: Metrics / aggregation scripts can be documented here when added.

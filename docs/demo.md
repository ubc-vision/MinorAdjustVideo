# Demo instructions

Assumes [Setup](../README.md#setup) and ZeroScope model have been executed.

## Box optimization (teaser)

```bash
python3 -m bin.Fwd_CmdTrailBlazer --config config/box_opt_teaser/leveltestD/swan/motion_0001.yaml --run_config config/run_opt_fwd.yaml --validate --validate_dirname teaser --val_model_name "optim" --shared_config config/common_shared.yaml --output-path output --bb_deviate_lambda 0.1 --outside_bbox_loss_scale 10 --width 320 --height 320 --set_global_deterministic
```

## TrailBlazer (teaser, same config)

```bash
python3 -m bin.Fwd_CmdTrailBlazer --config config/box_opt_teaser/leveltestD/swan/motion_0001.yaml --validate --validate_dirname teaser --val_model_name "trailblazer_origin" --shared_config config/common_shared.yaml --output-path output --width 320 --height 320 --set_global_deterministic
```

## LeveltestD demo examples

Configs under `config/box_opt_teaser/leveltestD/`:

| Object  | Config path |
|---------|----------------------------------------------------------------|
| Swan    | `config/box_opt_teaser/leveltestD/swan/motion_0001.yaml`       |
| Horse   | `config/box_opt_teaser/leveltestD/horse/motion_0035.yaml`      |
| Insect  | `config/box_opt_teaser/leveltestD/insect/motion_0018.yaml`     |

Use the same command pattern as above; replace `--config ...` with the path above. For **box optim** add `--run_config config/run_opt_fwd.yaml --val_model_name "optim"` and the extra flags; for **TrailBlazer** use `--val_model_name "trailblazer_origin"` without `--run_config`.

## Complex motions

Configs under `config/box_opt_teaser/complex_motions/`:

| Motion              | Config path |
|---------------------|----------------------------------------------------------------|
| U-turn (horse)      | `config/box_opt_teaser/complex_motions/uturn_horse/motion_0001.yaml` |
| Stationary→move (ant) | `config/box_opt_teaser/complex_motions/stationary_to_move_ant/motion_0001.yaml` |
| Zigzag (horse)     | `config/box_opt_teaser/complex_motions/zigzag_horse/motion_0000.yaml` |

Same command pattern: swap in the desired `--config ...` path for box optim or TrailBlazer as above.

## Config locations

- **Teaser (leveltestD):** `config/box_opt_teaser/leveltestD/` — swan, horse, insect demos.
- **Complex motions:** `config/box_opt_teaser/complex_motions/` — uturn, stationary_to_move, zigzag.

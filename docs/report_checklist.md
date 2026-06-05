# Submission Checklist

- [ ] Report front page contains all names, student IDs, and work split.
- [ ] Report includes public GitHub repository URL.
- [ ] Report includes permanent cloud link for model weights and extraction code if needed.
- [ ] Report metadata has been filled with `scripts/report/fill_report_metadata.py`.
- [ ] Task 1 includes object A/B/C generation details and final fused video frames.
- [ ] Task 1 compares multi-view reconstruction, text-to-3D, and image-to-3D on geometry, texture, and runtime.
- [ ] Task 1 explains the representation unification path: 2DGS and AIGC outputs are exported to textured meshes and fused in Blender.
- [ ] Task 2 includes ACT environment-B training and A/B/C joint training.
- [ ] Task 2 includes zero-shot D evaluation using success rate or Action L1.
- [ ] WandB or SwanLab loss curves are exported into the report.
- [ ] Hyperparameter table includes architecture, batch size, learning rate, optimizer, steps/epochs, and loss.
- [ ] README has copy-pasteable setup, train, and test commands.
- [ ] Final model weights are packaged under `weights/` before cloud upload.
- [ ] `python scripts/report/check_submission_ready.py --json results/submission_readiness.json` passes without failures.
- [ ] Local backup bundle has been built with `scripts/report/package_submission_bundle.sh`.

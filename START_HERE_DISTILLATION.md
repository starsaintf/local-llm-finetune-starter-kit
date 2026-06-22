# Start Here: Distillation for Beginners

Start with the noob-friendly guide here:

`docs/distillation_quickstart.md`

That guide walks through the full sequence in plain English:

1. Clone the repo.
2. Install the Python packages.
3. Check that your GPU is visible.
4. Create a prompt file.
5. Generate teacher answers.
6. Filter the teacher answers.
7. Split train and eval data.
8. Train the student model.
9. Watch the loss curve.
10. Test with real prompts.
11. Merge the adapter if needed.

Use this root file as the obvious entry point if you are new.

The short mental model:

```text
prompts -> teacher model -> teacher answers -> clean dataset -> student fine-tune -> better small model
```

Do not start by generating thousands of examples.

Start with 20 to 50 examples.

Make sure the workflow runs.

Then scale up.
